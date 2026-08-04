import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  gru_act: str = 'tanh'
  train_noise_std: float = 0.0  # post-gate Gaussian noise on deter, TRAINING only
  require_relu_deter: bool = True  # enforce gru_act='relu' so recorded wake AND
  #   dream deter share one (non-negative) activation space — see below

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    # gru_act gates the deter candidate (see _core), so it alone determines the
    # deter activation space for BOTH wake (observe) and dream (imagine): relu ->
    # deter >= 0, tanh -> signed. If a dream run silently defaults to tanh while
    # wake was recorded with relu, the two live in different spaces and every
    # downstream wake-vs-dream comparison (cosine SW distance, isomap) is garbage.
    # Guard it at construction so no rollout is wasted. Relax with
    # `--dyn.rssm.require_relu_deter False` if a non-relu run is truly intended.
    if self.require_relu_deter:
      assert self.gru_act == 'relu', (
          f"gru_act={self.gru_act!r} but require_relu_deter=True: wake and dream "
          f"deter must be in relu (non-negative) space to be comparable. Pass "
          f"gru_act=relu (e.g. --dyn.rssm.gru_act relu), or set "
          f"--dyn.rssm.require_relu_deter False to allow another activation.")
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    # carry is the recurrent state threaded across timesteps via jax scan:
    # {deter: h, stoch: z}. Initialized to zeros at sequence start.
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    # Extract carry from the last timestep of a sequence, so we can
    # resume the recurrent state across replay chunks / training batches.
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False,
              mask=None):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      m = mask if mask is not None else False
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training, mask=m)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      if mask is None:
        mask_input = jnp.zeros(reset.shape, dtype=bool)
      else:
        mask_input = mask
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, inputs[0], inputs[1], inputs[2], training,
              mask=inputs[3]),
          carry, (tokens, action, reset, mask_input),
          unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training, mask=False):
    # Posterior update: z ~ Cat(f(h, encoder_tokens)).
    # Uses both the new deterministic state h and real observations to
    # compute the posterior distribution over z.
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action, training=training)  # new h from GRU
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    post_logit = self._logit('obslogit', x)  # posterior logits from h + obs
    prior_logit = self._prior(deter)
    # On masked steps, use prior (dynamics-only) instead of posterior
    mask_bc = jnp.reshape(mask, (*deter.shape[:-1], 1, 1)) if jnp.ndim(mask) > 0 else mask
    logit = jnp.where(mask_bc, prior_logit, post_logit)
    # Sample z: 32x32 categorical with straight-through one-hot gradients
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)  # thread (h, z) to next timestep
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    # Prior update: z ~ Cat(f(h)) — no observations, only the dynamics.
    # Used during imagination / dreaming to predict z from h alone.
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb, training=training)  # new h
      logit = self._prior(deter)  # prior logits from h only (no obs)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))  # sample z
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training, mask=None):
    metrics = {}
    carry, entries, feat = self.observe(
        carry, tokens, acts, reset, training, mask=mask)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def rollout_loss(self, entries, feat, acts, reset, training, rollout_k):
    """Multi-step prior rollout loss.

    For each timestep t, rolls forward K steps using _core() + _prior()
    with real actions, comparing each rolled prior to sg(posterior[t+k]).
    """
    B, T = reset.shape
    posterior_logits = feat['logit']  # (B, T, stoch, classes)
    total_kl = jnp.zeros((B, T))
    count = jnp.zeros((B, T))
    metrics = {}

    # Precompute cumulative reset for cross-episode boundary detection
    reset_cumsum = jnp.cumsum(reset.astype(f32), axis=1)

    # Start from posterior states at each timestep
    rolled_deter = entries['deter']  # (B, T, D)
    rolled_stoch = entries['stoch']  # (B, T, S, C)

    for k in range(1, rollout_k + 1):
      valid_T = T - k
      if valid_T <= 0:
        break

      # Actions for this step: prevact[t+k] transitions state[t+k-1] -> state[t+k]
      step_acts = {key: v[:, k:k + valid_T] for key, v in acts.items()}

      # Embed actions (merge B*valid_T for DictConcat, then use directly)
      flat_acts = {key: v.reshape((-1, *v.shape[2:])) for key, v in step_acts.items()}
      actemb = nn.DictConcat(self.act_space, 1)(flat_acts)

      # Prepare rolled states: (B*valid_T, ...)
      rd = rolled_deter[:, :valid_T].reshape(-1, rolled_deter.shape[-1])
      rs = rolled_stoch[:, :valid_T].reshape(-1, *rolled_stoch.shape[2:])

      # Roll forward one step: GRU + prior
      new_deter = self._core(rd, rs, actemb, training=training)
      prior_logit = self._prior(new_deter)
      new_stoch = nn.cast(self._dist(prior_logit).sample(seed=nj.seed()))

      # Reshape back to (B, valid_T, ...)
      new_deter = new_deter.reshape(B, valid_T, -1)
      prior_logit = prior_logit.reshape(B, valid_T, self.stoch, self.classes)
      new_stoch = new_stoch.reshape(B, valid_T, self.stoch, self.classes)

      # Target: posterior logits at landing timestep t+k
      target_logit = posterior_logits[:, k:k + valid_T]

      # KL: prior tries to match stopped posterior (same direction as dyn loss)
      kl = self._dist(sg(target_logit)).kl(self._dist(prior_logit))
      if self.free_nats:
        kl = jnp.maximum(kl, self.free_nats)

      # Mask rollouts crossing episode boundaries
      # cross_reset[i] = any(reset[i+1], ..., reset[i+k]) for start position i
      cross_reset = (reset_cumsum[:, k:k + valid_T] - reset_cumsum[:, :valid_T]) > 0
      valid_mask = (~cross_reset).astype(f32)
      kl = kl * valid_mask

      # Accumulate into (B, T) tensor, zero-padded at trailing positions
      total_kl = total_kl.at[:, :valid_T].add(kl)
      count = count.at[:, :valid_T].add(valid_mask)

      # Per-depth metric
      metrics[f'rollout_dyn/depth_{k}'] = kl.sum() / valid_mask.sum().clip(1)

      # Carry rolled state forward for next depth
      rolled_deter = new_deter
      rolled_stoch = new_stoch

    # Average over depths (avoid div by zero)
    rollout_dyn = jnp.where(count > 0, total_kl / count, 0.0)
    return rollout_dyn, metrics

  def _core(self, deter, stoch, action, training=False):
    # Block-wise GRU: computes new h from (old_h, old_z, action).
    # Splits h into 8 blocks and applies independent gated updates per block.
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = nn.act(self.gru_act)(reset * cand)
    update = jax.nn.sigmoid(update - 1)  # bias toward 0 → default is to remember
    deter = update * cand + (1 - update) * deter  # GRU update: blend new candidate with old h
    # Train-noise injection: additive Gaussian on the post-gate deter, mirroring
    # pRNN's internal-noise sleep drive but placed AFTER the gated blend so it
    # perturbs the state that actually carries forward (matching the dream-seed
    # test distribution). Training only — forces the network to learn dynamics
    # that contract off-manifold deter back onto the wake manifold, the
    # "sufficient conditions for offline reactivation" mechanism.
    if training and self.train_noise_std:
      deter = deter + self.train_noise_std * jax.random.normal(
          nj.seed(), deter.shape, dtype=deter.dtype)
    return deter

  def _prior(self, feat):
    # Prior: predicts z logits from h alone (no observations).
    # Used during imagination and as the KL target during training.
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def post_from_deter(self, deter, tokens):
    # Posterior z logits from an arbitrary (e.g. injected / noise) deter plus
    # encoder tokens, WITHOUT the _core advance that _observe runs first. Reuses
    # the trained posterior obslayer params (identical sub-names to _observe), so
    # this MUST stay in sync with the posterior path in _observe. Used to seed
    # dreams for the four-condition seeding ablation (see
    # docs/training_and_dream_loop.md sections 11 and 13).
    deter, tokens = nn.cast((deter, tokens))
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    return self._logit('obslogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    # Categorical distribution over z with 1% uniform mixing (unimix=0.01)
    # and straight-through one-hot gradients for discrete sampling.
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw): #enc_space comes through here
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False,
               return_activations=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape
    acts = {}

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
        if return_activations:
          acts[f'enc/mlp{i}'] = x.reshape((*bshape, *x.shape[1:]))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
        if return_activations:
          acts[f'enc/cnn{i}'] = x.reshape(
              (*bshape, *x.shape[1:-3], x.shape[-3] * x.shape[-2] * x.shape[-1]))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    if return_activations:
      acts['enc/tokens'] = tokens
    entries = {}
    if return_activations:
      return carry, entries, tokens, acts
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
