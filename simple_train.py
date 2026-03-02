# %%
import functools
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType, reshard

import equinox as eqx

mesh = jax.make_mesh(
    (2, 4, 4),
    ("data", "fsdp", "model"),
    (AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
)
jax.set_mesh(mesh)
key = jax.random.key(42)


class ModelConfig(NamedTuple):
    n_layers: int = 4
    hidden_dim: int = 32
    intermediate_dim: int = 32
    vocab_size: int = 64
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    num_attention_heads: int = 16
    head_dim: int = 16
    num_kv_heads: int = 4
    sliding_window: int = 128
    sliding_window_ratio: int = 1  # 1:1


fsdp_remat = functools.partial(  # Don't save reshard (all-gathered weights)
    jax.remat,
    policy=jax.checkpoint_policies.save_anything_except_these_names("reshard"),
    static_argnums=(2,),  # train arg (self=0, x/tokens=1, train=2)
)


class RMSNorm(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    scale: jax.Array
    eps: float

    def __init__(self, config: ModelConfig, eps=1e-5):
        self.config = config
        self.scale = jnp.ones((config.hidden_dim,), dtype=jnp.float32)
        self.eps = eps

    def __call__(self, x):
        origin_dtype, x = x.dtype, jnp.astype(x, jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x = jnp.astype(x * self.scale, origin_dtype)
        return x


class MLPBlock(eqx.Module):
    """
    Interleaved swiglu with TP + FSDP
    """

    config: ModelConfig = eqx.field(static=True)
    w13: jax.Array
    b13: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(self, key, config: ModelConfig):
        self.config = config
        keys = jax.random.split(key, 4)
        D, F = config.hidden_dim, config.intermediate_dim
        self.w13 = jax.random.normal(keys[0], (F * 2, D)) * 0.05
        self.b13 = jax.random.normal(keys[1], (F * 2,)) * 0.05
        self.w2 = jax.random.normal(keys[2], (D, F)) * 0.05
        self.b2 = jax.random.normal(keys[3], (D,)) * 0.05

    def swiglu(self, x):
        x_glu, x_lin = jnp.split(x, 2, axis=-1)
        x_glu = x_glu * jax.nn.sigmoid(
            jnp.clip(x_glu, max=self.config.swiglu_limit) * self.config.swiglu_alpha
        )
        return x_glu * (x_lin + 1)

    @fsdp_remat
    def __call__(self, x, train=True):
        seq_out = P("model", None) if train else P(None, None)
        # all-gather fsdp weights before compute
        w13 = reshard(self.w13, P("model", None))
        w2 = reshard(self.w2, P(None, "model"))
        # sanity check: sharded on seq
        jax.lax.with_sharding_constraint(x, seq_out)
        # column-parallel, partial sum
        x = jnp.matmul(x, w13.T, out_sharding=P(None, "model")) + self.b13
        x = self.swiglu(x)
        # row-parallel: reduce-scatter, for seq parallel
        x = jnp.matmul(x, w2.T, out_sharding=seq_out) + self.b2
        return x


class AttentionBlock(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    b_o: jax.Array

    def __init__(self, key, config: ModelConfig):
        self.config = config
        keys = jax.random.split(key, 5)
        nq, nkv, hd, D = (
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.hidden_dim,
        )
        self.w_q = jax.random.normal(keys[0], (nq * hd, D)) * 0.05
        self.w_k = jax.random.normal(keys[1], (nkv * hd, D)) * 0.05
        self.w_v = jax.random.normal(keys[2], (nkv * hd, D)) * 0.05
        self.w_o = jax.random.normal(keys[3], (D, nq * hd)) * 0.05
        self.b_o = jax.random.normal(keys[4], (D,)) * 0.05

    @fsdp_remat
    def __call__(self, x, train=True):
        seq_out = P("model", None) if train else P(None, None)
        nq, nkv, hd = (
            self.config.num_attention_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        )

        # all-gather fsdp, keep heads on model
        w_q = reshard(self.w_q, P("model", None))
        w_k = reshard(self.w_k, P("model", None))
        w_v = reshard(self.w_v, P("model", None))
        w_o = reshard(self.w_o, P(None, "model"))

        # column-parallel: all-gather seq, shard heads on model
        q = jnp.matmul(x, w_q.T, out_sharding=P(None, "model")).reshape(-1, nq, hd)
        k = jnp.matmul(x, w_k.T, out_sharding=P(None, "model")).reshape(-1, nkv, hd)
        v = jnp.matmul(x, w_v.T, out_sharding=P(None, "model")).reshape(-1, nkv, hd)

        o = jax.nn.dot_product_attention(q, k, v, scale=hd**-0.5, is_causal=True)
        o = o.reshape(-1, nq * hd)

        # row-parallel: reduce-scatter for seq parallel
        return jnp.matmul(o, w_o.T, out_sharding=seq_out) + self.b_o


class TransformerBlock(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    norm: RMSNorm
    attn: AttentionBlock
    mlp: MLPBlock

    def __init__(self, key, config: ModelConfig):
        mlp_key, attn_key = jax.random.split(key)
        self.config = config
        self.norm = RMSNorm(config)
        self.attn = AttentionBlock(attn_key, config)
        self.mlp = MLPBlock(mlp_key, config)

    def __call__(self, x, train=True):
        x = x + self.norm(x)
        x = x + self.attn(x, train)
        x = x + self.mlp(x, train)
        return x


class Model(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    layers: list[TransformerBlock]
    w_embed: jax.Array
    w_unembed: jax.Array

    def __init__(self, key, config: ModelConfig):
        self.config = config
        self.layers = []
        for _ in range(self.config.n_layers):
            key, subkey = jax.random.split(key)
            self.layers.append(TransformerBlock(subkey, config))

        embedkey, unembedkey = jax.random.split(key, 2)

        self.w_embed = (
            jax.random.normal(embedkey, (config.vocab_size, config.hidden_dim)) * 0.05
        )
        self.w_unembed = (
            jax.random.normal(unembedkey, (config.vocab_size, config.hidden_dim)) * 0.05
        )

    @fsdp_remat
    def __call__(self, tokens, train=True):
        w_embed = reshard(self.w_embed, P("model", None))
        w_unembed = reshard(self.w_unembed, P(None, "model"))
        # tokens[seq] -> x[seq@model, d_model] -> x[seq, d_model] (all-reduce)
        x = w_embed.at[tokens].get(out_sharding=P(None, None))
        for layer in self.layers:
            x = layer(x, train)
        x = jnp.matmul(x, w_unembed.T, out_sharding=P(None, None))
        return x


def shard(tree):
    def shard_param(key, param):
        if param is None:
            return None
        param_name = key[-1].name
        if param_name in {"w13", "w_q", "w_k", "w_v"}:
            # column-parallel: (out_features@model, in_features@fsdp)
            return jax.device_put(param, P("model", "fsdp"))
        elif param_name in {"w2", "w_unembed", "w_o"}:
            return jax.device_put(param, P("fsdp", "model"))  # row-parallel
        elif param_name == "w_embed":
            return jax.device_put(param, P(("fsdp", "model"), None))
        elif param_name == "b13":
            return jax.device_put(param, P("model"))
        elif param_name in {"b2", "b_o", "scale"}:
            return jax.device_put(param, P(None))
        else:
            raise ValueError(f"unexpected param: {param_name}")

    params, static = eqx.partition(tree, eqx.is_array)
    sharded = jax.tree.map_with_path(shard_param, params)
    return eqx.combine(sharded, static)


key, model_key, data_key = jax.random.split(key, 3)
config = ModelConfig()
model = Model(model_key, config)

# dummy dataset: random token ids embedded as one-hot -> projected down
k1, k2 = jax.random.split(data_key)
batch_size, seq_len = 16, 32
tokens = jax.random.randint(k1, (batch_size, seq_len + 1), 0, config.vocab_size)
# simple learned-ish embedding: just random vectors per token (dummy)
tokens = jax.device_put(tokens, P("data", None))


out = jax.vmap(model, in_axes=(0, None))(tokens[:, :-1], True)
assert not jnp.isnan(out).any()

model = shard(model)

out_shard = jax.vmap(model, in_axes=(0, None))(tokens[:, :-1], True)
assert not jnp.isnan(out_shard).any()
assert jnp.allclose(out, out_shard, rtol=1e-5, atol=1e-5)

# %%


class AdamState(NamedTuple):
    mu: Model
    nu: Model
    step: int


def init_optimizer(model):
    params, _ = eqx.partition(model, eqx.is_array)
    mu = jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), params)
    nu = jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), params)
    return AdamState(mu, nu, step=0)


optim = init_optimizer(model)
optim = shard(optim)

# %%,


class TrainConfig(NamedTuple):
    num_steps: int = 30
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    model_config: ModelConfig = ModelConfig()


def cross_entropy_loss(logits, targets):
    """Standard cross-entropy loss. logits: (batch, seq, vocab), targets: (batch, seq)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


@eqx.filter_jit
def train_step(model, optim, tokens, train_config: TrainConfig = TrainConfig()):
    _, lr, b1, b2, eps, _ = train_config

    def loss_fn(model, tokens):
        input_ids, targets = tokens[:, :-1], tokens[:, 1:]
        logits = jax.vmap(model, in_axes=(0, None))(input_ids, True)
        return cross_entropy_loss(logits, targets)

    def adam_step(grad, optim):
        mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, optim.mu, grad)
        nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g**2, optim.nu, grad)
        return AdamState(mu, nu, optim.step + 1)

    def model_update(model, optim):
        params, static = eqx.partition(model, eqx.is_array)

        mu_hat = jax.tree.map(lambda m: m / (1 - b1**optim.step), optim.mu)
        nu_hat = jax.tree.map(lambda v: v / (1 - b2**optim.step), optim.nu)
        params = jax.tree.map(
            lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, mu_hat, nu_hat
        )
        model = eqx.combine(params, static)
        return model

    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, tokens)
    optim = adam_step(grad, optim)
    model = model_update(model, optim)
    return model, optim, loss


def train(key, config: TrainConfig):
    key, model_key = jax.random.split(key, 2)
    model = Model(model_key, config.model_config)
    optim = init_optimizer(model)

    batch_size, seq_len = 16, 32
    assert seq_len + 1 <= config.model_config.vocab_size
    seq = jnp.arange(0, seq_len + 1)
    tokens = jnp.stack([jnp.roll(seq, i) for i in range(batch_size)])
    tokens = jax.device_put(tokens, P("data", None))

    for i in range(config.num_steps):
        model, optim, loss = train_step(model, optim, tokens)
        print(f"step: {i}, loss: {loss:.4f}")

    return model


key = jax.random.key(0)
model = train(key, TrainConfig())
"""
step: 0, loss: 4.8094
step: 1, loss: 4.3392
step: 2, loss: 3.8934
step: 3, loss: 3.4697
step: 4, loss: 3.0677
step: 5, loss: 2.6879
step: 6, loss: 2.3307
step: 7, loss: 1.9965
step: 8, loss: 1.6852
step: 9, loss: 1.3974
step: 10, loss: 1.1350
step: 11, loss: 0.9006
step: 12, loss: 0.6964
step: 13, loss: 0.5233
step: 14, loss: 0.3813
step: 15, loss: 0.2700
step: 16, loss: 0.1869
step: 17, loss: 0.1276
step: 18, loss: 0.0864
step: 19, loss: 0.0585
step: 20, loss: 0.0400
step: 21, loss: 0.0279
step: 22, loss: 0.0199
step: 23, loss: 0.0146
step: 24, loss: 0.0109
step: 25, loss: 0.0084
step: 26, loss: 0.0066
step: 27, loss: 0.0052
step: 28, loss: 0.0042
step: 29, loss: 0.0034
"""

# %%

seq = jnp.array([0])
for i in range(64):
    seq = jax.device_put(seq)
    logits = model(seq, False)
    next_tok = jnp.argmax(logits, -1, keepdims=True)[-1]
    seq = jnp.concatenate([seq, next_tok])

print(seq)

"""
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
28 29 30 31 32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
23 24 25 26 27 28 29 30 31
"""
