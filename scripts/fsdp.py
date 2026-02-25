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
    ("batch", "fsdp", "model"),
    (AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
)
jax.set_mesh(mesh)
key = jax.random.key(42)


class ModelConfig(NamedTuple):
    n_layers: int
    d_model: int
    d_intermediate: int
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0


fsdp_remat = functools.partial(  # Don't save reshard (all-gathered weights)
    jax.remat,
    policy=jax.checkpoint_policies.save_anything_except_these_names("reshard"),
)


class MLP(eqx.Module):
    """
    Interleaved swiglu with TP + FSDP
    """

    config: ModelConfig
    w13: jax.Array
    b13: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(self, key, config: ModelConfig):
        self.config = config
        keys = jax.random.split(key, 4)
        D, F = config.d_model, config.d_intermediate
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
    def __call__(self, x):
        # all-gather fsdp weights before compute
        w13 = reshard(self.w13, P("model", None))
        w2 = reshard(self.w2, P(None, "model"))
        # column-parallel, partial sum
        x = jnp.matmul(x, w13.T, out_sharding=P(None, "model")) + self.b13
        x = self.swiglu(x)
        # row-parallel: all-reduce over model
        x = jnp.matmul(x, w2.T, out_sharding=P(None, None)) + self.b2
        return x


class Model(eqx.Module):
    config: ModelConfig
    layers: list[MLP]
    w_out: jax.Array
    b_out: jax.Array

    def __init__(self, key, config: ModelConfig):
        self.config = config
        self.layers = []
        for _ in range(self.config.n_layers):
            key, subkey = jax.random.split(key)
            self.layers.append(MLP(key, config))

        wkey, bkey = jax.random.split(key, 2)
        self.w_out = (
            jax.random.normal(wkey, (self.config.d_model, self.config.d_model)) * 0.05
        )
        self.b_out = jax.random.normal(bkey, (self.config.d_model,)) * 0.05

    @fsdp_remat
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        w_out = reshard(self.w_out, P("model", None))
        x = jnp.matmul(x, w_out.T, out_sharding=P(None, None)) + self.b_out
        return x


def shard(tree):
    def shard_param(key, param):
        if param is None:
            return
        param_name = key[-1].name
        if param_name in ("w13", "w_out"):
            # column-parallel: note weights are transposed
            return jax.device_put(param, P("model", "fsdp"))
        elif param_name == "w2":
            return jax.device_put(param, P("fsdp", "model"))  # # row-parallel
        elif param_name == "b13":
            return jax.device_put(param, P("model"))
        elif param_name in ("b2", "b_out"):
            return jax.device_put(param, P(None))
        else:
            raise ValueError(f"unexpected param: {param_name}")

    params, static = eqx.partition(tree, eqx.is_array)
    sharded = jax.tree.map_with_path(shard_param, params)
    return eqx.combine(sharded, static)


key, model_key, act_key = jax.random.split(key, 3)
config = ModelConfig(n_layers=5, d_model=32, d_intermediate=64)
model = Model(model_key, config)
x = jax.random.normal(act_key, (16, 512, 32))
x = jax.device_put(x, P("batch", None, None))

out = jax.vmap(model)(x)
assert not jnp.isnan(out).any()

print("\n---")
print("model sharding")
print("before", jax.typeof(model.layers[0].w13))
model = shard(model)
print("after", jax.typeof(model.layers[0].w13))

out_shard = jax.vmap(model)(x)
assert not jnp.isnan(out_shard).any()
assert jnp.allclose(out, out_shard, rtol=1e-5, atol=1e-5)

# %%


@eqx.filter_jit
def train_step(model, x):
    def loss_fn(model, x):
        logits = jax.vmap(model)(x)
        return jnp.mean(jnp.sum(logits, axis=-1))  # dummy

    def adam_update(param, grad, mu, nu):
        mu[...] = ()

    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x)


# %%


class AdamState(NamedTuple):
    mu: Model
    nu: Model
    step: int


def init_optimizer(model):
    params, _ = eqx.partition(model, eqx.is_array)
    mu = jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), params)
    nu = jax.tree.map(lambda p: jnp.zeros(p.shape, p.dtype), params)
    return Adam(mu, nu, step=0)


optim = init_optimizer(model)

print("\n---")
print("optimizer state", jax.tree.structure(optim))
print("before", jax.typeof(optim.mu.layers[0].w13))
optim = shard(optim)
print("after", jax.typeof(optim.mu.layers[0].w13))

# %%
