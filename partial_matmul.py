# %%
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

mesh = jax.make_mesh(
    (4, 2),
    ("data", "model"),
    (jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)
jax.set_mesh(mesh)


@jax.jit
@jax.shard_map(
    in_specs=(jax.P("data", "model"), jax.P("model", None)),
    out_specs=jax.P("data", unreduced={"model"}),
)
def partial_matmul(x_shard, w_shard):
    return x_shard @ w_shard  # each device holds a partial sum along 'model'


a = jax.random.normal(jax.random.key(0), (256, 512))
b = jax.random.normal(jax.random.key(1), (512, 128))
a = jax.device_put(a, jax.P("data", "model"))
b = jax.device_put(b, jax.P("model", None))
c = partial_matmul(a, b)

c.sharding.spec
# %%
