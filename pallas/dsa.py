# %%
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.experimental.pallas.tpu_sc as plsc

INTERPRET = jax.default_backend() == "cpu"
if INTERPRET:
    INTERPRET = True

mesh = jax.make_mesh((8,), ("x",), (jax.sharding.AxisType.Explicit,))
jax.set_mesh(mesh)


# %%
