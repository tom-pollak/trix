# %%
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental import checkify

if jax.default_backend() == "cpu":
    pltpu.set_tpu_interpret_mode(pltpu.InterpretParams(num_cores_or_threads=2))

mesh = jax.make_mesh((8,), ("x",), (jax.sharding.AxisType.Explicit,))
jax.set_mesh(mesh)
assert jax.device_count() == 8

# %%

inp = jax.random.uniform(jax.random.key(0), (512, 512))
inp = jax.device_put(inp, jax.P("x", None))

# %%


def local_barrier(left, right, double_barrier=True):
    def do_barrier(sem):
        for neighbor in left, right:
            pltpu.semaphore_signal(
                sem,
                inc=1,
                device_id=(neighbor,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(sem, 2)

    barrier_sem = pltpu.get_barrier_semaphore()
    do_barrier(barrier_sem)

    if double_barrier:
        pl.run_scoped(
            lambda second_barrier: do_barrier(second_barrier),
            pltpu.SemaphoreType.REGULAR,
        )


@jax.jit
@partial(
    jax.shard_map,
    in_specs=jax.P("x", None),
    out_specs=jax.P("x", None),
    check_vma=False,
)
# @pl.core_map(
#     mesh=pltpu.create_tensorcore_mesh("core", num_cores=2),
#     compiler_params=pltpu.CompilerParams(collective_id=0),
# )
def reduce_scatter(x_s):
    inner_block = (8, 128)
    assert np.prod(x_s.shape) % (2 * np.prod(inner_block)) == 0, (
        "x must be shape contrained"
    )

    origin_shape = x_s.shape
    x_s_lr = x_s.reshape(2, -1, *inner_block)
    n_iters = x_s_lr.shape[0]

    x_s_lr = jax.new_ref(x_s_lr)
    partial_x2_lr = jax.new_ref(jnp.empty((2, *x_s_lr.shape), x_s_lr.dtype))

    device_idx = jax.lax.axis_index("x")
    num_devices = jax.lax.axis_size("x")

    def prologue():
        left = jax.lax.rem(device_idx - 1, num_devices)
        right = jax.lax.rem(device_idx + 1, num_devices)
        local_barrier(left, right)
        pltpu.sync_copy(x_s_lr, partial_x2_lr.at[0])

    @pl.with_scoped(
        capacity_sem=pltpu.SemaphoreType.REGULAR,
        send_sem=pltpu.SemaphoreType.DMA,
        recv_sem=pltpu.SemaphoreType.DMA,
    )
    def pipeline(capacity_sem, send_sem, recv_sem):
        loop_idx = pl.program_id(0)
        working_slot = jax.lax.rem(loop_idx, 2)
        receiving_slot = jax.lax.rem(loop_idx + 1, 2)

        left_or_right = pl.program_id(1)  # left=0,right=1
        neighbor_id = jax.lax.rem(
            device_idx + left_or_right * 2 - 1, num_devices
        )  # left=device_id-1, right=device_id+1

        x_hbm_slc = x_s_lr.at[left_or_right]
        partial_x2 = partial_x2_lr.at[:, left_or_right]

        # trigger copy into receiving_slot
        pltpu.semaphore_signal(capacity_sem, inc=1, device_id=neighbor_id)

        def local_add(x_vmem, partial_vmem):
            partial_vmem[...] = x_vmem[...]  # should_accumulate_out

        inner_spec = pl.BlockSpec(inner_block, lambda i: (i, 0, 0))
        accum_pipeline = pltpu.emit_pipeline(
            local_add,
            grid=(n_iters,),
            in_specs=[inner_spec],
            out_specs=inner_spec,
            should_accumulate_out=True,
        )
        accum_pipeline(x_hbm_slc, partial_x2.at[working_slot])

        pltpu.semaphore_wait(capacity_sem, value=1)
        pltpu.async_remote_copy(
            partial_x2.at[working_slot],
            partial_x2.at[receiving_slot],
            send_sem,
            recv_sem,
            device_id=neighbor_id,
        ).wait()

    # @pl.when(jax.lax.axis_index("core") == 0)
    @pl.when(True)
    def main():
        prologue()
        pltpu.emit_pipeline(
            pipeline,
            grid=(num_devices, 2),
        )

    final_working_slot = jax.lax.rem(num_devices, 2)
    return partial_x2_lr.at[final_working_slot].reshape(origin_shape)[...]


out = reduce_scatter(inp)
# %%
