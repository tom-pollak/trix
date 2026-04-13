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

inp = jax.random.uniform(jax.random.key(0), (2048, 512))
inp = jax.device_put(inp, jax.P(None, None))


# %%
@jax.jit
@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=jax.P(None, None),
    out_specs=jax.P(None, None),
    check_vma=False,
)
def ref_reduce_scatter(x):
    x = jax.lax.psum_scatter(x, "x", scatter_dimension=0, tiled=True)
    return x


out_ref = ref_reduce_scatter(inp)

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


@jax.jit(static_argnums=1)
@partial(
    jax.shard_map,
    in_specs=jax.P(None, None),
    out_specs=jax.P(None, None),
    check_vma=False,
)
def reduce_scatter(x_s, reduce_fn=None):
    device_idx = jax.lax.axis_index("x")
    num_devices = jax.lax.axis_size("x")
    M, D = x_s.shape

    inner_block = (8, 128)
    assert np.prod(x_s.shape) % (2 * np.prod(inner_block)) == 0, (
        "x must be shape contrained"
    )

    x_s_lr = x_s.reshape(2, -1, *inner_block)
    x_s_lr = jax.new_ref(x_s_lr)
    partial_lr_x2 = jax.new_ref(jnp.empty((2, *x_s_lr.shape), x_s_lr.dtype))

    def prologue():
        left = jax.lax.rem(device_idx - 1, num_devices)
        right = jax.lax.rem(device_idx + 1, num_devices)
        local_barrier(left, right)
        pltpu.sync_copy(x_s_lr, partial_lr_x2.at[0])

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
        partial_x2 = partial_lr_x2.at[left_or_right]

        # trigger copy into receiving_slot
        pltpu.semaphore_signal(capacity_sem, inc=1, device_id=neighbor_id)

        def default_add(x_vmem, partial_vmem):
            partial_vmem[...] = x_vmem[...]  # should_accumulate_out

        inner_spec = pl.BlockSpec(inner_block, lambda i: (i, 0, 0))
        accum_pipeline = pltpu.emit_pipeline(
            reduce_fn or default_add,
            grid=(num_devices,),
            in_specs=[inner_spec],
            out_specs=inner_spec,
            should_accumulate_out=not reduce_fn,
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

    @pl.core_map(
        mesh=pltpu.create_tensorcore_mesh("core", num_cores=2),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )
    def main():
        @pl.when(jax.lax.axis_index("core") == 0)
        def _():
            prologue()
            pltpu.emit_pipeline(
                pipeline,
                grid=(num_devices, 2),
            )

    final_working_slot = jax.lax.rem(num_devices, 2)
    return (
        partial_lr_x2.at[:, final_working_slot]
        .reshape(num_devices, M // num_devices, D)
        .at[device_idx][...]
    )


print(inp)
out = reduce_scatter(inp).block_until_ready()
print(out)
assert jnp.allclose(out, out_ref)

# %%
