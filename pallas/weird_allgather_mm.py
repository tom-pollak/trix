# %%
"""
The sharding strategy doesn't really make an sense, but this is more for learning.

X["x", None] @ Y[None, None] -> Z[None, None]

1. X["x", None] @ Y[None, None] -> Z["x", None] -> Z[None, None] (all-reduce, recommended)
2. all-gather(X["x", None] -> X[None, None]) @ Y[None, None] -> Z[None, None]
    - Where the ring all-gather is overlapped with matmul
    - Wasted FLOPs, since all devices doing the full matmul, so this is not very good. But what the kernel
      is doing

I'm also intentionally not using pl.pallas_call / pl.kernel
"""

# %%
# import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import lovely_jax as lt

lt.monkey_patch()

from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

if jax.default_backend() == "cpu":
    pltpu.set_tpu_interpret_mode(pltpu.InterpretParams(num_cores_or_threads=2))

mesh = jax.make_mesh((jax.device_count(),), ("x",), (jax.sharding.AxisType.Explicit,))
jax.set_mesh(mesh)


# %%


def matmul(x_ref, y_ref, z_ref, acc_vmem, block_shape, write_z: bool, zero_acc: bool):
    def matmul_kernel(x_vmem, y_vmem, z_vmem):
        @pl.when(zero_acc & (pl.program_id(2) == 0))
        def _():
            acc_vmem[...] = jnp.zeros_like(acc_vmem)

        x = x_vmem[...]
        y = y_vmem[...]

        acc_vmem[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

        @pl.when(write_z & (pl.program_id(2) == pl.num_programs(2) - 1))
        def _():
            z_vmem[...] = acc_vmem[...].astype(z_vmem.dtype)

    bm, bk, bn = block_shape
    m, k = x_ref.shape
    _, n = y_ref.shape

    grid = (m // bm, n // bn, k // bk)
    x_spec = pl.BlockSpec((bm, bk), lambda i, j, k: (i, k))
    y_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    z_spec = pl.BlockSpec((bm, bn), lambda i, j, k: (i, j))

    # pltpu.VMEM((bm, bn), dtype=jnp.float32)
    pltpu.emit_pipeline(
        matmul_kernel,
        grid=grid,
        in_specs=[x_spec, y_spec],
        out_specs=[z_spec],
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
    )(x_ref, y_ref, z_ref)


BLOCK_SHAPE = (256, 256, 256)


@jax.jit
@jax.shard_map(mesh=mesh, out_specs=jax.P(None, None), check_vma=False)
def all_gather_matmul(x_s, y):
    bm, bk, bn = BLOCK_SHAPE
    m, k_s = x_s.shape
    k, n = y.shape

    x_hbm = jax.new_ref(x_s)
    y_hbm = jax.new_ref(y)

    # blocks of x
    xp_left_hbm = jax.new_ref(jnp.empty(x_s.shape, dtype=x_s.dtype))
    xp_right_hbm = jax.new_ref(jnp.empty_like(xp_left_hbm))

    device_idx, axis_size = jax.lax.axis_index("x"), jax.lax.axis_size("x")

    @pl.kernel(
        mesh=pltpu.create_tensorcore_mesh("core", num_cores=2),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=dict(
            acc_vmem=pltpu.VMEM((bm, bn), dtype=jnp.float32),
            local_sems=pltpu.SemaphoreType.DMA((2,)),
            send_sems=pltpu.SemaphoreType.DMA((axis_size - 1,)),
            recv_sems=pltpu.SemaphoreType.DMA((axis_size - 1,)),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )
    def kernel(z_hbm, acc_vmem, local_sems, send_sems, recv_sems):
        left_idx = (device_idx - 1) % axis_size
        right_idx = (device_idx + 1) % axis_size

        fut_left = pltpu.async_copy(x_hbm, xp_left_hbm, local_sems.at[0])
        fut_right = pltpu.async_copy(x_hbm, xp_right_hbm, local_sems.at[1])

        write_z = axis_size == 1
        matmul(
            x_hbm, y_hbm, z_hbm, acc_vmem, BLOCK_SHAPE, write_z=write_z, zero_acc=True
        )

        # Sync all devices
        sem_barrier = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(sem_barrier, 1, device_id={"x": left_idx})
        pltpu.semaphore_signal(sem_barrier, 1, device_id={"x": right_idx})
        pltpu.semaphore_wait(sem_barrier, 2)  # Ensure sync

        fut_left.wait()
        fut_right.wait()

        def bi_send(i):
            fut_left = pltpu.async_remote_copy(
                xp_left_hbm,
                xp_left_hbm,
                send_sem=send_sems.at[i],
                recv_sem=recv_sems.at[i],
                device_id={"x": left_idx},
            )
            fut_right = pltpu.async_remote_copy(
                xp_right_hbm,
                xp_right_hbm,
                send_sem=send_sems.at[i + 1],
                recv_sem=recv_sems.at[i + 1],
                device_id={"x": right_idx},
            )
            return fut_left, fut_right

        for i in range(0, (axis_size - 1), 2):
            write_z = i == axis_size - 2
            fut_left, fut_right = bi_send(i)

            fut_left.wait_recv()
            matmul(
                xp_left_hbm,
                y_hbm,
                z_hbm,
                acc_vmem,
                BLOCK_SHAPE,
                write_z=write_z,
                zero_acc=False,
            )

            fut_right.wait_recv()
            matmul(
                xp_right_hbm,
                y_hbm,
                z_hbm,
                acc_vmem,
                BLOCK_SHAPE,
                write_z=write_z,
                zero_acc=False,
            )

            fut_left.wait_send()
            fut_right.wait_send()

    return kernel()


x = jax.random.uniform(jax.random.key(0), (2048, 2048), dtype=jnp.bfloat16)
y = jax.random.uniform(jax.random.key(1), (2048, 2048), dtype=jnp.bfloat16)

x = jax.device_put(x, jax.P(None, "x"))
y = jax.device_put(y, jax.P(None, None))

z = all_gather_matmul(x, y)
z_ref = x @ y
print(z.v)
print(z_ref.v)
assert jnp.allclose(z, z_ref)

# %%
