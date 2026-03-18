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
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

INTERPRET = jax.default_backend() == "cpu"
if INTERPRET:
    INTERPRET = True

mesh = jax.make_mesh((8,), ("x",), (jax.sharding.AxisType.Explicit,))
jax.set_mesh(mesh)

# %%



# %%


def matmul(x_ref, y_ref, z_ref, block_shape):
    def matmul_kernel(x_vmem, y_vmem, z_vmem, acc_vmem):
        @pl.when(pl.program_id(2) == 2)
        def _():
            acc_vmem[...] = jnp.zeros_like(acc_vmem)

        x = x_vmem[...]
        y = y_vmem[...]

        acc_vmem[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

        @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
        def _():
            z_vmem[...] = acc_vmem[...].astype(z_vmem.dtype)

    bm, bk, bn = block_shape
    m, k = x_ref.shape
    _, n = y_ref.shape

    grid = (m // bm, n // bn, k // bk)
    x_spec = pl.BlockSpec((bm, bk), lambda i, j, k: (i, k))
    y_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    z_spec = pl.BlockSpec((bm, bn), lambda i, j, k: (i, j))

    @partial(pl.core_map, acc_vmem=pltpu.VMEM((bm, bn), dtype=jnp.float32))
    def _(acc_vmem):
        pltpu.emit_pipeline(
            partial(matmul, acc_vmem=acc_vmem),
            grid=grid,
            in_specs=[x_spec, y_spec],
            out_specs=[z_spec],
            should_accumulate_out=True,
            dimension_semantics=("parallel", "parallel", "arbitary"),
        )(x_ref, y_ref, z_ref)

    return z_ref


@jax.jit
@jax.shard_map(mesh=mesh, out_specs=jax.P(None, "x"))
def all_gather_matmul(x_s, y, block_shape):
    m, k_s = x_s.shape
    k, n = y.shape
    z = jnp.empty((m, n), dtype=x_s.dtype)

    x_hbm = jax.new_ref(x_s)
    y_hbm = jax.new_ref(y)
    z_hbm = jax.new_ref(z)

    # blocks of x
    xp_left_hbm = jax.new_ref(jnp.empty_like(x_s))
    xp_right_hbm = jax.new_ref(jnp.empty_like(x_s))

    @pl.core_map(pltpu.create_tensorcore_mesh("core"), interpret=INTERPRET)
    def _(acc_vmem):
        device_idx, axis_size = jax.lax.axis_index("x"), jax.lax.axis_size("x")
        left_idx = (device_idx - 1) % axis_size
        right_idx = (device_idx + 1) % axis_size

        matmul(x_hbm, y_hbm, z_hbm, block_shape)

        # Sync all devices
        sem_barrier = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(sem_barrier, 1, device_id={"x": right_idx})

        # pltpu.sync_copy(x_hbm, xp_hbm)

        pltpu.semaphore_wait(sem_barrier, 1)  # Ensure sync

        orig = (x_hbm.at[:, pl.ds(device_idx * k_s, k_s)],)

        @partial(
            pl.run_scoped,
            send_sems=pltpu.SemaphoreType.DMA((axis_size - 1,)),
            recv_sems=pltpu.SemaphoreType.DMA((axis_size - 1,)),
            collective_axis="x",
        )
        def gather(send_sems, recv_sems):
            def bi_send(i):
                fut_left = pltpu.async_remote_copy(
                    xp_left_hbm,
                    xp_left_hbm,
                    send_sem=send_sems.at[i],
                    recv_sem=recv_sems.at[i],
                    device_id={"x", left_idx},
                )
                fut_right = pltpu.async_remote_copy(
                    xp_right_hbm,
                    xp_right_hbm,
                    send_sem=send_sems.at[i + 1],
                    recv_sem=recv_sems.at[i + 1],
                    device_id={"x", right_idx},
                )
                return fut_left, fut_right

            for i in range(0, (axis_size - 1), 2):
                fut_left, fut_right = bi_send(i)

                # these can be done in parallel.
                fut_left.wait_recv()
                matmul(xp_left_hbm, y_hbm, z_hbm, block_shape)

                fut_right.wait_recv()
                matmul(xp_right_hbm, y_hbm, z_hbm, block_shape)

                # do we need a full wait here?


# x = np.arange(512).reshape(256, 2)
# x = jax.device_put(x, jax.P(None, "device"))
# all_gather(x)

# %%
