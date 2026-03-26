# %%
"""
Each shard has its own seperate roll parameter, and performs a roll on its own local
slice. No communication required between the devices.

This is a rather arbitary problem obviously, doing it in a few different ways:

1. Simple jax.jit
2. Pallas loading from HBM
3. Handrolled pipelining pallas kernel.
4. SparseCore gather/scatter

1,2,3 all match perf, 4 is noticeably slower (since it is a contiguous DMA block, not
suited to SC).
"""

from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.experimental.pallas.tpu_sc as plsc
import chex

# %%

mesh = jax.make_mesh(
    (2, 2),  # MOCK
    ("X", "Y"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)
jax.set_mesh(mesh)

# %%

# %%

S, D = 16384, 512

A = jnp.arange(S * D, dtype=jnp.bfloat16).reshape(S, D)
A = jax.device_put(A, jax.P("X", "Y"))
shifts = jax.random.randint(
    jax.random.key(0), (mesh.axis_sizes[0],), 0, S, dtype=jnp.int32
)
shifts = jax.device_put(shifts, jax.P("X"))

# %%


@jax.jit
@partial(
    jax.shard_map,
    in_specs=(jax.P("X", "Y"), jax.P("X")),
    out_specs=jax.P("X", "Y"),
)
@chex.assert_max_traces(n=1)
def roll_shard_ref(A_s, shift):
    return jnp.roll(A_s, shift, axis=0)


ref = roll_shard_ref(A, shifts)
ref

# %%


@jax.jit
@partial(
    jax.shard_map,
    in_specs=(jax.P("X", "Y"), jax.P("X")),
    out_specs=jax.P("X", "Y"),
    check_vma=False,
)
def roll_shard_simple(A_s, shift):  # ...ish
    M, N = A_s.shape
    BM = 2048
    num_blocks = M // BM

    A_tiled = jnp.concatenate([A_s, A_s], axis=0)
    shift = (-shift) % M

    @pl.kernel(
        out_shape=A_s,
        scratch_shapes=[
            pltpu.VMEM((BM + 8, N), A_s.dtype),  # big_buf: load BM+8 rows
            pltpu.VMEM((BM, N), A_s.dtype),  # out_buf
            pltpu.SMEM((1,), jnp.int32),
            pltpu.SemaphoreType.DMA,  # ssem
            pltpu.SemaphoreType.DMA,  # rsem
            pltpu.SemaphoreType.DMA,  # wsem
        ],
        mesh=pltpu.create_tensorcore_mesh("core"),
    )
    def kernel(a_hbm, s_hbm, o_hbm, big_buf, out_buf, s_smem, ssem, rsem, wsem):
        pltpu.async_copy(s_hbm, s_smem, ssem).wait()
        s = s_smem[0]
        s_8 = (s // 8) * 8  # 8-aligned offset for DMA
        s_rem = s % 8  # 0-7 remainder for VMEM

        @pl.loop(0, num_blocks)
        def _(block):
            # DMA: 8-aligned start, static size BM+8
            pltpu.async_copy(
                a_hbm.at[pl.ds(block * BM + s_8, BM + 8), :],
                big_buf,
                rsem,
            ).wait()

            for r in range(8):

                @pl.when(s_rem == r)
                def _():
                    out_buf[...] = big_buf[pl.ds(r, BM), :]

            pltpu.async_copy(
                out_buf,
                o_hbm.at[pl.ds(block * BM, BM), :],
                wsem,
            ).wait()

    return kernel(A_tiled, shift)


roll_shard_simple(A, shifts)

# %%


@jax.jit
@partial(
    jax.shard_map,
    in_specs=(jax.P("X", "Y"), jax.P("X")),
    out_specs=jax.P("X", "Y"),
    check_vma=False,
)
@chex.assert_max_traces(n=1)
def roll_shard_pipelined(A_s, shift, BM=2048, pipeline_len=2):  # also multi-core
    M, N = A_s.shape
    num_blocks = pl.cdiv(M, BM)
    A_tiled = jnp.concatenate([A_s, A_s], axis=0)
    shift = (-shift) % M

    @pl.kernel(
        out_shape=A_s,
        scratch_shapes=[
            pltpu.VMEM((pipeline_len, BM + 8, N), A_s.dtype),  # big_buf
            pltpu.VMEM((pipeline_len, BM, N), A_s.dtype),  # out_buf
            pltpu.SMEM((1,), jnp.int32),
            pltpu.SemaphoreType.DMA,  # ssem
            pltpu.SemaphoreType.DMA((pipeline_len,)),
            pltpu.SemaphoreType.DMA((pipeline_len,)),
        ],
        mesh=pltpu.create_tensorcore_mesh("core"),
    )
    def kernel(a_hbm, s_hbm, o_hbm, big_buf_x2, out_buf_x2, s_smem, ssem, rsems, wsems):
        pltpu.async_copy(s_hbm, s_smem, ssem).wait()
        s = s_smem[0]
        s_8 = (s // 8) * 8
        s_rem = s % 8
        core_idx = jax.lax.axis_index("core")
        num_cores = jax.lax.axis_size("core")
        num_blocks_per_core = pl.cdiv(num_blocks, num_cores)

        def get_slot(o, i):
            slot = jax.lax.rem(i, pipeline_len)
            return o.at[slot]

        def shift_copy(i):
            src, dst = get_slot(big_buf_x2, i), get_slot(out_buf_x2, i)
            for r in range(8):

                @pl.when(s_rem == r)
                def _():
                    dst[...] = src[r : r + BM, :]

        def get_read_desc(i):
            buf, sem = get_slot(big_buf_x2, i), get_slot(rsems, i)
            return pltpu.make_async_copy(
                a_hbm.at[pl.ds(i * BM + s_8, BM + 8), :], buf, sem
            )

        def get_write_desc(i):
            buf, sem = get_slot(out_buf_x2, i), get_slot(wsems, i)
            return pltpu.make_async_copy(buf, o_hbm.at[pl.ds(i * BM, BM), :], sem)

        core_start = num_blocks_per_core * core_idx

        # prologue
        for i in range(pipeline_len):
            get_read_desc(core_start + i).start()

        @pl.loop(0, num_blocks_per_core)
        def _(i):
            i = i + core_start
            get_read_desc(i).wait()

            @pl.when(i - pipeline_len >= core_start)  # wait previous copy
            def _():
                get_write_desc(i - pipeline_len).wait()

            shift_copy(i)
            get_write_desc(i).start()

            @pl.when(i + pipeline_len < core_start + num_blocks_per_core)
            def _():
                get_read_desc(i + pipeline_len).start()

        # epilogue
        for i in range(pipeline_len):
            get_write_desc(core_start + num_blocks_per_core - pipeline_len + i).wait()

    return kernel(A_tiled, shift)


chex.clear_trace_counter()
chex.assert_trees_all_close(roll_shard_pipelined(A, shifts), ref)
roll_shard_pipelined(A, shifts)

# %%


@jax.jit
@partial(
    jax.shard_map,
    in_specs=(jax.P("X", "Y"), jax.P("X")),
    out_specs=jax.P("X", "Y"),
    check_vma=False,
)
@chex.assert_max_traces(n=1)
def roll_shard_sparsecore(A_s, shift):  # slower since roll is a nice big DMA load/store
    M, N = A_s.shape
    gather_window = 256
    assert pltpu.get_tpu_info().sparse_core is not None, "tpu does not have sparse cores"

    indices = (jnp.arange(M, dtype=jnp.int32)[None, :] - shift) % M

    @pl.kernel(
        out_shape=A_s,
        mesh=plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore"),
    )
    def do_roll(A_hbm, i_hbm, o_hbm):
        def body(i_vmem, o_vmem):
            gather_a = A_hbm.at[i_vmem.at[0], :]
            pltpu.sync_copy(gather_a, o_vmem)

        pltpu.emit_pipeline(
            body,
            grid=(M // gather_window,),
            in_specs=[
                pl.BlockSpec(
                    block_shape=(1, gather_window),
                    index_map=lambda i: (0, i),
                )
            ],
            out_specs=[
                pl.BlockSpec(block_shape=(gather_window, N), index_map=lambda i: (i, 0))
            ],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(i_hbm, o_hbm)

    return do_roll(A_s, indices)


chex.clear_trace_counter()
chex.assert_trees_all_close(roll_shard_sparsecore(A, shifts), ref)
roll_shard_sparsecore(A, shifts)

# %%
