# %%
# fmt: off
import functools as ft
import timeit
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

pltpu.CompilerParams()
INTERPRET = jax.default_backend() == "cpu"

# %%

def dynamic_slice_kernel(indices, x_ref, o_ref):
    del indices
    o_ref[...] = x_ref[...]

@checkify.checkify
@ft.partial(jax.jit, static_argnames=["sizes"])
def block_dynamic_slice(x, starts, sizes):
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(1, 1),
        in_specs=[pl.BlockSpec(
            sizes,
            lambda i, j, block_idx: (block_idx[0], block_idx[1]))],
        out_specs=pl.BlockSpec(sizes, lambda *_: (0, 0)),
    )
    kernel = pl.pallas_call(
        dynamic_slice_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
        interpret=INTERPRET,
    )

    checkify.check(starts[0] % sizes[0] == 0, "starts must be divisible by size")
    checkify.check(starts[1] % sizes[1] == 0, "starts must be divisible by size")
    block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]])
    return kernel(block_idx, x)

shape = (16, 16)
starts, sizes = (4, 4), (2, 2)
x = jnp.reshape(jnp.arange(np.prod(shape), dtype=jnp.int32), shape)
err, result = block_dynamic_slice(x, starts, sizes)
err.throw()
ref = jax.lax.dynamic_slice(x, start_indices=starts, slice_sizes=sizes)
ref2 = x[starts[0] : starts[0] + sizes[0], starts[1] : starts[1] + sizes[1]]
np.testing.assert_allclose(result, ref)
np.testing.assert_allclose(result, ref2)

# %%

def generate_block_sparse_mat(key, m, n, bm, bn, p=0.2, dtype=jnp.float32):
    """Returns a sampled matrix and its block-sparse representation.

    Args:
        key: RNG Key.
        m: Major array dimension.
        n: Minor array dimension.
        bm: Block size along M dimension.
        bn: Block size along N dimension.
        p: Probability that a block will be non-zero.
        dtype: dtype of the sampled matrix.

    Returns:
        dense_mat: A (M, N) dense sampled array.
        block_data: A (num_blocks, blk_M, blk_N) array of data blocks representing
          the non-zero blocks of the matrix.
        indices_i: A (num_blocks,) array of block indices for the first axis.
        indices_j: A (num_blocks,) array of block indices for the second axis.
    """
    mask_key, blocks_key = jax.random.split(key)
    num_blocks = (m // bm, n // bn)
    # We first sample a block mask, denoting which blocks are nonzero.
    block_mask = jax.random.bernoulli(mask_key, p=p, shape=num_blocks)
    num_blocks = jnp.sum(block_mask)
    indices = jnp.where(block_mask)
    # For each non-zero block, we sample a block of random values.
    block_data = jax.random.uniform(blocks_key,
                                    shape=(num_blocks, bm, bn),
                                    dtype=dtype)
    # For checking purposes, create the dense version of the sparse matrix.
    dense_mat = jnp.zeros((m, n), dtype=dtype)
    for blk in range(num_blocks):
        idx_i = indices[0][blk]
        idx_j = indices[1][blk]
        slice_i = slice(idx_i * bm, (idx_i + 1) * bm)
        slice_j = slice(idx_j * bn, (idx_j + 1) * bn)
        dense_mat = dense_mat.at[slice_i, slice_j].set(block_data[blk])
    return dense_mat, block_data, indices[0], indices[1]

def dsd_kernel(idxs_i_ref, idxs_k_ref, # scale prefetch inputs
               x_ref, y_ref, _, o_ref,
               accum_scratch,
               num_blocks,
              ):
    """DSD (Dense = Sparse @ Dense) matmul kernel"""
    del idxs_k_ref
    blk_idx = pl.program_id(1)

    is_start = blk_idx == 0
    changed_blocks = idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx-1, 0)]
    @pl.when(is_start | changed_blocks)
    def _():
        accum_scratch[...] = jnp.zeros_like(accum_scratch)
    accum_scratch[...] += jnp.dot(
        x_ref[0, :, :], y_ref[...], preferred_element_type=jnp.float32)

    next_block_change = idxs_i_ref[blk_idx] != idxs_i_ref[jnp.minimum(blk_idx+1, num_blocks)]
    is_end = blk_idx == num_blocks - 1
    @pl.when(is_end | next_block_change)
    def _():
        o_ref[...] = accum_scratch[...].astype(o_ref.dtype)

def dsd_matmul(x_blocks, y, indices_i, indices_k):
    num_blocks = x_blocks.shape[0]
    x_map = lambda j, blk_idx, blk_idxs_i, blk_idxs_k: (blk_idx, 0, 0)  # noqa: E731
    y_map = lambda j, blk_idx, blk_idxs_i, blk_idxs_k: (blk_idxs_k[blk_idx], j)  # noqa: E731
    o_map = lambda j, blk_idx, blk_idxs_i, blk_idxs_k: (blk_idxs_i[blk_idx], j)  # noqa: E731
    out_shape = jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(n // bn, num_blocks),
        in_specs=[
            pl.BlockSpec((1, bm, bk), x_map),
            pl.BlockSpec((bk, bn), y_map),
            pl.BlockSpec((bm, bn), o_map), # placeholder for zero array
        ],
        out_specs=pl.BlockSpec((bm, bn), o_map),
        scratch_shapes=[pltpu.VMEM((bm, bn), dtype=jnp.float32)],
    )
    zeros = jnp.zeros((m, n), dtype=jnp.bfloat16)
    kernel = pl.pallas_call(
        ft.partial(dsd_kernel, num_blocks=num_blocks),
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases={4: 0}, # map zeros to o_ref -- we don't visit all blocks
        interpret=INTERPRET,
    )
    return kernel(indices_i, indices_k, x_blocks, y, zeros)

m = n = k = 1024
bm = bn = bk = 128
X_dense, X_blocks, indices_i, indices_k = generate_block_sparse_mat(
    jax.random.key(0), m, k, bm, bk, p=0.1, dtype=jnp.bfloat16)
Y = jax.random.uniform(jax.random.key(1), shape=(k, n), dtype=jnp.bfloat16)

result = dsd_matmul(X_blocks, Y, indices_i, indices_k)
ref = X_dense @ Y
np.testing.assert_allclose(result, ref)

# %%
