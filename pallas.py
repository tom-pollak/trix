# %%
import timeit
import itertools
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

INTERPRET = jax.default_backend() == "cpu"

# %%
# Simple add


@jax.jit
def add_vectors(x, y):
    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = x + y

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=INTERPRET,
    )(x, y)


add_vectors(jnp.arange(8), jnp.arange(8))
# %%


def itoa_kernel(o_ref):
    pid = pl.program_id(0)
    o_ref[pid] = pid


def itoa(n: int):
    return pl.pallas_call(
        itoa_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.int32),
        grid=(n,),
        interpret=True,
    )()


itoa(10)
# %%


def itoa(n: int):
    return pl.pallas_call(
        itoa_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        grid=(n,),
        interpret=True,
    )()


itoa(10)
# %%


def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, transpose_rhs):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    if transpose_rhs:
        dims = ((1,), (1,)), ((), ())  # lhs, rhs
    else:
        dims = ((1,), (0,)), ((), ())

    acc_ref[...] += jax.lax.dot(
        x_ref[...],
        y_ref[...],
        dimension_numbers=dims,
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@jax.jit(static_argnames=["bm", "bn", "bk", "transpose_rhs"])
def matmul(x, y, bm=128, bn=128, bk=128, transpose_rhs=False) -> jax.Array:
    m, k = x.shape
    k_, n = y.shape
    assert k == k_, f"contracting dims must match: {x.shape=}, {y.shape=}"

    grid_m, grid_n, grid_k = pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk)

    x_block_spec = pl.BlockSpec((bm, bk), lambda i, j, k: (i, k))
    if transpose_rhs:
        y = y.swapaxes(0, 1)
        y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
    else:
        y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))

    return pl.pallas_call(
        partial(matmul_kernel, transpose_rhs=transpose_rhs),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(grid_m, grid_n, grid_k),
        in_specs=[x_block_spec, y_block_spec],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=2**25,
        ),
        interpret=INTERPRET,
    )(x, y)


dtype = jnp.bfloat16
m, n, k = 1024, 1024, 1024
bm, bn, bk = 512, 512, 512

k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (m, k), dtype=dtype)
y = jax.random.normal(k2, (k, n), dtype=dtype)
z = matmul(x, y, bm=bm, bn=bn, bk=bk)
np.testing.assert_allclose(z, x @ y, rtol=1e-2, atol=1e-4)

k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (m, k), dtype=dtype)
y = jax.random.normal(k2, (n, k), dtype=dtype)


@jax.jit(static_argnames=["bm", "bn", "bk"])
def transpose_matmul(x, y, bm, bn, bk):
    y = y.swapaxes(0, 1)  # tranpose must be inside jit for it to be a logical xfm
    return matmul(x, y, bm, bn, bk, transpose_rhs=True)


z = transpose_matmul(x, y, bm=bm, bn=bn, bk=bk)
np.testing.assert_allclose(z, x @ y.T, rtol=1e-2, atol=1e-4)

# %%


def matmul_flops(m, n, k, bm, bn, bk, dtype: jnp.dtype):
    grid_m, grid_n, grid_k = pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk)
    return jnp.dtype(dtype).itemsize * bm * bn * bk * grid_m * grid_n * grid_k


def matmul_membw(m, n, k, bm, bn, bk, dtype: jnp.dtype):
    grid_m, grid_n, grid_k = pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk)
    return (
        jnp.dtype(dtype).itemsize
        # for each output tile (grid_m, grid_n)
        # Read a tile X[bm, bk] Y[bk, bn] for each part along contraction dim grid_k
        # Write output_tile Z[bm, bn]
        * ((bm * bk + bk * bn) * grid_k + bm * bn)
        * grid_m
        * grid_n
    )


def matmul_arithmetic_intensity(m, n, k, bm, bn, bk, dtype: jnp.dtype):
    return matmul_flops(m, n, k, bm, bn, bk, dtype) / matmul_membw(
        m, n, k, bm, bn, bk, dtype
    )


v5e_flops = 197e12
v5e_membw = 819e9
v5e_intensity = v5e_flops / v5e_membw  # ~240.5

matmul_intensity = matmul_arithmetic_intensity(m, n, k, bm, bn, bk, dtype)
print(f"{matmul_intensity=:.2f} {v5e_intensity=:.2f}")
# %%


@dataclass
class TuningResult:
    bm: int
    bn: int
    bk: int
    time: float
    flops: float
    utilization: float


def bench(f, ntrials=100):
    def run(*args, **kwargs):
        jax.block_until_ready(f(*args, **kwargs))
        t = timeit.timeit(
            lambda: jax.block_until_ready(f(*args, **kwargs)), number=ntrials
        )
        return t / ntrials

    return run


def autotune_matmul(
    m,
    n,
    k,
    block_sizes=(128, 256, 512, 1024),
    dtype=jnp.bfloat16,
    transpose_rhs=False,
    ntrials=50,
):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (m, k), dtype=dtype)
    if transpose_rhs:
        y = jax.random.normal(k2, (n, k), dtype=dtype)
    else:
        y = jax.random.normal(k2, (k, n), dtype=dtype)

    def wrap(f, transpose_rhs):
        @jax.jit(static_argnames=["bm", "bn", "bk", "transpose_rhs"])
        def inner(x, y, bm=None, bn=None, bk=None, transpose_rhs=None):
            if transpose_rhs:
                y = y.swapaxes(0, 1)
            if bm is None and bn is None and bk is None:
                return f(x, y)
            else:
                return f(x, y, bm, bn, bk, transpose_rhs)

        return inner

    xla_time = bench(wrap(jnp.matmul, transpose_rhs), ntrials=ntrials)(
        x, y, transpose_rhs=transpose_rhs
    )
    xla_flops = matmul_flops(m, n, k, m, n, k, dtype) / xla_time
    xla_util = xla_flops / v5e_flops

    candidates = []
    for bm, bn, bk in itertools.product(block_sizes, repeat=3):
        time = bench(wrap(matmul, transpose_rhs), ntrials=ntrials)(
            x, y, bm, bn, bk, transpose_rhs
        )
        flops = matmul_flops(m, n, k, bm, bn, bk, dtype) / time
        util = flops / v5e_flops
        candidates.append(TuningResult(bm, bn, bk, time, flops, util))

    candidates.sort(key=lambda r: r.time)
    return candidates, xla_time, xla_util


shapes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1, 8192, 8192),
    (128, 8192, 8192),
    (4096, 8192, 1024),
]

for m, n, k in shapes:
    candidates, xla_time, xla_util = autotune_matmul(m, n, k)
    print(f"\n{'=' * 65}")
    print(f"({m}, {n}, {k}) — {len(candidates)} configs tested")
    print(f"  XLA baseline: {xla_time:.5f}s  util={xla_util * 100:.2f}%")
    print(f"{'-' * 65}")
    for i, r in enumerate(candidates[:5]):
        speedup = xla_time / r.time
        print(
            f"  #{i + 1}  bm={r.bm:<5} bn={r.bn:<5} bk={r.bk:<5} "
            f"{r.time:.5f}s  util={r.utilization * 100:.2f}%  "
            f"vs XLA: {speedup:.2f}x"
        )
