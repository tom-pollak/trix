# %%
from jax.extend.mlir.dialects.func import return_
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


def fit_in_vmem(bm, bn, bk, dtype, vmem_limit=2**25, return_usage=False):
    elem_size = jnp.dtype(dtype).itemsize
    acc_elem_size = 4  # float32
    vmem_usage = (
        2 * (bm * bk + bn * bk) * elem_size  # double-buffered x,y buffers
        + bm * bn * acc_elem_size  # scratch accumulator (f32)
        + bm * bn * elem_size  # single output z buffer
    )
    if return_usage:
        return vmem_usage < vmem_limit, vmem_usage
    return vmem_usage < vmem_limit


def autotune_matmul(
    m,
    n,
    k,
    block_sizes=(768, 1024, 1280, 1536, 1792, 2048),
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
        partial(jax.jit, static_argnames=["bm", "bn", "bk", "transpose_rhs"])

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
        should_fit, vmem_bytes = fit_in_vmem(bm, bn, bk, dtype, return_usage=True)
        vmem_mib = vmem_bytes / 2**20
        # if not fit_in_vmem(bm, bn, bk, dtype):
        #     continue

        try:
            time = bench(wrap(matmul, transpose_rhs), ntrials=ntrials)(
                x, y, bm, bn, bk, transpose_rhs
            )
        except jax.errors.JaxRuntimeError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(
                    f"OOM on ({m} {n} {k}) ; ({bm} {bn} {bk}) -- {should_fit=} {vmem_mib=}"
                )
                continue

        if not should_fit:
            print(
                f"UNEXPECTED -- ({m} {n} {k}) ; ({bm} {bn} {bk} should have OOM! ({vmem_mib=})"
            )

        flops = matmul_flops(m, n, k, bm, bn, bk, dtype) / time
        util = flops / v5e_flops
        candidates.append(TuningResult(bm, bn, bk, time, flops, util))

    candidates.sort(key=lambda r: r.time)
    return candidates, xla_time, xla_util


shapes = [
    # (1024, 1024, 1024),
    # (2048, 2048, 2048),
    # (4096, 4096, 4096),
    (8192, 8192, 8192),
    # (1, 8192, 8192),
    # (128, 8192, 8192),
    # (4096, 8192, 1024),
]

for m, n, k in shapes:
    for transpose_rhs in False, True:
        candidates, xla_time, xla_util = autotune_matmul(
            m, n, k, transpose_rhs=transpose_rhs
        )
        print(f"\n{'=' * 65}")
        print(f"({m}, {n}, {k}) - {transpose_rhs=} - {len(candidates)} configs tested")
        print(f"  XLA baseline: {xla_time:.5f}s  util={xla_util * 100:.2f}%")
        print(f"{'-' * 65}")
        for i, r in enumerate(candidates[:5]):
            speedup = xla_time / r.time
            print(
                f"  #{i + 1}  bm={r.bm:<5} bn={r.bn:<5} bk={r.bk:<5} "
                f"{r.time:.5f}s  util={r.utilization * 100:.2f}%"
                f"vs XLA: {speedup:.2f}x"
            )


"""
=================================================================
(1024, 1024, 1024) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00019s  util=5.61%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=256   0.00015s  util=7.22%  vs XLA: 1.29x
  #2  bm=1024  bn=512   bk=256   0.00015s  util=7.19%  vs XLA: 1.28x
  #3  bm=1024  bn=1024  bk=128   0.00015s  util=7.16%  vs XLA: 1.28x
  #4  bm=1024  bn=256   bk=1024  0.00015s  util=7.16%  vs XLA: 1.28x
  #5  bm=1024  bn=1024  bk=1024  0.00015s  util=7.09%  vs XLA: 1.26x

=================================================================
(1024, 1024, 1024) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00022s  util=4.93%
-----------------------------------------------------------------
  #1  bm=256   bn=1024  bk=1024  0.00021s  util=5.30%  vs XLA: 1.08x
  #2  bm=512   bn=512   bk=1024  0.00021s  util=5.30%  vs XLA: 1.07x
  #3  bm=512   bn=256   bk=1024  0.00021s  util=5.26%  vs XLA: 1.07x
  #4  bm=128   bn=512   bk=1024  0.00021s  util=5.25%  vs XLA: 1.07x
  #5  bm=512   bn=1024  bk=256   0.00021s  util=5.21%  vs XLA: 1.06x

=================================================================
(2048, 2048, 2048) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00028s  util=31.50%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=1024  0.00024s  util=35.86%  vs XLA: 1.14x
  #2  bm=1024  bn=512   bk=1024  0.00025s  util=34.90%  vs XLA: 1.11x
  #3  bm=1024  bn=1024  bk=512   0.00025s  util=34.84%  vs XLA: 1.11x
  #4  bm=1024  bn=1024  bk=256   0.00026s  util=34.11%  vs XLA: 1.08x
  #5  bm=1024  bn=1024  bk=128   0.00026s  util=33.02%  vs XLA: 1.05x

=================================================================
(2048, 2048, 2048) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00030s  util=29.54%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=512   0.00030s  util=28.92%  vs XLA: 0.98x
  #2  bm=1024  bn=1024  bk=1024  0.00030s  util=28.82%  vs XLA: 0.98x
  #3  bm=1024  bn=512   bk=1024  0.00031s  util=28.44%  vs XLA: 0.96x
  #4  bm=512   bn=1024  bk=1024  0.00031s  util=28.32%  vs XLA: 0.96x
  #5  bm=1024  bn=512   bk=512   0.00031s  util=28.28%  vs XLA: 0.96x

=================================================================
(4096, 4096, 4096) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00093s  util=75.30%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=1024  0.00095s  util=73.46%  vs XLA: 0.98x
  #2  bm=1024  bn=512   bk=1024  0.00097s  util=71.69%  vs XLA: 0.95x
  #3  bm=1024  bn=1024  bk=512   0.00097s  util=71.69%  vs XLA: 0.95x
  #4  bm=512   bn=1024  bk=1024  0.00098s  util=71.52%  vs XLA: 0.95x
  #5  bm=1024  bn=1024  bk=256   0.00100s  util=69.76%  vs XLA: 0.93x

=================================================================
(4096, 4096, 4096) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00108s  util=64.31%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=512   0.00119s  util=58.40%  vs XLA: 0.91x
  #2  bm=1024  bn=1024  bk=1024  0.00120s  util=58.26%  vs XLA: 0.91x
  #3  bm=1024  bn=512   bk=1024  0.00120s  util=57.90%  vs XLA: 0.90x
  #4  bm=512   bn=1024  bk=1024  0.00123s  util=56.90%  vs XLA: 0.88x
  #5  bm=1024  bn=1024  bk=256   0.00123s  util=56.87%  vs XLA: 0.88x

=================================================================
(8192, 8192, 8192) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00619s  util=90.12%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=1024  0.00611s  util=91.30%  vs XLA: 1.01x
  #2  bm=1024  bn=1024  bk=512   0.00620s  util=89.95%  vs XLA: 1.00x
  #3  bm=1024  bn=512   bk=1024  0.00622s  util=89.68%  vs XLA: 1.00x
  #4  bm=512   bn=1024  bk=1024  0.00624s  util=89.48%  vs XLA: 0.99x
  #5  bm=1024  bn=1024  bk=256   0.00644s  util=86.66%  vs XLA: 0.96x

=================================================================
(8192, 8192, 8192) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00662s  util=84.27%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=1024  0.00692s  util=80.66%  vs XLA: 0.96x
  #2  bm=512   bn=1024  bk=1024  0.00703s  util=79.45%  vs XLA: 0.94x
  #3  bm=1024  bn=512   bk=1024  0.00704s  util=79.31%  vs XLA: 0.94x
  #4  bm=1024  bn=1024  bk=512   0.00709s  util=78.74%  vs XLA: 0.93x
  #5  bm=1024  bn=1024  bk=256   0.00733s  util=76.17%  vs XLA: 0.90x

=================================================================
(1, 8192, 8192) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00038s  util=0.18%
-----------------------------------------------------------------
  #1  bm=128   bn=1024  bk=1024  0.00038s  util=22.89%  vs XLA: 1.00x
  #2  bm=128   bn=1024  bk=512   0.00039s  util=22.65%  vs XLA: 0.99x
  #3  bm=128   bn=512   bk=1024  0.00040s  util=21.85%  vs XLA: 0.95x
  #4  bm=256   bn=1024  bk=1024  0.00040s  util=43.29%  vs XLA: 0.94x
  #5  bm=128   bn=1024  bk=256   0.00042s  util=20.57%  vs XLA: 0.90x

=================================================================
(1, 8192, 8192) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00079s  util=0.09%
-----------------------------------------------------------------
  #1  bm=128   bn=1024  bk=1024  0.00120s  util=7.27%  vs XLA: 0.66x
  #2  bm=128   bn=512   bk=1024  0.00121s  util=7.22%  vs XLA: 0.66x
  #3  bm=256   bn=1024  bk=1024  0.00122s  util=14.35%  vs XLA: 0.65x
  #4  bm=128   bn=1024  bk=512   0.00123s  util=7.08%  vs XLA: 0.64x
  #5  bm=128   bn=512   bk=512   0.00125s  util=6.95%  vs XLA: 0.63x

=================================================================
(128, 8192, 8192) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00040s  util=21.93%
-----------------------------------------------------------------
  #1  bm=128   bn=1024  bk=512   0.00042s  util=20.80%  vs XLA: 0.95x
  #2  bm=128   bn=1024  bk=1024  0.00042s  util=20.80%  vs XLA: 0.95x
  #3  bm=256   bn=1024  bk=1024  0.00044s  util=39.68%  vs XLA: 0.90x
  #4  bm=128   bn=1024  bk=256   0.00044s  util=19.62%  vs XLA: 0.89x
  #5  bm=128   bn=512   bk=1024  0.00046s  util=19.05%  vs XLA: 0.87x

=================================================================
(128, 8192, 8192) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00080s  util=10.92%
-----------------------------------------------------------------
  #1  bm=128   bn=512   bk=1024  0.00124s  util=7.06%  vs XLA: 0.65x
  #2  bm=128   bn=1024  bk=1024  0.00124s  util=7.02%  vs XLA: 0.64x
  #3  bm=128   bn=1024  bk=512   0.00125s  util=6.98%  vs XLA: 0.64x
  #4  bm=256   bn=1024  bk=1024  0.00126s  util=13.84%  vs XLA: 0.63x
  #5  bm=128   bn=512   bk=512   0.00127s  util=6.88%  vs XLA: 0.63x

=================================================================
(4096, 8192, 1024) - transpose_rhs=False - 64 configs tested
  XLA baseline: 0.00056s  util=61.84%
-----------------------------------------------------------------
  #1  bm=1024  bn=512   bk=1024  0.00057s  util=61.25%  vs XLA: 0.99x
  #2  bm=1024  bn=1024  bk=1024  0.00058s  util=60.39%  vs XLA: 0.98x
  #3  bm=1024  bn=256   bk=1024  0.00058s  util=60.00%  vs XLA: 0.97x
  #4  bm=512   bn=1024  bk=1024  0.00059s  util=58.89%  vs XLA: 0.95x
  #5  bm=1024  bn=128   bk=1024  0.00062s  util=56.26%  vs XLA: 0.91x

=================================================================
(4096, 8192, 1024) - transpose_rhs=True - 64 configs tested
  XLA baseline: 0.00064s  util=54.88%
-----------------------------------------------------------------
  #1  bm=1024  bn=1024  bk=1024  0.00064s  util=54.31%  vs XLA: 0.99x
  #2  bm=512   bn=1024  bk=1024  0.00065s  util=53.55%  vs XLA: 0.98x
  #3  bm=1024  bn=256   bk=1024  0.00066s  util=52.88%  vs XLA: 0.96x
  #4  bm=512   bn=512   bk=1024  0.00066s  util=52.63%  vs XLA: 0.96x
  #5  bm=1024  bn=128   bk=1024  0.00067s  util=52.35%  vs XLA: 0.95x
"""

# %%

"""
=================================================================
(8192, 8192, 8192) - transpose_rhs=False - 220 configs tested
  XLA baseline: 0.00621s  util=89.92%
-----------------------------------------------------------------
  #1  bm=1024  bn=2048  bk=1024  0.00605s  util=92.32%  vs XLA: 1.03x
  #2  bm=1024  bn=512   bk=2048  0.00605s  util=92.19%  vs XLA: 1.03x
  #3  bm=512   bn=2048  bk=2048  0.00606s  util=92.14%  vs XLA: 1.02x
  #4  bm=1024  bn=2048  bk=512   0.00607s  util=91.93%  vs XLA: 1.02x
  #5  bm=512   bn=1024  bk=2048  0.00608s  util=91.73%  vs XLA: 1.02x
"""

"""
"""
