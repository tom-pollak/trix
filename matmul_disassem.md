## Jaxpr

```css
{ lambda ; a:Ref{bf16[1024,2048]} b:Ref{bf16[2048,1024]} c:Ref{bf16[1024,1024]} d:Ref<vmem>{f32[1024,1024]}. let
    e:i32[] = program_id[axis=2]
    f:bool[] = eq e 0:i32[]
    g:i32[] = convert_element_type[new_dtype=int32 weak_type=False] f
    cond[
      branches=(
        { lambda ; h:Ref<vmem>{f32[1024,1024]}. let  in () }
        { lambda ; i:Ref<vmem>{f32[1024,1024]}. let
            j:f32[1024,1024] = broadcast_in_dim[
              broadcast_dimensions=()
              shape=(1024, 1024)
              sharding=None
            ] 0.0:f32[]
            i[:,:] <- j
          in () }
      )
    ] g d
    k:f32[1024,1024] <- d[:,:]
    l:bf16[1024,2048] <- a[:,:]
    m:bf16[2048,1024] <- b[:,:]
    n:f32[1024,1024] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] l m
    o:f32[1024,1024] = add k n
    d[:,:] <- o
    p:i32[] = program_id[axis=2]
    q:bool[] = eq p 3:i32[]
    r:i32[] = convert_element_type[new_dtype=int32 weak_type=False] q
    cond[
      branches=(
        { lambda ; s:Ref<vmem>{f32[1024,1024]} t:Ref{bf16[1024,1024]}. let

          in () }
        { lambda ; u:Ref<vmem>{f32[1024,1024]} v:Ref{bf16[1024,1024]}. let
            w:f32[1024,1024] <- u[:,:]
            x:bf16[1024,1024] = convert_element_type[
              new_dtype=bfloat16
              weak_type=False
            ] w
            v[:,:] <- x
          in () }
      )
    ] r d c
  in () }
```

## Mosaic

```javascript
The Mosaic module for pallas_call matmul_kernel at /tmp/ipykernel_40507/2097818761.py:1:
module @matmul_kernel {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: i32, // pids
                  %arg3: memref<1024x2048xbf16, #tpu.memory_space<vmem>>, // x tile
                  %arg4: memref<2048x1024xbf16, #tpu.memory_space<vmem>>, // y tile
                  %arg5: memref<1024x1024xbf16, #tpu.memory_space<vmem>>, // z tile
                  %arg6: memref<1024x1024xf32, #tpu.memory_space<vmem>>,  // acc scratch buffer
                  ),
                  attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>], // CompilerParams(dimension_semantics=)
                  iteration_bounds = array<i64: 8, 8, 4>, scalar_prefetch = 0 : i64, scratch_operands = 1 : i64, // grid, prefetch, scratch buffers
                  // block spec
                  window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 1024, 2048>}, {transform_indices = @transform_1, window_bounds = array<i64: 2048, 1024>},
                                   {transform_indices = @transform_2, window_bounds = array<i64: 1024, 1024>}]}
  {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 // pl.program_id(2) == 0
    %1 = arith.extui %0 : i1 to i32          // to i32 -- aka 1 if true
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.cmpi ne, %1, %c0_i32_0 : i32 // // (pl.program_id(2) == 0) or (1 != 0)

    // init accumulator with 0s
    scf.if %2 {  // if pl.program_id(2) == 0
      %cst_9 = arith.constant 0.000000e+00 : f32 // 0 f32
      %12 = vector.broadcast %cst_9 : f32 to vector<1024x1024xf32> // zeros(1024, 1024, f32)
      %c0_10 = arith.constant 0 : index
      %c0_11 = arith.constant 0 : index
      // dead code: load acc[0, 0] to vreg : f32[1024, 1024]
      %13 = vector.load %arg6[%c0_10, %c0_11] : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>
      // store acc[0, 0] zeros(1024, 1024, f32), dense strides, store vmem f32[1024, 1024], from a vreg f32[1024, 1024]
      tpu.vector_store %arg6[%c0_10, %c0_11], %12 {strides = array<i32>} : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>,
    } else {
    }
    %c0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    // load acc to vreg
    %3 = vector.load %arg6[%c0, %c0_1] : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    // load x tile
    %4 = vector.load %arg3[%c0_2, %c0_3] : memref<1024x2048xbf16, #tpu.memory_space<vmem>>, vector<1024x2048xbf16>
    %c0_4 = arith.constant 0 : index
    %c0_5 = arith.constant 0 : index
    // load y tile
    %5 = vector.load %arg4[%c0_4, %c0_5] : memref<2048x1024xbf16, #tpu.memory_space<vmem>>, vector<2048x1024xbf16>
    %cst = arith.constant dense<0.000000e+00> : vector<1024x1024xf32> // bias term?
    // do x @ y matmul
    // #tpu.dot_dimension_numbers<
    // [1], // lhs contracting
    // [0], // rhs contracting
    // [0], // lhs non-contracting
    // [1], // rhs non-contracting
    // [0, 0, 1, 1], // flat output dimension mapping of non-contracting dims
    // (0 -> 0) (1 -> 1) aka Z[M, N]
    // if this was [0, 1, 1, 0] => (0 -> 1) (1 -> 0) aka Z[N, M]
    // or for a transposed matmul [0, 0, 0, 1] X[M,K] @ Y[N,K] => Z[M,N]
    // [], // lhs batch
    // []  // rhs batch
    // >
    %6 = tpu.matmul %4, %5, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>} \
                                  : vector<1024x2048xbf16>, vector<2048x1024xbf16>, vector<1024x1024xf32> -> vector<1024x1024xf32>
    // add float acc += output
    %7 = arith.addf %3, %6 : vector<1024x1024xf32>
    %c0_6 = arith.constant 0 : index
    %c0_7 = arith.constant 0 : index
    // dead code: load acc buffer
    %8 = vector.load %arg6[%c0_6, %c0_7] : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>
    // store accumulated output
    tpu.vector_store %arg6[%c0_6, %c0_7], %7 {strides = array<i32>} : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>,
    %c3_i32 = arith.constant 3 : i32
    %9 = arith.cmpi eq, %arg2, %c3_i32 : i32 // pl.program_id(2) == 3
    %10 = arith.extui %9 : i1 to i32         // to i32
    %c0_i32_8 = arith.constant 0 : i32
    %11 = arith.cmpi ne, %10, %c0_i32_8 : i32 // if pl.program_id(2) == 3 aka (1 (true) != 0)
    // end of k loop: store acc to z
    scf.if %11 {
      %c0_9 = arith.constant 0 : index
      %c0_10 = arith.constant 0 : index
      // load acc
      %12 = vector.load %arg6[%c0_9, %c0_10] : memref<1024x1024xf32, #tpu.memory_space<vmem>>, vector<1024x1024xf32>
      // trucf: convert f32 to bf16
      %13 = arith.truncf %12 : vector<1024x1024xf32> to vector<1024x1024xbf16>
      %c0_11 = arith.constant 0 : index
      %c0_12 = arith.constant 0 : index
      // dead code: load z
      %14 = vector.load %arg5[%c0_11, %c0_12] : memref<1024x1024xbf16, #tpu.memory_space<vmem>>, vector<1024x1024xbf16>
      // store in z
      tpu.vector_store %arg5[%c0_11, %c0_12], %13 {strides = array<i32>} : memref<1024x1024xbf16, #tpu.memory_space<vmem>>, vector<1024x1024xbf16>,
    } else {
    }
    return
  }

  // block specs
  func.func @transform_0(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    return %arg0, %arg2 : i32, i32
  }
  func.func @transform_1(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    return %arg2, %arg1 : i32, i32
  }
  func.func @transform_2(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    return %arg0, %arg1 : i32, i32
  }
}
```

## Mosaic 005 -- post-canonicalize

> m, n, k = 8192, 8192, 8192
> bm, bn, bk = 1024, 2048, 2048

```javascript
module @matmul_kernel {
  func.func @main(
    %arg0: i32 loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), // pids
    // #tpu.tiled<(8,128)(2,1),[16,1]>
    // (8,128): hardware tile shape in vreg
    // (2,1): element packing, pack 2 bf16 in one 32-bit slot. vertical row are packed together
    // (16,1): tile stride (8, 128) over block <1024x2048>: [cdiv(2048, 128), 1]
    %arg3: memref<1024x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> loc(unknown), // x
    %arg4: memref<2048x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> loc(unknown), // y
    %arg5: memref<1024x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> loc(unknown), // z
    %arg6: memref<1024x2048xf32, #tpu.tiled<(8,128),[16,1]>, #tpu.memory_space<vmem>> loc(unknown))       // acc
    attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>],
                iteration_bounds = array<i64: 8, 4, 4>, scalar_prefetch = 0 : i64, scratch_operands = 1 : i64,
                window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 1024, 2048>},
                {transform_indices = @transform_1, window_bounds = array<i64: 2048, 2048>}, {transform_indices = @transform_2, window_bounds = array<i64: 1024, 2048>}]} {
    %c3_i32 = arith.constant 3 : i32 loc(#loc89)
    %cst = arith.constant dense<0.000000e+00> : vector<1024x2048xf32> loc(#loc)
    %c0 = arith.constant 0 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    // type erasure of tpu.tiled
    %0 = tpu.erase_memref_layout %arg3 : memref<1024x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> -> memref<1024x2048xbf16, #tpu.memory_space<vmem>> loc(#loc)
    %1 = tpu.erase_memref_layout %arg4 : memref<2048x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> -> memref<2048x2048xbf16, #tpu.memory_space<vmem>> loc(#loc)
    %2 = tpu.erase_memref_layout %arg5 : memref<1024x2048xbf16, #tpu.tiled<(8,128)(2,1),[16,1]>, #tpu.memory_space<vmem>> -> memref<1024x2048xbf16, #tpu.memory_space<vmem>> loc(#loc)
    %3 = tpu.erase_memref_layout %arg6 : memref<1024x2048xf32, #tpu.tiled<(8,128),[16,1]>, #tpu.memory_space<vmem>> -> memref<1024x2048xf32, #tpu.memory_space<vmem>> loc(#loc)
    // if pl.program_id(2) == 0
    %4 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc90)
    scf.if %4 {
      // store to acc with %cst -- zeros
      tpu.vector_store %3[%c0, %c0], %cst {strides = array<i32>} : memref<1024x2048xf32, #tpu.memory_space<vmem>>, vector<1024x2048xf32>,  loc(#loc92)
    } loc(#loc91)
    // load acc, x, y
    %5 = vector.load %3[%c0, %c0] : memref<1024x2048xf32, #tpu.memory_space<vmem>>, vector<1024x2048xf32> loc(#loc93)
    %6 = vector.load %0[%c0, %c0] : memref<1024x2048xbf16, #tpu.memory_space<vmem>>, vector<1024x2048xbf16> loc(#loc94)
    %7 = vector.load %1[%c0, %c0] : memref<2048x2048xbf16, #tpu.memory_space<vmem>>, vector<2048x2048xbf16> loc(#loc95)
    // matmul y,x, %cst is bias? not using accumulator as bias
    %8 = tpu.matmul %6, %7, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>} : vector<1024x2048xbf16>, vector<2048x2048xbf16>, vector<1024x2048xf32> -> vector<1024x2048xf32> loc(#loc96)
    // add accumulator
    %9 = arith.addf %5, %8 : vector<1024x2048xf32> loc(#loc97)
    // sotre in accumulator
    tpu.vector_store %3[%c0, %c0], %9 {strides = array<i32>} : memref<1024x2048xf32, #tpu.memory_space<vmem>>, vector<1024x2048xf32>,  loc(#loc98)
    // if tl.program_id(2) == 3
    %10 = arith.cmpi eq, %arg2, %c3_i32 : i32 loc(#loc89)
    scf.if %10 {
      // load accumualtor
      %11 = vector.load %3[%c0, %c0] : memref<1024x2048xf32, #tpu.memory_space<vmem>>, vector<1024x2048xf32> loc(#loc100)
      // convert to f32
      %12 = tpu.truncf %11 {rounding_mode = #tpu.rounding_mode<to_nearest_even>} : vector<1024x2048xf32> -> vector<1024x2048xbf16> loc(#loc101)
      // store to z
      tpu.vector_store %2[%c0, %c0], %12 {strides = array<i32>} : memref<1024x2048xbf16, #tpu.memory_space<vmem>>, vector<1024x2048xbf16>,  loc(#loc102)
    } loc(#loc99)
    return loc(#loc)
  } loc(#loc)
  func.func @transform_0(%arg0: i32 loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown)) -> (i32, i32) {
    return %arg0, %arg2 : i32, i32 loc(#loc)
  } loc(#loc)
  func.func @transform_1(%arg0: i32 loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown)) -> (i32, i32) {
    return %arg2, %arg1 : i32, i32 loc(#loc)
  } loc(#loc)
  func.func @transform_2(%arg0: i32 loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown)) -> (i32, i32) {
    return %arg0, %arg1 : i32, i32 loc(#loc)
  } loc(#loc)
} loc(#loc)
```
