# %%
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

INTERPRET = jax.default_backend() == "cpu"

mesh = jax.make_mesh((4,), ("x",), (jax.sharding.AxisType.Explicit,))
jax.set_mesh(mesh)
assert jax.device_count() == 4

# %%


def comm_kernel(input_ref, output_ref, send_sem, recv_sem):
    device_id = jax.lax.axis_index("x")
    copy_0_to_1 = pltpu.make_async_remote_copy(
        src_ref=input_ref,
        dst_ref=output_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=1,
    )
    copy_2_to_3 = pltpu.make_async_remote_copy(
        src_ref=input_ref,
        dst_ref=output_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=3,
    )
    copy_3_to_2 = pltpu.make_async_remote_copy(
        src_ref=input_ref,
        dst_ref=output_ref,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=2,
    )

    @pl.when(device_id == 0)
    def _():
        copy_0_to_1.start()
        copy_0_to_1.wait_send()

    @pl.when(device_id == 1)
    def _():
        copy_0_to_1.wait_recv()

    @pl.when(device_id == 2)
    def _():
        copy_2_to_3.start()
        copy_2_to_3.wait_send()
        copy_3_to_2.wait_recv()

    @pl.when(device_id == 3)
    def _():
        copy_3_to_2.start()
        copy_3_to_2.wait_send()
        copy_2_to_3.wait_recv()


num_devices = jax.device_count()
x = jnp.arange(0, 128, dtype=jnp.int32)
x = jax.device_put(x, jax.P("x"))


@jax.jit
@jax.shard_map(in_specs=jax.P("x"), out_specs=jax.P("x"), check_vma=False)
def call_comm(x_s):
    return pl.pallas_call(
        comm_kernel,
        out_shape=jax.ShapeDtypeStruct(x_s.shape, dtype=x_s.dtype, sharding=jax.P("x")),
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 2,
        interpret=INTERPRET,
    )(x_s)


call_comm(x)


# %%
