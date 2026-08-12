# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, coord, row_major

from nn.resize import CoordinateTransformationMode, resize_linear
from std.testing import assert_almost_equal


def _run_cpu_vs_gpu[
    dtype: DType,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
](input_dim: Coord, output_dim: Coord, ctx: DeviceContext) raises where (
    input_dim.all_dims_known and output_dim.all_dims_known
):
    """Runs linear resize on CPU and GPU and asserts the results match.

    `resize_linear` dispatches through `elementwise`, which runs on the host
    for host buffers and on the device for device buffers. Both paths must be
    numerically identical for every coordinate-transformation mode, antialias
    setting, and resize-dimension combination.
    """
    var in_n = input_dim.static_product
    var out_n = output_dim.static_product

    var in_host = ctx.enqueue_create_host_buffer[dtype](in_n)
    var out_cpu_host = ctx.enqueue_create_host_buffer[dtype](out_n)
    var out_gpu_host = ctx.enqueue_create_host_buffer[dtype](out_n)
    ctx.synchronize()

    # Fill the input with a deterministic ramp.
    for i in range(in_n):
        in_host[i] = Scalar[dtype](i)

    # CPU reference on host memory.
    var in_host_t = TileTensor(in_host, row_major(input_dim))
    var out_cpu_t = TileTensor(out_cpu_host, row_major(output_dim))
    resize_linear[coordinate_transformation_mode, antialias, target="cpu"](
        in_host_t, out_cpu_t, ctx
    )

    # GPU on device memory.
    var in_dev = ctx.enqueue_create_buffer[dtype](in_n)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_n)
    var in_dev_t = TileTensor(in_dev, row_major(input_dim))
    var out_dev_t = TileTensor(out_dev, row_major(output_dim))

    ctx.enqueue_copy(in_dev, in_host)
    resize_linear[coordinate_transformation_mode, antialias, target="gpu"](
        in_dev_t, out_dev_t, ctx
    )
    ctx.enqueue_copy(out_gpu_host, out_dev)
    ctx.synchronize()

    for i in range(out_n):
        assert_almost_equal(
            out_cpu_host[i], out_gpu_host[i], atol=1e-4, rtol=1e-4
        )

    _ = in_dev^
    _ = out_dev^


def test_gpu_vs_cpu(ctx: DeviceContext) raises:
    print("== test_gpu_vs_cpu")

    # Bilinear upsample (half_pixel).
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, False
    ](coord[1, 1, 2, 2], coord[1, 1, 4, 4], ctx)

    # Bilinear upsample (align_corners).
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.AlignCorners, False
    ](coord[1, 1, 2, 2], coord[1, 1, 4, 4], ctx)

    # Bilinear upsample (asymmetric), non-square.
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.Asymmetric, False
    ](coord[1, 3, 4, 4], coord[1, 3, 9, 7], ctx)

    # Bilinear downsample (half_pixel).
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, False
    ](coord[1, 2, 8, 8], coord[1, 2, 3, 5], ctx)

    # Antialiased downsample.
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, True
    ](coord[1, 1, 4, 4], coord[1, 1, 2, 2], ctx)

    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, True
    ](coord[1, 2, 8, 6], coord[1, 2, 3, 2], ctx)

    # Trilinear-style resize across channel, height, and width dimensions.
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, False
    ](coord[1, 4, 2, 2], coord[1, 6, 4, 4], ctx)

    # Identity (no resize) must copy input to output.
    _run_cpu_vs_gpu[
        DType.float32, CoordinateTransformationMode.HalfPixel, False
    ](coord[1, 3, 5, 5], coord[1, 3, 5, 5], ctx)

    print("test_gpu_vs_cpu passed")


def test_gpu_vs_torch_reference(ctx: DeviceContext) raises:
    """Checks the GPU output against PyTorch's `interpolate(mode="bilinear")`.

    Guards against the CPU and GPU paths sharing the same defect: values come
    from an independent PyTorch reference rather than the CPU kernel.

    TORCH REFERENCE:
        x = np.array([[[[1, 2], [3, 4]]]])
        y = torch.nn.functional.interpolate(torch.Tensor(x), (4, 4),
                                             mode="bilinear")
    """
    print("== test_gpu_vs_torch_reference")
    comptime input_dim = coord[1, 1, 2, 2]
    comptime output_dim = coord[1, 1, 4, 4]

    var reference = [
        Float32(1.0000),
        1.2500,
        1.7500,
        2.0000,
        1.5000,
        1.7500,
        2.2500,
        2.5000,
        2.5000,
        2.7500,
        3.2500,
        3.5000,
        3.0000,
        3.2500,
        3.7500,
        4.0000,
    ]

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](4)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](16)
    ctx.synchronize()

    for i in range(4):
        in_host[i] = Float32(i + 1)

    var in_dev = ctx.enqueue_create_buffer[DType.float32](4)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](16)
    var in_dev_t = TileTensor(in_dev, row_major(input_dim))
    var out_dev_t = TileTensor(out_dev, row_major(output_dim))

    ctx.enqueue_copy(in_dev, in_host)
    resize_linear[CoordinateTransformationMode.HalfPixel, False, target="gpu"](
        in_dev_t, out_dev_t, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(16):
        assert_almost_equal(out_host[i], reference[i], atol=1e-5, rtol=1e-4)

    _ = in_dev^
    _ = out_dev^
    print("test_gpu_vs_torch_reference passed")


def main() raises:
    with DeviceContext() as ctx:
        test_gpu_vs_cpu(ctx)
        test_gpu_vs_torch_reference(ctx)
