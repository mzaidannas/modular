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
"""Validates the vendor-neutral gather-based transposed-convolution GPU kernel.

Runs `conv_transposed_gpu_native` on the device and compares against the
`conv_transpose_naive` CPU reference. The native kernel is vendor-neutral, so
this test exercises it on whatever GPU the `DeviceContext` selects (Apple
Silicon, AMD, or NVIDIA), independent of the cuDNN path.
"""

from std.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from nn.conv.conv_transpose import (
    conv_transpose_naive,
    conv_transposed_gpu_native,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_conv_transposed_native[
    H: Int,
    W: Int,
    R: Int,
    S: Int,
    in_channels: Int = 1,
    out_channels: Int = 1,
    N: Int = 1,
    stride_val: Int = 1,
    dilation_val: Int = 1,
    pad_val: Int = 0,
    dtype: DType = DType.float32,
](ctx: DeviceContext) raises:
    """Compares the native gather kernel against the naive reference for one 2D
    transposed-convolution configuration.

    Parameters:
        H: Input height.
        W: Input width.
        R: Filter height.
        S: Filter width.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        N: Batch size.
        stride_val: Stride along both spatial axes.
        dilation_val: Dilation along both spatial axes.
        pad_val: Symmetric padding along both spatial axes.
        dtype: Element type used throughout.

    Args:
        ctx: Device context used to launch the kernel.
    """
    comptime HO = stride_val * (H - 1) + dilation_val * (
        R - 1
    ) - 2 * pad_val + 1
    comptime WO = stride_val * (W - 1) + dilation_val * (
        S - 1
    ) - 2 * pad_val + 1

    print(
        "N=",
        N,
        " H=",
        H,
        " W=",
        W,
        " R=",
        R,
        " S=",
        S,
        " Cin=",
        in_channels,
        " Cout=",
        out_channels,
        " stride=",
        stride_val,
        " dilation=",
        dilation_val,
        " pad=",
        pad_val,
        " -> HO=",
        HO,
        " WO=",
        WO,
    )

    comptime input_size = N * H * W * in_channels
    comptime filter_size = R * S * out_channels * in_channels
    comptime output_size = N * HO * WO * out_channels

    # NHWC input, RSFC filter, NHWC output.
    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_ref_ptr = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_dev_host_ptr = ctx.enqueue_create_host_buffer[dtype](output_size)

    var input_host = TileTensor(
        input_host_ptr, row_major(Coord(IndexList[4](N, H, W, in_channels)))
    )
    var filter_host = TileTensor(
        filter_host_ptr,
        row_major(Coord(IndexList[4](R, S, out_channels, in_channels))),
    )
    random(input_host)
    random(filter_host)
    ctx.synchronize()

    # ---- Reference: naive CPU kernel (NDHWC / QRSFC with D = Q = 1). ----
    conv_transpose_naive[dtype](
        TileTensor(
            output_ref_ptr,
            row_major(Coord(IndexList[5](N, 1, HO, WO, out_channels))),
        ),
        TileTensor(
            input_host_ptr,
            row_major(Coord(IndexList[5](N, 1, H, W, in_channels))),
        ),
        TileTensor(
            filter_host_ptr,
            row_major(Coord(IndexList[5](1, R, S, out_channels, in_channels))),
        ),
        Index(1, stride_val, stride_val),
        Index(1, dilation_val, dilation_val),
        Index(0, 0),  # pad_d
        Index(pad_val, pad_val),  # pad_h
        Index(pad_val, pad_val),  # pad_w
    )

    # ---- Device: native gather kernel. ----
    var d_input = ctx.enqueue_create_buffer[dtype](input_size)
    var d_filter = ctx.enqueue_create_buffer[dtype](filter_size)
    var d_output = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(d_input, input_host_ptr)
    ctx.enqueue_copy(d_filter, filter_host_ptr)

    conv_transposed_gpu_native[dtype, dtype, dtype](
        TileTensor(
            d_output, row_major(Coord(IndexList[4](N, HO, WO, out_channels)))
        ),
        TileTensor(
            d_input, row_major(Coord(IndexList[4](N, H, W, in_channels)))
        ),
        TileTensor(
            d_filter,
            row_major(Coord(IndexList[4](R, S, out_channels, in_channels))),
        ),
        Index(stride_val, stride_val),
        Index(dilation_val, dilation_val),
        Index(0, 0),  # pad_d (unused for 2D)
        Index(pad_val, pad_val),  # pad_h
        Index(pad_val, pad_val),  # pad_w
        ctx,
    )

    ctx.enqueue_copy(output_dev_host_ptr, d_output)
    ctx.synchronize()

    # ---- Compare. ----
    for i in range(output_size):
        assert_almost_equal(
            output_ref_ptr[i], output_dev_host_ptr[i], rtol=1e-4
        )
    print("  Succeed")

    _ = d_input^
    _ = d_filter^
    _ = d_output^


def main() raises:
    with DeviceContext() as ctx:
        # Basic 1x1 in/out channel sweeps over stride / dilation / padding.
        test_conv_transposed_native[H=5, W=7, R=3, S=3](ctx)
        test_conv_transposed_native[H=5, W=7, R=3, S=3, stride_val=2](ctx)
        test_conv_transposed_native[H=5, W=7, R=4, S=4, stride_val=2](ctx)
        test_conv_transposed_native[H=5, W=7, R=3, S=3, dilation_val=2](ctx)
        test_conv_transposed_native[H=6, W=6, R=3, S=3, pad_val=1](ctx)
        test_conv_transposed_native[
            H=6, W=6, R=4, S=4, stride_val=2, pad_val=1
        ](ctx)
        test_conv_transposed_native[
            H=6, W=6, R=3, S=3, stride_val=2, dilation_val=2, pad_val=1
        ](ctx)

        # Multi-channel / multi-batch, incl. a non-SIMD-aligned channel count
        # to exercise the scalar remainder tail of the C contraction.
        test_conv_transposed_native[
            H=8, W=8, R=3, S=3, in_channels=8, out_channels=16, stride_val=2
        ](ctx)
        test_conv_transposed_native[
            H=8,
            W=8,
            R=3,
            S=3,
            in_channels=3,
            out_channels=5,
            stride_val=2,
            pad_val=1,
        ](ctx)
        test_conv_transposed_native[
            H=4,
            W=5,
            R=3,
            S=3,
            in_channels=6,
            out_channels=4,
            N=2,
            stride_val=2,
            pad_val=1,
        ](ctx)
        print("All native conv_transpose tests passed.")
