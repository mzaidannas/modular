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
"""Implements tensor resize (upsample/downsample) with nearest, bilinear, and other interpolation modes."""

from std.math import ceil, floor


from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor

from std.utils import IndexList, StaticTuple


struct CoordinateTransformationMode(ImplicitlyCopyable):
    """Specifies how output coordinates map to input coordinates during resize.
    """

    var value: Int
    comptime HalfPixel = CoordinateTransformationMode(0)
    comptime AlignCorners = CoordinateTransformationMode(1)
    comptime Asymmetric = CoordinateTransformationMode(2)
    comptime HalfPixel1D = CoordinateTransformationMode(3)

    @always_inline
    def __init__(out self, value: Int):
        self.value = value

    @always_inline
    def __eq__(self, other: CoordinateTransformationMode) -> Bool:
        return self.value == other.value


@__parameter
@always_inline
def coord_transform[
    mode: CoordinateTransformationMode
](out_coord: Int, in_dim: Int, out_dim: Int, scale: Float32) -> Float32:
    """Maps an output coordinate to an input coordinate according to the given transformation mode.

    Parameters:
        mode: The coordinate transformation mode governing the mapping.

    Args:
        out_coord: The output coordinate to map.
        in_dim: The size of the input dimension.
        out_dim: The size of the output dimension.
        scale: The ratio of output dimension size to input dimension size.

    Returns:
        The corresponding input coordinate as a floating-point value.
    """
    var out_coord_f32 = Float32(out_coord)

    comptime if mode == CoordinateTransformationMode.HalfPixel:
        # note: coordinates are for the CENTER of the pixel
        # - 0.5 term at the end is so that when we round to the nearest integer
        # coordinate, we get the coordinate whose center is closest
        return (out_coord_f32 + Float32(0.5)) / scale - 0.5
    elif mode == CoordinateTransformationMode.HalfPixel1D:
        # Same as HalfPixel except for 1D output. Described here:
        # https://onnx.ai/onnx/operators/onnx__Resize.html
        if out_dim == 1:
            return 0
        return (out_coord_f32 + Float32(0.5)) / scale - 0.5
    elif mode == CoordinateTransformationMode.AlignCorners:
        # aligning "corners" when output is 1D isn't well defined
        # this matches pytorch
        if out_dim == 1:
            return 0
        # note: resized image will have same corners as original image
        # use Float32 throughout so the kernel stays valid on GPU targets
        # (Metal rejects Float64), matching the CPU path within tolerance.
        return out_coord_f32 * (Float32(in_dim - 1) / Float32(out_dim - 1))
    elif mode == CoordinateTransformationMode.Asymmetric:
        return out_coord_f32 / scale
    else:
        comptime assert False, "coordinate_transformation_mode not implemented"


struct RoundMode(ImplicitlyCopyable):
    """Specifies how fractional coordinates are rounded to integer indices during nearest-neighbor resize.
    """

    var value: Int
    comptime HalfDown = RoundMode(0)
    comptime HalfUp = RoundMode(1)
    comptime Floor = RoundMode(2)
    comptime Ceil = RoundMode(3)

    @always_inline
    def __init__(out self, value: Int):
        self.value = value

    @always_inline
    def __eq__(self, other: RoundMode) -> Bool:
        return self.value == other.value


@fieldwise_init
struct InterpolationMode(ImplicitlyCopyable):
    """Specifies the interpolation method used during resize."""

    var value: Int
    comptime Linear = InterpolationMode(0)

    @always_inline
    def __eq__(self, other: InterpolationMode) -> Bool:
        return self.value == other.value


struct Interpolator[mode: InterpolationMode](
    Defaultable, TrivialRegisterPassable
):
    """Holds interpolation filter state and applies the filter for a given interpolation mode.
    """

    var cubic_coeff: Float32

    @always_inline
    def __init__(out self, cubic_coeff: Float32):
        self.cubic_coeff = cubic_coeff

    @always_inline
    def __init__(out self):
        self.cubic_coeff = 0

    @staticmethod
    @always_inline
    def filter_length() -> Int:
        comptime assert (
            Self.mode == InterpolationMode.Linear
        ), "InterpolationMode not supported"
        return 1

    @always_inline
    def filter(self, x: Float32) -> Float32:
        comptime assert (
            Self.mode == InterpolationMode.Linear
        ), "InterpolationMode not supported"
        return linear_filter(x)


def resize_nearest_neighbor[
    coordinate_transformation_mode: CoordinateTransformationMode,
    round_mode: RoundMode,
    dtype: DType,
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Resizes input to output shape using nearest-neighbor interpolation.

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        round_mode: How to round fractional input coordinates to integer indices.
        dtype: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
        ctx: The device context used to launch the kernel.
    """
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"
    var scales = StaticTuple[Float32, input.rank]()
    for i in range(input.rank):
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()

    @__parameter
    @always_inline
    def round[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
        comptime if round_mode == RoundMode.HalfDown:
            return ceil(val - 0.5)
        elif round_mode == RoundMode.HalfUp:
            return floor(val + 0.5)
        elif round_mode == RoundMode.Floor:
            return floor(val)
        elif round_mode == RoundMode.Ceil:
            return ceil(val)
        else:
            comptime assert False, "round_mode not implemented"

    def nn_interpolate[
        simd_width: Int, alignment: Int = 1
    ](out_coords: Coord) {var}:
        var in_coords = IndexList[input.rank](0)

        comptime for i in range(input.rank):
            in_coords[i] = min(
                Int(
                    round(
                        coord_transform[coordinate_transformation_mode](
                            Int(out_coords[i].value()),
                            Int(input.dim(i)),
                            Int(output.dim(i)),
                            scales[i],
                        )
                    )
                ),
                Int(input.dim(i)) - 1,
            )

        var in_idx = input.layout(Coord(in_coords))
        var out_idx = output.layout(out_coords)

        output.raw_store(out_idx, input.ptr[in_idx])

    # TODO (#21439): can use unsafe_memcpy when scale on inner dimension is 1
    elementwise[1](nn_interpolate, output.layout.shape_coord(), ctx)


@always_inline
def linear_filter(x: Float32) -> Float32:
    """This is a tent filter.

    f(x) = 1 + x, x < 0
    f(x) = 1 - x, 0 <= x < 1
    f(x) = 0, x >= 1

    """
    var coeff = x
    if x < 0:
        coeff = -x
    if x < 1:
        return 1 - coeff
    return 0


def resize_linear[
    dtype: DType,
    //,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    target: StaticString = "cpu",
](
    input: TileTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    output: TileTensor[
        mut=True, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Resizes input to output shape using linear interpolation.

    Linear interpolation is separable, so an N-D output value is the weighted
    sum over the Cartesian product of the per-dimension 1-D filter windows,
    normalized by the total accumulated weight. This is expressed as a single
    `elementwise` pass over the output, so the same code runs on CPU or GPU
    depending on `target`, without allocating intermediate buffers.

    Parameters:
        dtype: Type of input and output (inferred).
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        antialias: Whether or not to use an antialiasing linear/cubic filter, which when downsampling, uses
            more points to avoid aliasing artifacts. Effectively stretches the filter by a factor of 1 / scale.
        target: `StaticString` identifying the execution platform, used to
            select between the GPU and CPU code paths.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
        ctx: The device context used to launch the kernel.
    """
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"

    comptime rank = input.rank
    var interpolator = Interpolator[InterpolationMode.Linear]()

    # Precomputed on the host; captured by value into the kernel so no Float64
    # arithmetic runs on the device (Metal rejects Float64).
    var scales = StaticTuple[Float32, rank]()
    for i in range(rank):
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()

    def linear_interpolate[
        simd_width: Int, alignment: Int = 1
    ](out_coords: Coord) {var}:
        # Build the per-dimension interpolation window.
        var win_min = IndexList[rank](0)
        var win_count = IndexList[rank](0)
        var centers = StaticTuple[Float32, rank]()
        var inv_filter_scale = StaticTuple[Float32, rank]()
        var total_taps = 1
        comptime for d in range(rank):
            var out_coord = Int(out_coords[d].value())
            var in_dim = Int(input.dim(d))
            var out_dim = Int(output.dim(d))
            if in_dim == out_dim:
                # Dimension is not resized: a single unit-weight tap that copies
                # the input coordinate through unchanged.
                win_min[d] = out_coord
                win_count[d] = 1
                centers[d] = Float32(out_coord) + Float32(0.5)
                inv_filter_scale[d] = 1
            else:
                var center = coord_transform[coordinate_transformation_mode](
                    out_coord, in_dim, out_dim, scales[d]
                ) + Float32(0.5)
                var filter_scale = 1 / scales[d] if antialias and scales[
                    d
                ] < 1 else Float32(1)
                var support = (
                    Float32(interpolator.filter_length()) * filter_scale
                )
                win_min[d] = max(Int(center - support + 0.5), 0)
                win_count[d] = (
                    min(in_dim, Int(center + support + 0.5)) - win_min[d]
                )
                centers[d] = center
                inv_filter_scale[d] = 1 / filter_scale
            total_taps *= win_count[d]

        # Accumulate over the Cartesian product of the per-dimension windows.
        var acc = Float32(0)
        var weight_sum = Float32(0)
        for tap in range(total_taps):
            var rem = tap
            var in_coords = IndexList[rank](0)
            var weight = Float32(1)
            comptime for d in range(rank):
                var k = rem % win_count[d]
                rem //= win_count[d]
                var idx = win_min[d] + k
                in_coords[d] = idx
                var dist = (
                    (Float32(idx) + Float32(0.5)) - centers[d]
                ) * inv_filter_scale[d]
                weight *= interpolator.filter(dist)
            var in_idx = input.layout(Coord(in_coords))
            acc += input.raw_load(in_idx).cast[DType.float32]() * weight
            weight_sum += weight

        # Normalize; handles image boundaries where only some taps are in range.
        var out_idx = output.layout(out_coords)
        output.raw_store(out_idx, (acc / weight_sum).cast[dtype]())

    elementwise[1, target=target](
        linear_interpolate, output.layout.shape_coord(), ctx
    )
