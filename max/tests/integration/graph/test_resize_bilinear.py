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
"""Test resize bilinear operation execution on CPU and GPU."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray


def torch_bilinear_resize_nchw(
    input_array: NDArray[np.float32], height: int, width: int
) -> NDArray[np.float32]:
    """Reference implementation using torch bilinear resize for NCHW format.

    Uses ``align_corners=False`` (half-pixel coordinate mapping), matching the
    default ``coordinate_transform_mode`` of ``ops.resize`` with bilinear
    interpolation.

    Args:
        input_array: Input array of shape (batch, channels, height, width).
        height: Target height.
        width: Target width.

    Returns:
        Resized array using torch's bilinear interpolation in NCHW format.
    """
    input_tensor = torch.from_numpy(input_array)
    output_tensor = F.interpolate(
        input_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return output_tensor.numpy()


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # Upscale 2x (NCHW format)
        ([1, 3, 224, 224], [1, 3, 448, 448]),
        # Downscale 2x
        ([1, 3, 448, 448], [1, 3, 224, 224]),
        # Non-square input
        ([1, 3, 336, 224], [1, 3, 448, 448]),
        # Different batch / channels
        ([2, 1, 128, 128], [2, 1, 256, 256]),
    ],
)
def test_resize_bilinear_execution(
    session: InferenceSession,
    device: DeviceRef,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test bilinear resize compilation and execution against torch."""
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(
        dtype=DType.float32, shape=input_shape, device=device
    )

    with Graph("test_resize_bilinear", input_types=[input_type]) as graph:
        resized = ops.resize(
            graph.inputs[0].tensor,
            output_shape,
            interpolation=ops.InterpolationMode.BILINEAR,
        )
        graph.output(resized)

    model = session.load(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    _, _, out_h, out_w = output_shape
    expected = torch_bilinear_resize_nchw(input_data, out_h, out_w)

    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    result_np = result.to_numpy()

    assert result_np.shape == tuple(output_shape)
    np.testing.assert_allclose(result_np, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
def test_resize_bilinear_identity(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Test identity transformation (same size copies the input)."""
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    shape = [1, 3, 256, 256]  # NCHW format
    input_type = TensorType(dtype=DType.float32, shape=shape, device=device)

    with Graph(
        "test_resize_bilinear_identity", input_types=[input_type]
    ) as graph:
        resized = ops.resize(
            graph.inputs[0].tensor,
            shape,
            interpolation=ops.InterpolationMode.BILINEAR,
        )
        graph.output(resized)

    model = session.load(graph)

    input_data = np.random.rand(*shape).astype(np.float32)
    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy(), input_data, rtol=1e-5, atol=1e-6
    )
