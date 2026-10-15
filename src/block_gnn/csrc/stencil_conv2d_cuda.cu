#include <algorithm>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void stencil_conv2d_direct_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t out_h,
    int64_t out_w,
    int64_t padding_h,
    int64_t padding_w) {
  using acc_t = at::acc_type<scalar_t, true>;
  const int64_t total = batch * out_channels * out_h * out_w;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total;
       linear += blockDim.x * gridDim.x) {
    int64_t remainder = linear;
    const int64_t out_x = remainder % out_w;
    remainder /= out_w;
    const int64_t out_y = remainder % out_h;
    remainder /= out_h;
    const int64_t out_channel = remainder % out_channels;
    const int64_t sample = remainder / out_channels;

    acc_t sum = bias == nullptr ? acc_t(0) : static_cast<acc_t>(bias[out_channel]);
    for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
      for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
        const int64_t in_y = out_y - padding_h + kernel_y;
        if (in_y < 0 || in_y >= in_h) {
          continue;
        }
        for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
          const int64_t in_x = out_x - padding_w + kernel_x;
          if (in_x < 0 || in_x >= in_w) {
            continue;
          }
          const int64_t input_index =
              ((sample * in_channels + in_channel) * in_h + in_y) * in_w + in_x;
          const int64_t weight_index =
              ((out_channel * in_channels + in_channel) * kernel_h + kernel_y) *
                  kernel_w +
              kernel_x;
          sum += static_cast<acc_t>(input[input_index]) *
              static_cast<acc_t>(weight[weight_index]);
        }
      }
    }
    output[linear] = static_cast<scalar_t>(sum);
  }
}

template <typename scalar_t>
__global__ void stencil_conv2d_tiled_single_channel_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t in_h,
    int64_t in_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t out_h,
    int64_t out_w,
    int64_t padding_h,
    int64_t padding_w) {
  using acc_t = at::acc_type<scalar_t, true>;
  extern __shared__ unsigned char shared_bytes[];
  auto* tile = reinterpret_cast<scalar_t*>(shared_bytes);

  const int64_t tile_h = blockDim.y + kernel_h - 1;
  const int64_t tile_w = blockDim.x + kernel_w - 1;
  const int64_t tile_elements = tile_h * tile_w;
  const int64_t block_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int64_t block_threads = blockDim.x * blockDim.y;
  const int64_t input_origin_y = blockIdx.y * blockDim.y - padding_h;
  const int64_t input_origin_x = blockIdx.x * blockDim.x - padding_w;
  const int64_t sample = blockIdx.z;

  for (int64_t index = block_linear; index < tile_elements; index += block_threads) {
    const int64_t tile_y = index / tile_w;
    const int64_t tile_x = index % tile_w;
    const int64_t in_y = input_origin_y + tile_y;
    const int64_t in_x = input_origin_x + tile_x;
    scalar_t value = scalar_t(0);
    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
      value = input[(sample * in_h + in_y) * in_w + in_x];
    }
    tile[index] = value;
  }
  __syncthreads();

  const int64_t out_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t out_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_y >= out_h || out_x >= out_w) {
    return;
  }

  acc_t sum = bias == nullptr ? acc_t(0) : static_cast<acc_t>(bias[0]);
  for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
      sum += static_cast<acc_t>(tile[(threadIdx.y + kernel_y) * tile_w +
                                     threadIdx.x + kernel_x]) *
          static_cast<acc_t>(weight[kernel_y * kernel_w + kernel_x]);
    }
  }
  output[(sample * out_h + out_y) * out_w + out_x] = static_cast<scalar_t>(sum);
}

struct ConvGeometry {
  int64_t batch;
  int64_t in_channels;
  int64_t in_h;
  int64_t in_w;
  int64_t out_channels;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t out_h;
  int64_t out_w;
};

ConvGeometry validate_cuda_inputs(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t padding_h,
    int64_t padding_w) {
  TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
  TORCH_CHECK(input.dim() == 4, "input must be NCHW");
  TORCH_CHECK(weight.dim() == 4, "weight must be OIHW");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input/weight dtype mismatch");
  TORCH_CHECK(input.size(1) == weight.size(1), "input/weight channel mismatch");
  TORCH_CHECK(input.is_floating_point(), "only floating-point inputs are supported");
  TORCH_CHECK(input.get_device() == weight.get_device(), "input/weight device mismatch");
  TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(bias.value().get_device() == input.get_device(), "bias device mismatch");
    TORCH_CHECK(bias.value().scalar_type() == input.scalar_type(), "bias dtype mismatch");
    TORCH_CHECK(bias.value().numel() == weight.size(0), "bias size mismatch");
  }

  const int64_t out_h = input.size(2) + 2 * padding_h - weight.size(2) + 1;
  const int64_t out_w = input.size(3) + 2 * padding_w - weight.size(3) + 1;
  TORCH_CHECK(out_h > 0 && out_w > 0, "invalid output shape");
  return {
      input.size(0), input.size(1), input.size(2), input.size(3), weight.size(0),
      weight.size(2), weight.size(3), out_h, out_w};
}

torch::Tensor stencil_conv2d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t padding_h,
    int64_t padding_w) {
  const auto geometry = validate_cuda_inputs(input, weight, bias, padding_h, padding_w);
  c10::cuda::CUDAGuard device_guard(input.device());
  auto x = input.contiguous();
  auto w = weight.contiguous();
  c10::optional<torch::Tensor> b =
      bias.has_value() ? c10::optional<torch::Tensor>(bias.value().contiguous()) : c10::nullopt;
  auto output = torch::empty(
      {geometry.batch, geometry.out_channels, geometry.out_h, geometry.out_w}, x.options());

  constexpr int threads = 256;
  const int64_t total = output.numel();
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "stencil_conv2d_cuda", [&] {
    stencil_conv2d_direct_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        w.data_ptr<scalar_t>(),
        b.has_value() ? b.value().data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        geometry.batch,
        geometry.in_channels,
        geometry.in_h,
        geometry.in_w,
        geometry.out_channels,
        geometry.kernel_h,
        geometry.kernel_w,
        geometry.out_h,
        geometry.out_w,
        padding_h,
        padding_w);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor stencil_conv2d_tiled_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t padding_h,
    int64_t padding_w) {
  const auto geometry = validate_cuda_inputs(input, weight, bias, padding_h, padding_w);
  TORCH_CHECK(
      geometry.in_channels == 1 && geometry.out_channels == 1,
      "tiled CUDA kernel currently requires one input and one output channel");
  TORCH_CHECK(
      geometry.kernel_h <= 31 && geometry.kernel_w <= 31,
      "tiled CUDA kernel supports kernels up to 31x31");
  TORCH_CHECK(geometry.batch <= 65535, "batch is too large for the tiled CUDA grid");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto x = input.contiguous();
  auto w = weight.contiguous();
  c10::optional<torch::Tensor> b =
      bias.has_value() ? c10::optional<torch::Tensor>(bias.value().contiguous()) : c10::nullopt;
  auto output = torch::empty(
      {geometry.batch, geometry.out_channels, geometry.out_h, geometry.out_w}, x.options());

  constexpr int tile_x = 16;
  constexpr int tile_y = 16;
  const dim3 threads(tile_x, tile_y);
  const dim3 blocks(
      (geometry.out_w + tile_x - 1) / tile_x,
      (geometry.out_h + tile_y - 1) / tile_y,
      geometry.batch);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "stencil_conv2d_tiled_cuda", [&] {
    const size_t shared_bytes =
        (tile_y + geometry.kernel_h - 1) * (tile_x + geometry.kernel_w - 1) *
        sizeof(scalar_t);
    stencil_conv2d_tiled_single_channel_kernel<scalar_t>
        <<<blocks, threads, shared_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            b.has_value() ? b.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            geometry.in_h,
            geometry.in_w,
            geometry.kernel_h,
            geometry.kernel_w,
            geometry.out_h,
            geometry.out_w,
            padding_h,
            padding_w);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(block_gnn, CUDA, m) {
  m.impl("stencil_conv2d", &stencil_conv2d_cuda);
  m.impl("stencil_conv2d_tiled", &stencil_conv2d_tiled_cuda);
}
