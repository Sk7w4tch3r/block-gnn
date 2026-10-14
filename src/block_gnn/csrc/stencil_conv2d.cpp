#include <ATen/Parallel.h>
#include <torch/extension.h>

namespace {

// Deliberately small reference kernel for the original repeated-stencil idea.
// It is useful for studying dispatcher integration and direct execution, not as
// a replacement for oneDNN. The benchmark suite decides when a backend wins.
torch::Tensor stencil_conv2d_cpu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t padding_h,
    int64_t padding_w) {
  TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
  TORCH_CHECK(input.dim() == 4, "input must be NCHW");
  TORCH_CHECK(weight.dim() == 4, "weight must be OIHW");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input/weight dtype mismatch");
  TORCH_CHECK(input.size(1) == weight.size(1), "input/weight channel mismatch");
  TORCH_CHECK(input.is_floating_point(), "only floating-point inputs are supported");

  auto x = input.contiguous();
  auto w = weight.contiguous();
  c10::optional<torch::Tensor> b = c10::nullopt;
  if (bias.has_value()) {
    b = bias.value().contiguous();
    TORCH_CHECK(b.value().device().is_cpu(), "bias must be on CPU");
    TORCH_CHECK(b.value().scalar_type() == input.scalar_type(), "bias dtype mismatch");
    TORCH_CHECK(b.value().numel() == weight.size(0), "bias size mismatch");
  }

  const auto batch = x.size(0);
  const auto in_channels = x.size(1);
  const auto in_h = x.size(2);
  const auto in_w = x.size(3);
  const auto out_channels = w.size(0);
  const auto kernel_h = w.size(2);
  const auto kernel_w = w.size(3);
  const auto out_h = in_h + 2 * padding_h - kernel_h + 1;
  const auto out_w = in_w + 2 * padding_w - kernel_w + 1;
  TORCH_CHECK(out_h > 0 && out_w > 0, "invalid output shape");

  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "stencil_conv2d_cpu", [&] {
    const auto xa = x.accessor<scalar_t, 4>();
    const auto wa = w.accessor<scalar_t, 4>();
    auto ya = output.accessor<scalar_t, 4>();
    const scalar_t* bias_ptr = b.has_value() ? b.value().data_ptr<scalar_t>() : nullptr;
    const int64_t tasks = batch * out_channels * out_h;

    at::parallel_for(0, tasks, 0, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t out_y = task % out_h;
        const int64_t outer = task / out_h;
        const int64_t out_channel = outer % out_channels;
        const int64_t sample = outer / out_channels;

        for (int64_t out_x = 0; out_x < out_w; ++out_x) {
          scalar_t sum = bias_ptr == nullptr ? scalar_t(0) : bias_ptr[out_channel];
          for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
              const int64_t in_y = out_y - padding_h + kernel_y;
              if (in_y < 0 || in_y >= in_h) {
                continue;
              }
              for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                const int64_t in_x = out_x - padding_w + kernel_x;
                if (in_x >= 0 && in_x < in_w) {
                  sum += xa[sample][in_channel][in_y][in_x] *
                      wa[out_channel][in_channel][kernel_y][kernel_x];
                }
              }
            }
          }
          ya[sample][out_channel][out_y][out_x] = sum;
        }
      }
    });
  });
  return output;
}

}  // namespace

TORCH_LIBRARY(block_gnn, m) {
  m.def(
      "stencil_conv2d(Tensor input, Tensor weight, Tensor? bias, "
      "int padding_h, int padding_w) -> Tensor");
}

TORCH_LIBRARY_IMPL(block_gnn, CPU, m) {
  m.impl("stencil_conv2d", &stencil_conv2d_cpu);
}
