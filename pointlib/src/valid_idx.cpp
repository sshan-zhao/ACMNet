#include "valid_idx.h"
#include "utils.h"

void valid_idx_kernel_wrapper(int b, int h, int w, int n, 
                                 const int *mask, int *tmp1, int *tmp2, int *v_idx);

// mask: b*h*w
// v_idx: b*n
at::Tensor valid_idx(at::Tensor mask, const int n) {
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_INT(mask);

  at::Tensor v_idx = 
      torch::zeros({mask.size(0), n},
                  at::device(mask.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp1 = 
      torch::zeros({mask.size(0), mask.size(1), mask.size(2)},
                  at::device(mask.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp2 = 
      torch::zeros({mask.size(0), mask.size(1)},
                  at::device(mask.device()).dtype(at::ScalarType::Int));

  if (mask.type().is_cuda()) {
    valid_idx_kernel_wrapper(mask.size(0), mask.size(1), mask.size(2), n,
                              mask.data<int>(), tmp1.data<int>(), tmp2.data<int>(), v_idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return v_idx;
}
