#include "knn_points.h"
#include "utils.h"

void knn_points_kernel_wrapper(int b, int npointsa, int npointsb, int knn,
                                 const float *pointsa, const float *pointsb, int *nnidx);

// pointsa: b*Na*3
// pointsb: b*Nb*3
// nnidx: b*Na*knn
at::Tensor knn_points(at::Tensor pointsa, at::Tensor pointsb, const int knn) {
  CHECK_CONTIGUOUS(pointsa);
  CHECK_CONTIGUOUS(pointsb);

  if (pointsa.type().is_cuda())
    CHECK_CUDA(pointsb);

  at::Tensor nnidx = 
      torch::zeros({pointsa.size(0), pointsa.size(1), knn},
                  at::device(pointsa.device()).dtype(at::ScalarType::Int));

  if (pointsa.type().is_cuda()) {
    knn_points_kernel_wrapper(pointsa.size(0), pointsa.size(1), pointsb.size(1), knn, 
                              pointsa.data<float>(), pointsb.data<float>(), nnidx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return nnidx;
}

