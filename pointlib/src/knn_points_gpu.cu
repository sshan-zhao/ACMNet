#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

// input: points(b, npoints, 3), nsample 
// output: idx(b, npoints, nsample)
__global__ void knn_points_kernel(int b, int npointsa, int npointsb, int knn,
                                    const float *__restrict__ pointsa,
                                    const float *__restrict__ pointsb,
                                    int *__restrict__ nnidx) {
  int batch_index = blockIdx.x;
  pointsa += batch_index * npointsa * 3;
  pointsb += batch_index * npointsb * 3;
  nnidx += batch_index * npointsa * knn;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < npointsa; i += stride) {

    float x = pointsa[i * 3 + 0];
    float y = pointsa[i * 3 + 1];
    float z = pointsa[i * 3 + 2];
    int ind = i * knn;
    float *dists = new float[knn];
    for (int j = 0; j < npointsb; ++j) {
      float x0 = pointsb[j * 3 + 0];
      float y0 = pointsb[j * 3 + 1];
      float z0 = pointsb[j * 3 + 2];
      float dist = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
      if (j < knn) {
        dists[j] = dist;
        nnidx[ind + j] = j;
      }
      else {
        float dist_ = -1.0;
        int k_ = -1;
        for (int k = 0; k < knn; ++k) {
          if (dists[k] > dist_) {
            dist_ = dists[k];
            k_ = k;
          }
        }
        if (dist < dist_) {
          dists[k_] = dist;
          nnidx[ind + k_] = j;
        }
      }
    }
    delete[] dists;
  }
}

void knn_points_kernel_wrapper(int b, int npointsa, int npointsb, int knn,
                                 const float *pointsa, const float *pointsb, int *nnidx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  knn_points_kernel<<<b, opt_n_threads(npointsa), 0, stream>>>(
      b, npointsa, npointsb, knn, pointsa, pointsb, nnidx);

  //CUDA_CHECK_ERRORS();
}
