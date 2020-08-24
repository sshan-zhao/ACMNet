#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

// input: points(b, npoints, nfeats), index(b, npoints) 
// output: v_points(b, n, nfeats), v_idx(b, n)
__global__ void valid_idx_row_kernel(int b, int h, int w, 
                                    const int *__restrict__ mask,
                                    int *__restrict__ tmp) {
  int batch_index = blockIdx.x;
  mask += batch_index * h * w;
  tmp += batch_index * h * w;

  int th_index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = th_index; i < h; i += stride) {
      int ind = i*w;
      for (int j = 0; j < w; ++j) {
          if (j == 0)
            tmp[ind + j] = mask[ind + j];
          else
            tmp[ind + j] = tmp[ind + j - 1] + mask[ind + j];
      }
    
  }
}

// input: points(b, npoints, nfeats), index(b, npoints) 
// output: v_points(b, n, nfeats), v_idx(b, n)
__global__ void valid_idx_col_kernel(int b, int h, int w, int n, 
                                    const int *__restrict__ mask,
                                    int *__restrict__ tmp1,
                                    int *__restrict__ tmp2,
                                    int *__restrict__ v_idx) {
  int batch_index = blockIdx.x;
  mask += batch_index * h * w;
  tmp1 += batch_index * h * w;
  tmp2 += batch_index * h;
  v_idx += batch_index * n;

  int th_index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = th_index; i < h; i += stride) {
      int ind = i * w;
      int ind2 = tmp2[i];
      for (int j = 0; j < w; ++j) {
          if (mask[ind + j] == 1) {
              v_idx[tmp1[ind + j] + ind2 - 1] = ind + j;
          }
      }
  }
}

__global__ void valid_idx_last_col_kernel(int b, int h, int w,
                                    int *__restrict__ tmp1,
                                    int *__restrict__ tmp2) {
  int batch_index = blockIdx.x;
  tmp1 += batch_index * h * w;
  tmp2 += batch_index * h;

  int stride = blockDim.x;

  for (int i = 0; i < h; ++i) {
     if (i == 0)
        tmp2[i] = 0;
     else
        tmp2[i] = tmp2[i-1] + tmp1[i * w - 1];
  }
}

void valid_idx_kernel_wrapper(int b, int h, int w, int n,
                                 const int *mask, int *tmp1, int *tmp2, int *v_idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  valid_idx_row_kernel<<<b, opt_n_threads(h), 0, stream>>>(
      b, h, w, mask, tmp1);
  valid_idx_last_col_kernel<<<b, opt_n_threads(1), 0, stream>>>(
      b, h, w, tmp1, tmp2);
  valid_idx_col_kernel<<<b, opt_n_threads(h), 0, stream>>>(
      b, h, w, n, mask, tmp1, tmp2, v_idx);

  //CUDA_CHECK_ERRORS();
}
