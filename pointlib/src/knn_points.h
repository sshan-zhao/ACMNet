#pragma once
#include <torch/extension.h>
at::Tensor knn_points(at::Tensor pointsa, at::Tensor pointsb, const int knn);

