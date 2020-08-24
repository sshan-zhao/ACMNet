#pragma once
#include <torch/extension.h>
#include <vector>
at::Tensor valid_idx(at::Tensor mask, const int n);

