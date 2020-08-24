#include "knn_points.h"
#include "sampling.h"
#include "group_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_points", &knn_points);
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);
}
