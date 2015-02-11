#include "chrono_parallel/solver/ChSolverGD.h"

using namespace chrono;

uint ChSolverGD::SolveGD(const uint max_iter, const uint size, DenseVector& mb, DenseVector& ml) {
  real& residual = data_container->measures.solver.residual;
  real& objective_value = data_container->measures.solver.objective_value;
  custom_vector<real>& iter_hist = data_container->measures.solver.iter_hist;

  real eps = data_container->settings.step_size;
  r.resize(size);

  ShurProduct(ml, r);    // r = data_container->host_data.D_T *
                         // (data_container->host_data.M_invD * ml);
  r = mb - r;
  real resold = 1, resnew, normb = sqrt((mb, mb));
  if (normb == 0.0) {
    normb = 1;
  };
  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
    ml = ml + eps * r;
    ShurProduct(ml, r);
    r = mb - r;
    resnew = sqrt((ml, ml));
    residual = std::abs(resnew - resold);
    objective_value = GetObjective(ml, mb);
    AtIterationEnd(residual, objective_value);
    if (residual < data_container->settings.solver.tolerance) {
      break;
    }
    resold = resnew;
  }
  Project(ml.data());

  return current_iteration;
}
