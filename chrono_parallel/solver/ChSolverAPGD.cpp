#include "chrono_parallel/solver/ChSolverAPGD.h"
#include <blaze/math/CompressedVector.h>
using namespace chrono;

#define SHUR(x) (data_container->host_data.D_T * (data_container->host_data.M_invD * x) + data_container->host_data.E * x)

ChSolverAPGD::ChSolverAPGD() : ChSolverParallel(), mg_tmp_norm(0), mb_tmp_norm(0), obj1(0), obj2(0), norm_ms(0), dot_g_temp(0), theta(1), theta_new(0), beta_new(0), t(0), L(0), g_diff(0) {}

void ChSolverAPGD::UpdateR() {

  if (data_container->num_constraints <= 0) {
    return;
  }

  if(!data_container->settings.solver.update_rhs){
    return;
  }

  DynamicVector<real>& M_invk = data_container->host_data.M_invk;
  CompressedMatrix<real>& D_T = data_container->host_data.D_T;
  DynamicVector<real>& b = data_container->host_data.b;
  DynamicVector<real>& v = data_container->host_data.hf;
  DynamicVector<real>& s = data_container->host_data.s;
  DynamicVector<real>& hf = data_container->host_data.hf;

  s.resize(data_container->num_constraints);
  s.reset();
  rigid_rigid->Build_s();


  data_container->host_data.R = -b - D_T * M_invk - s;
}

uint ChSolverAPGD::SolveAPGD(const uint max_iter, const uint size, const blaze::DynamicVector<real>& r, blaze::DynamicVector<real>& gamma) {
  real& residual = data_container->measures.solver.residual;
  real& objective_value = data_container->measures.solver.objective_value;
  custom_vector<real>& iter_hist = data_container->measures.solver.iter_hist;

  blaze::DynamicVector<real> one(size, 1.0);
  data_container->system_timer.start("ChSolverParallel_Solve");

  N_gamma_new.resize(size);
  temp.resize(size);
  g.resize(size);
  gamma_new.resize(size);
  y.resize(size);

  residual = 10e30;
  g_diff = 1.0 / pow(size, 2.0);

  theta = 1;
  theta_new = theta;
  beta_new = 0.0;
  mb_tmp_norm = 0, mg_tmp_norm = 0;
  obj1 = 0.0, obj2 = 0.0;
  dot_g_temp = 0, norm_ms = 0;

  // Is the initial projection necessary?
  // Project(gamma.data());
  // gamma_hat = gamma;
  // ShurProduct(gamma, mg);
  // mg = mg - r;

  temp = gamma - one;
  L = sqrt((real)(temp, temp));
  temp = SHUR(temp);    // ShurProduct(temp, temp);
  L = L == 0 ? 1 : L;
  L = sqrt((real)(temp, temp)) / L;

  t = 1.0 / L;
  y = gamma;

  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {

    // ShurProduct(y, g);
    g = SHUR(y) - r;
    gamma_new = y - t * g;

    Project(gamma_new.data());

    N_gamma_new = SHUR(gamma_new);    // ShurProduct(mx, temp_N_mx);
    obj1 = 0.5 * (gamma_new, N_gamma_new) - (gamma_new, r);

    // ShurProduct(y, temp);
    obj2 = 0.5 * (y, SHUR(y)) - (y, r);

    temp = gamma_new - y;
    dot_g_temp = (g, temp);
    norm_ms = (temp, temp);

    while (obj1 > obj2 + dot_g_temp + 0.5 * L * norm_ms) {
      L = 2.0 * L;
      t = 1.0 / L;
      gamma_new = y - t * g;
      Project(gamma_new.data());
      N_gamma_new = SHUR(gamma_new);    // ShurProduct(mx, temp_N_mx);
      obj1 = 0.5 * (gamma_new, N_gamma_new) - (gamma_new, r);
      temp = gamma_new - y;
      dot_g_temp = (g, temp);
      norm_ms = (temp, temp);
    }
    theta_new = (-pow(theta, 2.0) + theta * sqrt(pow(theta, 2.0) + 4.0)) / 2.0;
    beta_new = theta * (1.0 - theta) / (pow(theta, 2.0) + theta_new);

    temp = gamma_new - gamma;
    y = beta_new * temp + gamma_new;
    dot_g_temp = (g, temp);

    // Compute the residual
    temp = gamma_new - g_diff * (N_gamma_new - r);
    Project(temp.data());
    temp = (1.0 / g_diff) * (gamma_new - temp);
    real res = sqrt((real)(temp, temp));

    if (res < residual) {
      residual = res;
      gamma_hat = gamma_new;
    }

    // Compute the objective value
    temp = 0.5 * N_gamma_new - r;
    objective_value = (gamma_new, temp);

    AtIterationEnd(residual, objective_value);

    if (data_container->settings.solver.test_objective) {
      if (objective_value <= data_container->settings.solver.tolerance_objective) {
        break;
      }
    } else {
      if (residual < data_container->settings.solver.tolerance) {
        break;
      }
    }

    if (dot_g_temp > 0) {
      y = gamma_new;
      theta_new = 1.0;
    }

    L = 0.9 * L;
    t = 1.0 / L;
    theta = theta_new;

    gamma = gamma_new;

    UpdateR();
  }

  gamma = gamma_hat;

  data_container->system_timer.stop("ChSolverParallel_Solve");
  return current_iteration;
}
