#include "chrono_parallel/solver/ChSolverPDIP.h"
#include <blaze/math/CompressedVector.h>

#include "core/ChFileutils.h"
#include "core/ChStream.h"

#include <algorithm>

using namespace chrono;

unsigned int offset = 3;

#define _index_ i*offset

real ChSolverPDIP::Res4(DenseVector & gamma,
                        DenseVector & tmp)
{
  real gdiff = 1e-6;
  SchurComplementProduct(gamma, tmp);
  tmp = tmp + r;
  tmp = gamma - gdiff * (tmp);
  Project(tmp.data());
  tmp = (1.0 / gdiff) * (gamma - tmp);

  return sqrt((double) tmp.dot( tmp));
}

void ChSolverPDIP::SchurComplementProduct(DenseVector & src,
                                          DenseVector & dst)
{
  dst = D_T * (M_invD * src);
}

void ChSolverPDIP::getConstraintVector(DenseVector & src,
                                       DenseVector & dst, const uint size)
{
#pragma omp parallel for
  for (int i = 0; i < data_container->num_contacts; i++) {
  dst[i] = 0.5 * (pow(src[_index_ + 1], 2) + pow(src[_index_ + 2], 2) - pow(data_container->host_data.fric_rigid_rigid[i].x, 2) * pow(src[_index_], 2));
  dst[i + data_container->num_contacts] = -src[_index_];
}
}

void ChSolverPDIP::initializeConstraintGradient(DenseVector & src,
                                                const uint size)
{
  //#pragma omp parallel for
  for (int i = 0; i < data_container->num_contacts; i++) {
    grad_f.insert(i, _index_ + 0)= -pow(data_container->host_data.fric_rigid_rigid[i].x, 2) * src[_index_ + 0];
    grad_f.insert(i, _index_ + 1)= src[_index_ + 1];
    grad_f.insert(i, _index_ + 2)= src[_index_ + 2];

  }
  for (int i = 0; i < data_container->num_contacts; i++) {
    grad_f.insert(i + data_container->num_contacts, _index_)= -1;

  }
}

void ChSolverPDIP::updateConstraintGradient(DenseVector & src,
                                            const uint size)
{
#pragma omp parallel for
  for (int i = 0; i < data_container->num_contacts; i++) {
  grad_f.coeffRef(i, _index_ + 0) = -pow(data_container->host_data.fric_rigid_rigid[i].x, 2) * src[_index_ + 0];
  grad_f.coeffRef(i, _index_ + 1) = src[_index_ + 1];
  grad_f.coeffRef(i, _index_ + 2) = src[_index_ + 2];
}
#pragma omp parallel for
  for (int i = 0; i < data_container->num_contacts; i++) {
    grad_f.coeffRef(i + data_container->num_contacts, _index_) = -1;
}
}

void ChSolverPDIP::initializeNewtonStepMatrix(DenseVector & gamma,
                                              DenseVector & lambda,
                                              DenseVector & f,
                                              const uint size)
{
  for (int i = 0; i < data_container->num_contacts; i++) {
    M_hat.insert(_index_ + 0, _index_ + 0)= -pow(data_container->host_data.fric_rigid_rigid[i].x, 2) * lambda[i];
    M_hat.insert(_index_ + 1, _index_ + 1)= lambda[i];
    M_hat.insert(_index_ + 2, _index_ + 2)= lambda[i];
  }
  updateConstraintGradient(gamma, size);
  B = grad_f.transpose();

  for (int i = 0; i < 2 * data_container->num_contacts; i++) {
    diaglambda.insert(i, i)= lambda[i];
    Dinv.insert(i, i)= -1 / f[i];

  }
}

void ChSolverPDIP::updateNewtonStepMatrix(DenseVector & gamma,
                                          DenseVector & lambda,
                                          DenseVector & f,
                                          const uint size)
{
#pragma omp parallel for
  for (int i = 0; i < data_container->num_contacts; i++) {
    M_hat.coeffRef(_index_ + 0, _index_ + 0) = -pow(data_container->host_data.fric_rigid_rigid[i].x, 2) * lambda[i];
    M_hat.coeffRef(_index_ + 1, _index_ + 1) = lambda[i];
    M_hat.coeffRef(_index_ + 2, _index_ + 2) = lambda[i];
  }
  updateConstraintGradient(gamma, size);
  B = grad_f.transpose();

#pragma omp parallel for
  for (int i = 0; i < 2 * data_container->num_contacts; i++) {
    diaglambda.coeffRef(i, i) = lambda[i];

    Dinv.coeffRef(i, i) = -1 / f[i];
  }
}

void ChSolverPDIP::MultiplyByDiagMatrix(DenseVector & diagVec,
                                        DenseVector & src,
                                        DenseVector & dst)
{
#pragma omp parallel for
  for (int i = 0; i < 2 * data_container->num_contacts; i++) {
    dst[i] = diagVec[i] * src[i];
  }
}

void ChSolverPDIP::updateNewtonStepVector(DenseVector & gamma,
                                          DenseVector & lambda,
                                          DenseVector & f,
                                          real t,
                                          const uint size)
{
  updateConstraintGradient(gamma, size);
  r_d = D_T * (M_invD * gamma) + r + grad_f.transpose() * lambda;
  MultiplyByDiagMatrix(lambda, f, r_g);
  r_g = - (1 / t) * ones - r_g;
}

void ChSolverPDIP::conjugateGradient(DenseVector & x)
{
  real rsold_cg = 0;
  real rsnew_cg = 0;
  real alpha_cg = 0;
  r_cg = (B * (Dinv * r_g) - r_d) - (D_T * M_invD + M_hat + B * Dinv * diaglambda * grad_f) * x;
  p_cg = r_cg;
  rsold_cg = r_cg.dot( r_cg);

  for (int i = 0; i < 10*gamma.size(); i++) {
    Ap_cg = (D_T * M_invD + M_hat + B * Dinv * diaglambda * grad_f) * p_cg;
    alpha_cg = rsold_cg / p_cg.dot( Ap_cg);
    x = x + alpha_cg * p_cg;
    r_cg = r_cg - alpha_cg * Ap_cg;
    rsnew_cg = r_cg.dot( r_cg);
    if (sqrt(rsnew_cg) < data_container->settings.solver.tol_speed / 100.0) {
      return;
    }
    p_cg = r_cg + rsnew_cg / rsold_cg * p_cg;
    rsold_cg = rsnew_cg;
  }
}

void ChSolverPDIP::buildPreconditioner(const uint size)
{
  prec_cg.resize(size);
  SparseMatrix A = D_T * M_invD + M_hat + B * Dinv * diaglambda * grad_f;
#pragma omp parallel for
  for(int i=0; i<prec_cg.size(); i++)
  {
    prec_cg[i] = A.coeff(i,i);
  }
}

void ChSolverPDIP::applyPreconditioning(DenseVector & src,
                                        DenseVector & dst)
{
#pragma omp parallel for
  for(int i=0; i<prec_cg.size(); i++)
  {
    dst[i] = src[i]/prec_cg[i];
  }
}

int ChSolverPDIP::preconditionedConjugateGradient(DenseVector & x,
                                                  const uint size)
{
  buildPreconditioner(size);
  int iter = 0;
  real rsold_cg = 0;
  real rsnew_cg = 0;
  real alpha_cg = 0;
  r_cg = (B * (Dinv * r_g) - r_d) - (D_T * M_invD + M_hat + B * Dinv * diaglambda * grad_f) * x;
  applyPreconditioning(r_cg,z_cg);
  p_cg = z_cg;
  rsold_cg = r_cg.dot( z_cg);

  for (int i = 0; i < 1000*gamma.size(); i++) {
    iter++;
    Ap_cg = (D_T * M_invD + M_hat + B * Dinv * diaglambda * grad_f) * p_cg;
    alpha_cg = rsold_cg / p_cg.dot( Ap_cg);
    x = x + alpha_cg * p_cg;
    r_cg = r_cg - alpha_cg * Ap_cg;
    applyPreconditioning(r_cg,z_cg);
    rsnew_cg = z_cg.dot( r_cg);
    if (sqrt(rsnew_cg) < data_container->settings.solver.tol_speed / 100.0) {
      return iter;
    }
    p_cg = z_cg + rsnew_cg / rsold_cg * p_cg;
    rsold_cg = rsnew_cg;
  }

  return iter;
}

uint ChSolverPDIP::SolvePDIP(const uint max_iter,
                             const uint size,
                             const DenseVector& b,
                             DenseVector& x)
{
  bool verbose = false;
  if(verbose) std::cout << "Number of constraints: " << size << "\nNumber of variables  : " << data_container->num_bodies << std::endl;

  real& residual = data_container->measures.solver.residual;
  real& objective_value = data_container->measures.solver.objective_value;
  custom_vector<real>& iter_hist = data_container->measures.solver.iter_hist;

  data_container->system_timer.start("ChSolverParallel_solverA");
  int totalKrylovIterations = 0;

  uint num_contacts = data_container->num_contacts;

  // Initialize scalars
  real mu = 10;
  real alpha = 0.001;  // should be [0.01, 0.1]
  real beta = 0.8;  // should be [0.3, 0.8]
  real eta_hat = 0;
  real obj = 0;
  real newobj = 0;
  real t = 0;
  real s = 1;
  real s_max = 1;
  real norm_rt = 0;

  // Initialize vectors
  gamma.resize(size);
  gamma_tmp.resize(size);
  r.resize(size);
  f.resize(2 * num_contacts);
  lambda.resize(2 * num_contacts);
  lambda_tmp.resize(2 * num_contacts);
  ones.resize(2 * num_contacts);
  r_d.resize(size);
  r_g.resize(2 * num_contacts);
  delta_gamma.resize(size);
  delta_lambda.resize(size);

  r_cg.resize(size);
  p_cg.resize(size);
  z_cg.resize(size);
  Ap_cg.resize(size);

  // Initialize matrices
  grad_f.setZero();
  grad_f.resize(2 * num_contacts, size);
  grad_f.reserve(4 * num_contacts);  // there are (4*num_contacts) nonzero entries

  M_hat.setZero();
  M_hat.resize(size, size);
  M_hat.reserve(size);  // there are (size) nonzero entries

  B.setZero();
  B.resize(size, 2 * num_contacts);
  B.reserve(4 * num_contacts);  // there are (4*num_contacts) nonzero entries

  diaglambda.setZero();
  diaglambda.resize(2 * num_contacts, 2 * num_contacts);
  diaglambda.reserve(2 * num_contacts);  // there are (2 * num_contacts) nonzero entries

  Dinv.setZero();
  Dinv.resize(2 * num_contacts, 2 * num_contacts);
  Dinv.reserve(2 * num_contacts);  // there are (2 * num_contacts) nonzero entries

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    delta_gamma[i] = 0.0;
    gamma[i] = 0.0;
    r[i] = -b[i];  // convert r to a blaze vector
  }

#pragma omp parallel for
  for (int i = 0; i < num_contacts; i++) {
    gamma[3*i] = 1.0;  // provide an initial guess!
  }

  // (1) f = f(gamma_0)
  getConstraintVector(gamma, f, size);

  // (2) lambda_0 = -1/f
#pragma omp parallel for
  for (int i = 0; i < f.size(); i++) {
    ones[i] = 1.0;
    lambda[i] = -1 / f[i];
  }
  initializeConstraintGradient(gamma, size);
  initializeNewtonStepMatrix(gamma, lambda, f, size);

  // (3) for k := 0 to N_max
  for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {

    // (4) f = f(gamma_k)
    getConstraintVector(gamma, f, size);

    // (5) eta_hat = -f^T * lambda_k
    eta_hat = -f.dot( lambda);

    // (6) t = mu*m/eta_hat
    t = mu * (f.size()) / eta_hat;

    // (7) A = A(gamma_k, lambda_k, f)
    updateConstraintGradient(gamma, size);
    updateNewtonStepMatrix(gamma, lambda, f, size);

    // (8) r_t = r_t(gamma_k, lambda_k, t)
    updateNewtonStepVector(gamma, lambda, f, t, size);

    // (9) Solve the linear system A * y = -r_t
    int krylovIter = preconditionedConjugateGradient(delta_gamma,size);
    totalKrylovIterations += krylovIter;
    delta_lambda = Dinv * (diaglambda * (grad_f * delta_gamma) - r_g);

    // (10) s_max = sup{s in [0,1]|lambda+s*delta_lambda>=0} = min{1,min{-lambda_i/delta_lambda_i|delta_lambda_i < 0 }}
#pragma omp parallel for
    for (int i = 0; i < lambda.size(); i++) {
      lambda_tmp[i] = -lambda[i] / delta_lambda[i];
      if (delta_lambda[i] >= 0)
        lambda_tmp[i] = 1.0;
    }
    s_max = std::fmin(1.0, lambda_tmp.minCoeff());

    // (11) s = 0.99 * s_max
    s = 0.99 * s_max;

    // (12) while max(f(gamma_k + s * delta_gamma) > 0)
    gamma_tmp = gamma + s * delta_gamma;
    getConstraintVector(gamma_tmp, lambda_tmp, size);
    while (lambda_tmp.maxCoeff() > 0) {
      // (13) s = beta * s
      s = beta * s;
      gamma_tmp = gamma + s * delta_gamma;
      getConstraintVector(gamma_tmp, lambda_tmp, size);

      // (14) endwhile
    }

    // (15) while norm(r_t(gamma_k + s * delta_gamma, lambda_k + s * delta_lambda),2) > (1-alpha*s)*norm(r_t,2)
    norm_rt = sqrt(r_d.dot( r_d) + r_g.dot( r_g));
    lambda_tmp = lambda + s * delta_lambda;
    getConstraintVector(gamma_tmp, f, size);
    updateNewtonStepVector(gamma_tmp, lambda_tmp, f, t, size);
    while (sqrt(r_d.dot( r_d) + r_g.dot( r_g)) > (1 - alpha * s) * norm_rt) {
      // (16) s = beta * s
      s = beta * s;
      gamma_tmp = gamma + s * delta_gamma;
      lambda_tmp = lambda + s * delta_lambda;
      getConstraintVector(gamma_tmp, f, size);
      updateNewtonStepVector(gamma_tmp, lambda_tmp, f, t, size);

      // (17) endwhile
    }

    // (18) gamma_(k+1) = gamma_k + s * delta_gamma
    gamma = gamma + s * delta_gamma;

    // (19) lambda_(k+1) = lamda_k + s * delta_lambda
    lambda = lambda + s * delta_lambda;

    // (20) r = r(gamma_(k+1))
    residual = sqrt(r_g.dot( r_g));//Res4(gamma, gamma_tmp);

    // (21) if r < tau
    AtIterationEnd(residual, objective_value);
    if(verbose) std::cout << "Residual: " << residual << ", Iter: " << current_iteration << ", Krylov Iter: " << totalKrylovIterations << std::endl;

    if (residual < data_container->settings.solver.tol_speed) {
      // (22) break
      break;

      // (23) endif
    }

    // (24) endfor
  }

  // (25) return Value at time step t_(l+1), gamma_(l+1) := gamma_(k+1)
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    x[i] = gamma[i];
  }

  data_container->system_timer.stop("ChSolverParallel_solverA");

  return current_iteration;
}
