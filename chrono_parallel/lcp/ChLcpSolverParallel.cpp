#include "chrono_parallel/lcp/ChLcpSolverParallel.h"
#include "chrono_parallel/math/ChThrustLinearAlgebra.h"
#include "core/ChSpmatrix.h"
#include "physics/ChBody.h"
using namespace chrono;

ChLcpSolverParallel::ChLcpSolverParallel(ChParallelDataManager* dc)
: data_container(dc)
{
  tolerance = 1e-7;
  record_violation_history = true;
  warm_start = false;
  residual = 0;
  solver = new ChSolverAPGD();
}

ChLcpSolverParallel::~ChLcpSolverParallel()
{
  delete solver;
}

void ChLcpSolverParallel::ComputeMassMatrix()
{
  uint num_bodies = data_container->num_bodies;
  uint num_shafts = data_container->num_shafts;
  uint num_dof = data_container->num_dof;

  const custom_vector<real>& shaft_inr = data_container->host_data.shaft_inr;

  const std::vector<ChBody*> *body_list = data_container->body_list;
  const std::vector<ChLink*> *link_list= data_container->link_list;
  const std::vector<ChPhysicsItem*> *other_physics_list= data_container->other_physics_list;

  const DenseVector& hf = data_container->host_data.hf;
  const DenseVector& v = data_container->host_data.v;

  DenseVector& M_invk = data_container->host_data.M_invk;
  SparseMatrix& M_inv = data_container->host_data.M_inv;

  M_inv.setZero();

  // Each rigid object has 3 mass entries and 9 inertia entries
  // Each shaft has one inertia entry
  M_inv.reserve(num_bodies * 12 + num_shafts * 1);
  // The mass matrix is square and each rigid body has 6 DOF
  // Shafts have one DOF
  M_inv.resize(num_dof, num_dof);

  for (int i = 0; i < num_bodies; i++) {

    if (data_container->host_data.active_data[i]) {
      real inv_mass = 1.0/body_list->at(i)->GetMass();
      ChMatrix33<>&   body_inv_inr = body_list->at(i)->VariablesBody().GetBodyInvInertia();

      M_inv.insert(i * 6 + 0, i * 6 + 0)= inv_mass;
      M_inv.insert(i * 6 + 1, i * 6 + 1)= inv_mass;
      M_inv.insert(i * 6 + 2, i * 6 + 2)= inv_mass;

      M_inv.insert(i * 6 + 3, i * 6 + 3)= body_inv_inr.GetElement(0, 0);
      M_inv.insert(i * 6 + 3, i * 6 + 4)= body_inv_inr.GetElement(0, 1);
      M_inv.insert(i * 6 + 3, i * 6 + 5)= body_inv_inr.GetElement(0, 2);
      M_inv.insert(i * 6 + 4, i * 6 + 3)= body_inv_inr.GetElement(1, 0);
      M_inv.insert(i * 6 + 4, i * 6 + 4)= body_inv_inr.GetElement(1, 1);
      M_inv.insert(i * 6 + 4, i * 6 + 5)= body_inv_inr.GetElement(1, 2);
      M_inv.insert(i * 6 + 5, i * 6 + 3)= body_inv_inr.GetElement(2, 0);
      M_inv.insert(i * 6 + 5, i * 6 + 4)= body_inv_inr.GetElement(2, 1);
      M_inv.insert(i * 6 + 5, i * 6 + 5)= body_inv_inr.GetElement(2, 2);
    } else {

    }
  }

  for (int i = 0; i < num_shafts; i++) {
    M_inv.insert(num_bodies * 6 + i, num_bodies * 6 + i)= shaft_inr[i];
  }

  //If this experimental feature is requested compute the real mass matrix
  if(data_container->settings.solver.scale_mass_matrix){

    SparseMatrix& M = data_container->host_data.M;
    M.setZero();

      // Each rigid object has 3 mass entries and 9 inertia entries
      // Each shaft has one inertia entry
      M.reserve(num_bodies * 12 + num_shafts * 1);
      // The mass matrix is square and each rigid body has 6 DOF
      // Shafts have one DOF
      M.resize(num_dof, num_dof);

      for (int i = 0; i < num_bodies; i++) {
        if (data_container->host_data.active_data[i]) {
          real mass = body_list->at(i)->GetMass();
          ChMatrix33<>&   body_inr = body_list->at(i)->VariablesBody().GetBodyInertia();

          M.insert(i * 6 + 0, i * 6 + 0)= mass;
          M.insert(i * 6 + 1, i * 6 + 1)= mass;
          M.insert(i * 6 + 2, i * 6 + 2)= mass;

          M.insert(i * 6 + 3, i * 6 + 3)= body_inr.GetElement(0, 0);
          M.insert(i * 6 + 3, i * 6 + 4)= body_inr.GetElement(0, 1);
          M.insert(i * 6 + 3, i * 6 + 5)= body_inr.GetElement(0, 2);
          M.insert(i * 6 + 4, i * 6 + 3)= body_inr.GetElement(1, 0);
          M.insert(i * 6 + 4, i * 6 + 4)= body_inr.GetElement(1, 1);
          M.insert(i * 6 + 4, i * 6 + 5)= body_inr.GetElement(1, 2);
          M.insert(i * 6 + 5, i * 6 + 3)= body_inr.GetElement(2, 0);
          M.insert(i * 6 + 5, i * 6 + 4)= body_inr.GetElement(2, 1);
          M.insert(i * 6 + 5, i * 6 + 5)= body_inr.GetElement(2, 2);
        } else {

        }
      }

      for (int i = 0; i < num_shafts; i++) {
        M.insert(num_bodies * 6 + i, num_bodies * 6 + i)= 1.0/shaft_inr[i];
      }
  }


  M_invk = v + M_inv * hf;
}

void ChLcpSolverParallel::PerformStabilization() {
  const DenseVector& R_full = data_container->host_data.R_full;
  DenseVector& gamma = data_container->host_data.gamma;
  uint num_unilaterals = data_container->num_unilaterals;
  uint num_bilaterals = data_container->num_bilaterals;

  if (data_container->settings.solver.max_iteration_bilateral <= 0 || num_bilaterals <= 0) {
    return;
  }

  Eigen::VectorBlock<const DenseVector> R_b = R_full.segment( num_unilaterals, num_bilaterals);
  Eigen::VectorBlock<DenseVector> gamma_b = gamma.segment( num_unilaterals, num_bilaterals);

  data_container->system_timer.start("ChLcpSolverParallel_Stab");
  //solver->SolveStab(data_container->settings.solver.max_iteration_bilateral, num_bilaterals, R_b, gamma_b);
  data_container->system_timer.stop("ChLcpSolverParallel_Stab");
}
