#include "chrono_parallel/physics/ChSystemParallel.h"
#include <omp.h>

using namespace chrono;

ChSystemParallelDVI::ChSystemParallelDVI(unsigned int max_objects)
      : ChSystemParallel(max_objects)
{
   LCP_solver_speed = new ChLcpSolverParallelDVI(data_manager);

   //Set this so that the CD can check what type of system it is (needed for narrowphase)
   data_manager->settings.system_type = SYSTEM_DVI;

   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurA_normal");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurA_sliding");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurA_spinning");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurA_reduce");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurB_normal");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurB_sliding");
   data_manager->system_timer.AddTimer("ChConstraintRigidRigid_shurB_spinning");

   data_manager->system_timer.AddTimer("ChSolverParallel_solverA");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverB");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverC");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverD");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverE");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverF");
   data_manager->system_timer.AddTimer("ChSolverParallel_solverG");
   data_manager->system_timer.AddTimer("ChSolverParallel_Project");
   data_manager->system_timer.AddTimer("ChSolverParallel_Solve");
}

void ChSystemParallelDVI::AddMaterialSurfaceData(ChSharedPtr<ChBody> newbody) {
   assert(typeid(*newbody.get_ptr()) == typeid(ChBody) || typeid(*newbody.get_ptr()) == typeid(ChBodyAuxRef));

   // Reserve space for material properties for the specified body. Not that the
   // actual data is set in UpdateMaterialProperties().
   data_manager->host_data.fric_data.push_back(R3(0));
   data_manager->host_data.cohesion_data.push_back(0);
   data_manager->host_data.compliance_data.push_back(R4(0));
}

void ChSystemParallelDVI::UpdateMaterialSurfaceData(int index, ChBody* body)
{
  custom_vector<real>& cohesion = data_manager->host_data.cohesion_data;
  custom_vector<real3>& friction = data_manager->host_data.fric_data;
  custom_vector<real4>& compliance = data_manager->host_data.compliance_data;

  ChSharedPtr<ChMaterialSurface>& mat = body->GetMaterialSurface();

  friction[index] = R3(mat->GetKfriction(), mat->GetRollingFriction(), mat->GetSpinningFriction());
  cohesion[index] = mat->GetCohesion();
  compliance[index] = R4(mat->GetCompliance(), mat->GetComplianceT(), mat->GetComplianceRolling(), mat->GetComplianceSpinning());
}

void ChSystemParallelDVI::CalculateContactForces()
{
  uint num_contacts = data_manager->num_contacts;
  DenseVector& Fc = data_manager->host_data.Fc;

  data_manager->Fc_current = true;

  if (num_contacts == 0) {
    Fc.resize(6 * data_manager->num_bodies);
    Fc = 0;
    return;
  }

  DenseVector& gamma = data_manager->host_data.gamma;

  switch (data_manager->settings.solver.solver_mode) {
  case NORMAL: {
    const SparseMatrix& D_n = data_manager->host_data.D_n;

    Fc = D_n * gamma.segment(0, num_contacts);
  } break;
  case SLIDING: {
    const SparseMatrix& D_n = data_manager->host_data.D_n;
    const SparseMatrix& D_t = data_manager->host_data.D_t;
    Fc = D_n * gamma.segment(0, num_contacts) + D_t * gamma.segment(num_contacts, 2 * num_contacts);
  } break;
  case SPINNING: {
    const SparseMatrix& D_n = data_manager->host_data.D_n;
    const SparseMatrix& D_t = data_manager->host_data.D_t;
    const SparseMatrix& D_s = data_manager->host_data.D_s;
    Fc = D_n * gamma.segment(0, num_contacts) + D_t * gamma.segment(num_contacts, 2 * num_contacts) + D_s * gamma.segment(3 * num_contacts, 3 * num_contacts);
  } break;
  }

  Fc = Fc / data_manager->settings.step_size;
}

real3 ChSystemParallelDVI::GetBodyContactForce(uint body_id) const
{
  assert(data_manager->Fc_current);
  return R3(data_manager->host_data.Fc[body_id * 6 + 0],
            data_manager->host_data.Fc[body_id * 6 + 1],
            data_manager->host_data.Fc[body_id * 6 + 2]);
}

real3 ChSystemParallelDVI::GetBodyContactTorque(uint body_id) const
{
  assert(data_manager->Fc_current);
  return R3(data_manager->host_data.Fc[body_id * 6 + 3],
            data_manager->host_data.Fc[body_id * 6 + 4],
            data_manager->host_data.Fc[body_id * 6 + 5]);
}

static inline chrono::ChVector<real> ToChVector(const real3 &a) {
   return chrono::ChVector<real>(a.x, a.y, a.z);
}

void ChSystemParallelDVI::SolveSystem() {
   data_manager->system_timer.Reset();
   data_manager->system_timer.start("step");
   data_manager->system_timer.start("update");
   Update();
   data_manager->system_timer.stop("update");
   Prepare();
   data_manager->system_timer.start("collision");
   collision_system->Run();
   collision_system->ReportContacts(this->contact_container);
   data_manager->system_timer.stop("collision");
   data_manager->system_timer.start("lcp");
   ((ChLcpSolverParallel *) (LCP_solver_speed))->RunTimeStep();
   data_manager->system_timer.stop("lcp");
   data_manager->system_timer.stop("step");
   timer_update = data_manager->system_timer.GetTime("update");
   timer_collision = data_manager->system_timer.GetTime("collision");
   timer_lcp = data_manager->system_timer.GetTime("lcp");
   timer_step = data_manager->system_timer.GetTime("step");
}
void ChSystemParallelDVI::AssembleSystem() {
   Setup();

   collision_system->Run();
   collision_system->ReportContacts(this->contact_container);
   ChSystem::Update();
   this->contact_container->BeginAddContact();
   chrono::collision::ChCollisionInfo icontact;
   for (int i = 0; i < data_manager->num_contacts; i++) {
      int2 cd_pair = data_manager->host_data.bids_rigid_rigid[i];
      icontact.modelA = bodylist[cd_pair.x]->GetCollisionModel();
      icontact.modelB = bodylist[cd_pair.y]->GetCollisionModel();
      icontact.vN = ToChVector(data_manager->host_data.norm_rigid_rigid[i]);
      icontact.vpA = ToChVector(data_manager->host_data.cpta_rigid_rigid[i] + data_manager->host_data.pos_data[cd_pair.x]);
      icontact.vpB = ToChVector(data_manager->host_data.cptb_rigid_rigid[i] + data_manager->host_data.pos_data[cd_pair.y]);
      icontact.distance = data_manager->host_data.dpth_rigid_rigid[i];
      this->contact_container->AddContact(icontact);
   }
   this->contact_container->EndAddContact();

   {
      std::vector<ChLink*>::iterator iterlink = linklist.begin();
      while (iterlink != linklist.end()) {
         (*iterlink)->ConstraintsBiReset();
         iterlink++;
      }
      std::vector<ChBody*>::iterator ibody = bodylist.begin();
      while (ibody != bodylist.end()) {
         (*ibody)->VariablesFbReset();
         ibody++;
      }
      this->contact_container->ConstraintsBiReset();
   }

   LCPprepare_load(true,                  // Cq,
         true,                            // adds [M]*v_old to the known vector
         step,                            // f*dt
         step * step,                     // dt^2*K  (nb only non-Schur based solvers support K matrix blocks)
         step,                            // dt*R   (nb only non-Schur based solvers support R matrix blocks)
         1.0,                             // M (for FEM with non-lumped masses, add their mass-matrixes)
         1.0,                             // Ct   (needed, for rheonomic motors)
         1.0 / step,                      // C/dt
         max_penetration_recovery_speed,  // vlim, max penetrations recovery speed (positive for exiting)
         true                             // do above max. clamping on -C/dt
         );

   this->LCP_descriptor->BeginInsertion();
   for (int i = 0; i < bodylist.size(); i++) {
      bodylist[i]->InjectVariables(*this->LCP_descriptor);
   }
   std::vector<ChLink *>::iterator it;
   for (it = linklist.begin(); it != linklist.end(); it++) {
      (*it)->InjectConstraints(*this->LCP_descriptor);
   }
   this->contact_container->InjectConstraints(*this->LCP_descriptor);
   this->LCP_descriptor->EndInsertion();

}

