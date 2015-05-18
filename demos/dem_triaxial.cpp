// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Jonathan Fleischmann, Radu Serban
// =============================================================================
//
// Soft-sphere (DEM) or hard-sphere (DVI) triaxial test validation code.
//
// The global reference frame has Z up.
// All units SI.
//
// =============================================================================

#include <iostream>
#include <vector>
#include <valarray>
#include <string>
#include <sstream>
#include <cmath>

#include "core/ChFileutils.h"
#include "core/ChStream.h"

#include "chrono_utils/ChUtilsCreators.h"
#include "chrono_utils/ChUtilsGenerators.h"
#include "chrono_utils/ChUtilsInputOutput.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/lcp/ChLcpSystemDescriptorParallel.h"

// Control use of OpenGL run-time rendering
//#undef CHRONO_PARALLEL_HAS_OPENGL

#ifdef CHRONO_PARALLEL_HAS_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

#include "demo_utils.h"

using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::flush;
using std::endl;

// -----------------------------------------------------------------------------
// Problem definitions
// -----------------------------------------------------------------------------

// Comment the following line to use DVI contact
#define USE_DEM

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 20;

// Perform dynamic tuning of number of threads?
bool thread_tuning = false;

// Solver settings
#ifdef USE_DEM
double time_step = 1e-5;
double tolerance = 0.01;
int max_iteration_bilateral = 100;
#else
double time_step = 1e-4;
double tolerance = 0.1;
int max_iteration_normal = 0;
int max_iteration_sliding = 10000;
int max_iteration_spinning = 0;
int max_iteration_bilateral = 100;
double contact_recovery_speed = 10e30;
#endif

bool clamp_bilaterals = false;
double bilateral_clamp_speed = 0.1;

// Simulation parameters
#ifdef USE_DEM
double settling_time = 1.0;
double begin_test_time = 2.0;
double end_simulation_time = 12.0;
double axial_speed = 0.001;  // m/s
#else
double settling_time = 0.5;
double begin_test_time = 1.0;
double end_simulation_time = 3.0;
double axial_speed = 0.005;  // m/s
#endif

// Confining stress sigma_b (Pa)
//double sigma_b = 24.2e3;
//double sigma_b = 12.5e3;
//double sigma_b = 6.4e3;
double sigma_b = 3.1e3;

// Confining stress sigma_c (Pa)
//double sigma_c = 24.2e3;
//double sigma_c = 12.5e3;
//double sigma_c = 6.4e3;
double sigma_c = 3.1e3;

bool plain_strain = false;

// Output
#ifdef USE_DEM
const std::string out_dir = "../TRIAXIAL_DEM";
#else
const std::string out_dir = "../TRIAXIAL_DVI";
#endif

const std::string pov_dir = out_dir + "/POVRAY";
const std::string stress_file = out_dir + "/triaxial_stress.dat";
const std::string force_file = out_dir + "/triaxial_force.dat";
const std::string stats_file = out_dir + "/stats.dat";

bool write_povray_data = true;

double data_out_step = 1e-2;       // time interval between data outputs
double visual_out_step = 1e-1;     // time interval between PovRay outputs


// -----------------------------------------------------------------------------
// Utility for adding (visible or invisible) walls
// -----------------------------------------------------------------------------
void AddWall(ChSharedPtr<ChBody>& body, const ChVector<>& dim, const ChVector<>& loc, bool visible) {
  body->GetCollisionModel()->AddBox(dim.x, dim.y, dim.z, loc);

  if (visible == true) {
    ChSharedPtr<ChBoxShape> box(new ChBoxShape);
    box->GetBoxGeometry().Size = dim;
    box->Pos = loc;
    box->SetColor(ChColor(1, 0, 0));
    box->SetFading(0.6f);
    body->AddAsset(box);
  }
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // Create output directories

  if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
    cout << "Error creating directory " << out_dir << endl;
    return 1;
  }
  if (ChFileutils::MakeDirectory(pov_dir.c_str()) < 0) {
    cout << "Error creating directory " << pov_dir << endl;
    return 1;
  }

  // Parameters for the system

  double gravity = 9.81;  // m/s^2

  // Parameters for the balls

  int ballId = 1;  // first ball id

  const int a = 100;
  const int b = 10;
  const int c = 10;
  int numballs = a * b * c;  // number of falling balls = (a X b X c)

  bool dense = false;

  double radius = 0.003;  // m
  double density = 2550;  // kg/m^3
  double mass = density * (4.0 / 3) * CH_C_PI * radius * radius * radius;
  float Y = 4.0e7;  // Pa
  float nu = 0.22f;
  float COR = 0.87f;
  float mu = 0.18f;

  // Parameters for lower plate (fixed), upper plate (prescribed velocity in -z),
  // and side walls (wall 1 and wall 2 opposite in x, wall 3 and 4 opposite in y)

  float mu_ext = 0.13f;

  int groundId = 0;
  int wall_1Id = -1;
  int wall_2Id = -2;
  int wall_3Id = -3;
  int wall_4Id = -4;
  int plateId = -5;
  double width = 0.12;
  double length = 0.12;
  double height = 0.12;
  double thickness = 0.01;

  double height_0;
  double Lx, Ly, Lz;
  double Lx0, Ly0, Lz0;
  ChVector<> pos(0, 0, 0);
  ChQuaternion<> rot(1, 0, 0, 0);
  ChVector<> vel(0, 0, 0);
  real3 force0(0, 0, 0);
  real3 force1(0, 0, 0);
  real3 force2(0, 0, 0);
  real3 force3(0, 0, 0);
  real3 force4(0, 0, 0);
  real3 force5(0, 0, 0);

  // Define two quaternions representing:
  // - a rotation of -90 degrees around x (z2y)
  // - a rotation of +90 degrees around y (z2x)

  ChQuaternion<> z2y;
  ChQuaternion<> z2x;
  z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));
  z2x.Q_from_AngAxis(CH_C_PI / 2, ChVector<>(0, 1, 0));

  // Create the system

#ifdef USE_DEM
  cout << "Create DEM system" << endl;
  const std::string title = "soft-sphere (DEM) triaxial test";
  ChBody::ContactMethod contact_method = ChBody::DEM;
  ChSystemParallelDEM* my_system = new ChSystemParallelDEM();
#else
  cout << "Create DVI system" << endl;
  const std::string title = "hard-sphere (DVI) triaxial test";
  ChBody::ContactMethod contact_method = ChBody::DVI;
  ChSystemParallelDVI* my_system = new ChSystemParallelDVI();
#endif

  my_system->Set_G_acc(ChVector<>(0, 0, -gravity));

  // Set number of threads

  int max_threads = my_system->GetParallelThreadNumber();
  if (threads > max_threads) threads = max_threads;
  my_system->SetParallelThreadNumber(threads);
  omp_set_num_threads(threads);

  my_system->GetSettings()->max_threads = threads;
  my_system->GetSettings()->perform_thread_tuning = thread_tuning;

  // Edit system settings

  my_system->GetSettings()->solver.use_full_inertia_tensor = false;
  my_system->GetSettings()->solver.tolerance = tolerance;
  my_system->GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;
  my_system->GetSettings()->solver.clamp_bilaterals = clamp_bilaterals;
  my_system->GetSettings()->solver.bilateral_clamp_speed = bilateral_clamp_speed;

#ifdef USE_DEM
  my_system->GetSettings()->solver.contact_force_model = HERTZ;
  my_system->GetSettings()->solver.tangential_displ_mode = MULTI_STEP;
#else
  my_system->GetSettings()->solver.solver_mode = SLIDING;
  my_system->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
  my_system->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
  my_system->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
  my_system->GetSettings()->solver.alpha = 0;
  my_system->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
  my_system->ChangeSolverType(APGD);

  my_system->GetSettings()->collision.collision_envelope = 0.05 * radius;
#endif

  my_system->GetSettings()->collision.bins_per_axis = I3(10, 10, 10);
  my_system->GetSettings()->collision.narrowphase_algorithm = NARROWPHASE_HYBRID_MPR;



  // Create a ball material (will be used by balls only)

#ifdef USE_DEM
  ChSharedPtr<ChMaterialSurfaceDEM> material;
  material = ChSharedPtr<ChMaterialSurfaceDEM>(new ChMaterialSurfaceDEM);
  material->SetYoungModulus(Y);
  material->SetPoissonRatio(nu);
  material->SetRestitution(COR);
  material->SetFriction(mu);
#else
  ChSharedPtr<ChMaterialSurface> material;
  material = ChSharedPtr<ChMaterialSurface>(new ChMaterialSurface);
  material->SetRestitution(COR);
  material->SetFriction(mu);
#endif

  // Create a material for all objects other than balls

#ifdef USE_DEM
  ChSharedPtr<ChMaterialSurfaceDEM> mat_ext;
  mat_ext = ChSharedPtr<ChMaterialSurfaceDEM>(new ChMaterialSurfaceDEM);
  mat_ext->SetYoungModulus(Y);
  mat_ext->SetPoissonRatio(nu);
  mat_ext->SetRestitution(COR);
  mat_ext->SetFriction(mu_ext);
#else
  ChSharedPtr<ChMaterialSurface> mat_ext;
  mat_ext = ChSharedPtr<ChMaterialSurface>(new ChMaterialSurface);
  mat_ext->SetRestitution(COR);
  mat_ext->SetFriction(mu_ext);
#endif

  // Create lower plate (ground, fixed)

  ChSharedPtr<ChBody> ground(new ChBody(new ChCollisionModelParallel, contact_method));

  ground->SetIdentifier(groundId);
  ground->SetMass(mass);
  ground->SetPos(ChVector<>(0, 0, -height / 2));
  ground->SetBodyFixed(true);
  ground->SetCollide(true);

  ground->SetMaterialSurface(mat_ext);

  ground->GetCollisionModel()->ClearModel();
  AddWall(ground, ChVector<>(width, length, thickness / 2), ChVector<>(0, 0, 0), true);
  ground->GetCollisionModel()->SetFamily(1);
  ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
  ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(3);
  ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);
  ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(5);
  ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(6);
  ground->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  ground->GetCollisionModel()->BuildModel();

  my_system->AddBody(ground);

  // Create wall 1

  ChSharedPtr<ChBody> wall_1(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_1->SetIdentifier(wall_1Id);
  wall_1->SetMass(mass);
  wall_1->SetPos(ChVector<>(-width / 2 - thickness / 2, 0, 0));
  wall_1->SetBodyFixed(true);
  wall_1->SetCollide(true);

  wall_1->SetMaterialSurface(mat_ext);

  wall_1->GetCollisionModel()->ClearModel();
  AddWall(wall_1, ChVector<>(thickness / 2, length, height), ChVector<>(0, 0, 0), true);
  wall_1->GetCollisionModel()->SetFamily(2);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(3);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(5);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(6);
  wall_1->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  wall_1->GetCollisionModel()->BuildModel();

  my_system->AddBody(wall_1);

  // Create wall 2

  ChSharedPtr<ChBody> wall_2(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_2->SetIdentifier(wall_2Id);
  wall_2->SetMass(mass);
  wall_2->SetPos(ChVector<>(width / 2 + thickness / 2, 0, 0));
  wall_2->SetBodyFixed(true);
  wall_2->SetCollide(true);

  wall_2->SetMaterialSurface(mat_ext);

  wall_2->GetCollisionModel()->ClearModel();
  AddWall(wall_2, ChVector<>(thickness / 2, length, height), ChVector<>(0, 0, 0), false);
  wall_2->GetCollisionModel()->SetFamily(3);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(5);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(6);
  wall_2->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  wall_2->GetCollisionModel()->BuildModel();

  my_system->AddBody(wall_2);

  // Create wall 3

  ChSharedPtr<ChBody> wall_3(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_3->SetIdentifier(wall_3Id);
  wall_3->SetMass(mass);
  wall_3->SetPos(ChVector<>(0, -length / 2 - thickness / 2, 0));
  wall_3->SetBodyFixed(true);
  wall_3->SetCollide(true);

  wall_3->SetMaterialSurface(mat_ext);

  wall_3->GetCollisionModel()->ClearModel();
  AddWall(wall_3, ChVector<>(width, thickness / 2, height), ChVector<>(0, 0, 0), true);
  wall_3->GetCollisionModel()->SetFamily(4);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(3);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(5);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(6);
  wall_3->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  wall_3->GetCollisionModel()->BuildModel();

  my_system->AddBody(wall_3);

  // Create wall 4

  ChSharedPtr<ChBody> wall_4(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_4->SetIdentifier(wall_4Id);
  wall_4->SetMass(mass);
  wall_4->SetPos(ChVector<>(0, length / 2 + thickness / 2, 0));
  wall_4->SetBodyFixed(true);
  wall_4->SetCollide(true);

  wall_4->SetMaterialSurface(mat_ext);

  wall_4->GetCollisionModel()->ClearModel();
  AddWall(wall_4, ChVector<>(width, thickness / 2, height), ChVector<>(0, 0, 0), true);
  wall_4->GetCollisionModel()->SetFamily(5);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(3);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(6);
  wall_4->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  wall_4->GetCollisionModel()->BuildModel();

  my_system->AddBody(wall_4);

  // Create upper plate (prescribed velocity)

  ChSharedPtr<ChBody> plate(new ChBody(new ChCollisionModelParallel, contact_method));

  plate->SetIdentifier(plateId);
  plate->SetMass(mass);
  plate->SetPos(ChVector<>(0, 0, 2.0 * radius * float(a) + thickness));
  plate->SetBodyFixed(false);
  plate->SetCollide(true);

  plate->SetMaterialSurface(mat_ext);

  plate->GetCollisionModel()->ClearModel();
  AddWall(plate, ChVector<>(width, length, thickness / 2), ChVector<>(0, 0, 0), true);
  plate->GetCollisionModel()->SetFamily(6);
  plate->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  plate->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
  plate->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(3);
  plate->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);
  plate->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(5);
  plate->GetCollisionModel()->SetFamilyMaskDoCollisionWithFamily(7);
  plate->GetCollisionModel()->BuildModel();

  my_system->AddBody(plate);

  // Create (a X b X c) many falling balls

  int i, j, k;
  double ball_x, ball_y, ball_z;

  for (i = 0; i < a; i++) {
    for (j = 0; j < b; j++) {
      for (k = 0; k < c; k++) {
        ball_z = 2.0 * radius * float(i);

        ball_x = 4.0 * radius * (float(j - b / 2) + 0.5) + 0.99 * radius * (float(rand() % 100) / 50 - 1.0);
        ball_y = 4.0 * radius * (float(k - c / 2) + 0.5) + 0.99 * radius * (float(rand() % 100) / 50 - 1.0);

        ChSharedPtr<ChBody> ball(new ChBody(new ChCollisionModelParallel, contact_method));

        ball->SetIdentifier(ballId + 6 * 6 * i + 6 * j + k);
        ball->SetMass(mass);
        ball->SetInertiaXX((2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1));
        ball->SetPos(ChVector<>(ball_x, ball_y, ball_z));
        ball->SetBodyFixed(false);
        ball->SetCollide(true);

        ball->SetMaterialSurface(material);

        ball->GetCollisionModel()->ClearModel();
        ball->GetCollisionModel()->AddSphere(radius);
        ball->GetCollisionModel()->SetFamily(7);
        ball->GetCollisionModel()->BuildModel();

        ChSharedPtr<ChSphereShape> sphere(new ChSphereShape);

        sphere->GetSphereGeometry().rad = radius;
        sphere->SetColor(ChColor(1, 0, 1));
        ball->AddAsset(sphere);

        my_system->AddBody(ball);
      }
    }
  }

  // Create prismatic (translational) joint between the plate and the ground.
  // The translational axis of a prismatic joint is along the Z axis of the
  // specified joint coordinate system.  Here, we apply the 'z2y' rotation to
  // align it with the Y axis of the global reference frame.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_plate_ground(new ChLinkLockPrismatic);
  prismatic_plate_ground->SetName("prismatic_plate_ground");
  prismatic_plate_ground->Initialize(plate, ground, ChCoordsys<>(ChVector<>(0, 0, 0), QUNIT));
  my_system->AddLink(prismatic_plate_ground);

  // Create prismatic (translational) joints between each of the four walls and the ground.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_1_ground(new ChLinkLockPrismatic);
  prismatic_wall_1_ground->SetName("prismatic_wall_1_ground");
  prismatic_wall_1_ground->Initialize(wall_1, ground, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));
  my_system->AddLink(prismatic_wall_1_ground);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_2_ground(new ChLinkLockPrismatic);
  prismatic_wall_2_ground->SetName("prismatic_wall_2_ground");
  prismatic_wall_2_ground->Initialize(wall_2, ground, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));
  my_system->AddLink(prismatic_wall_2_ground);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_3_ground(new ChLinkLockPrismatic);
  prismatic_wall_3_ground->SetName("prismatic_wall_3_ground");
  prismatic_wall_3_ground->Initialize(wall_3, ground, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));
  my_system->AddLink(prismatic_wall_3_ground);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_4_ground(new ChLinkLockPrismatic);
  prismatic_wall_4_ground->SetName("prismatic_wall_4_ground");
  prismatic_wall_4_ground->Initialize(wall_4, ground, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));
  my_system->AddLink(prismatic_wall_4_ground);

  // Setup output

  ChStreamOutAsciiFile stressStream(stress_file.c_str());
  ChStreamOutAsciiFile forceStream(force_file.c_str());
  ChStreamOutAsciiFile statsStream(stats_file.c_str());
  stressStream.SetNumFormat("%16.4e");
  forceStream.SetNumFormat("%16.4e");

  // Create the OpenGL visualization window

#ifdef CHRONO_PARALLEL_HAS_OPENGL
  opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
  gl_window.Initialize(800, 600, title.c_str(), my_system);
  gl_window.SetCamera(ChVector<>(3 * width, 0, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), radius, radius);
  gl_window.SetRenderMode(opengl::SOLID);
#endif

  // Begin simulation

  bool settling = true;
  bool testing = false;

  int data_out_frame = 0;
  int visual_out_frame = 0;

  while (my_system->GetChTime() < end_simulation_time) {
    if (my_system->GetChTime() > settling_time && settling == true) {
      if (dense == true)
        material->SetFriction(0.01f);
      plate->SetBodyFixed(true);
      if (plain_strain == false) {
        wall_1->SetBodyFixed(false);
        wall_2->SetBodyFixed(false);
      }
      wall_3->SetBodyFixed(false);
      wall_4->SetBodyFixed(false);
      settling = false;
    }

    if (settling == false) {
      Lz = plate->GetPos().z - ground->GetPos().z - thickness;

      Lx = wall_2->GetPos().x - wall_1->GetPos().x - thickness;
      Ly = wall_4->GetPos().y - wall_3->GetPos().y - thickness;

      if (plain_strain == false) {
        wall_1->Empty_forces_accumulators();
        wall_2->Empty_forces_accumulators();
        wall_1->Accumulate_force(ChVector<>(sigma_b*Ly*Lz, 0, 0),wall_1->GetPos(),false);
        wall_2->Accumulate_force(ChVector<>(-sigma_b*Ly*Lz, 0, 0),wall_2->GetPos(),false);
      }
      wall_3->Empty_forces_accumulators();
      wall_4->Empty_forces_accumulators();
      wall_3->Accumulate_force(ChVector<>(0, sigma_c*Lx*Lz, 0),wall_3->GetPos(),false);
      wall_4->Accumulate_force(ChVector<>(0, -sigma_c*Lx*Lz, 0),wall_4->GetPos(),false);
    }

    if (my_system->GetChTime() > begin_test_time && testing == false) {
      if (dense == true)
        material->SetFriction(mu);
      Lx0 = Lx;
      Ly0 = Ly;
      Lz0 = Lz;
      height_0 = plate->GetPos().z;
      testing = true;
    }

    if (testing == true) {
      plate->SetPos(ChVector<>(0, 0, height_0 + axial_speed * begin_test_time - axial_speed * my_system->GetChTime()));
      plate->SetPos_dt(ChVector<>(0, 0, -axial_speed));
    }

    //  Do time step

#ifdef CHRONO_PARALLEL_HAS_OPENGL
    if (gl_window.Active()) {
      gl_window.DoStepDynamics(time_step);
      gl_window.Render();
    }
    else
      break;
#else
    my_system->DoStepDynamics(time_step);
#endif

//    TimingOutput(my_system, &statsStream);

    //  Output to files

    if (my_system->GetChTime() >= data_out_frame * data_out_step) {
#ifndef USE_DEM
      my_system->CalculateContactForces();
#endif
      force0 = my_system->GetBodyContactForce(0);
      force1 = my_system->GetBodyContactForce(1);
      force2 = my_system->GetBodyContactForce(2);
      force3 = my_system->GetBodyContactForce(3);
      force4 = my_system->GetBodyContactForce(4);
      force5 = my_system->GetBodyContactForce(5);

      forceStream << my_system->GetChTime() << "\t" << Lx << "\t" << Ly << "\t" << Lz << "\t";
      forceStream << force1.x << "\t" << force2.x << "\t" << force3.y << "\t" << force4.y << "\t";
      forceStream << force0.z << "\t" << force5.z << "\n";

      cout << my_system->GetChTime() << "\t" << Lx << "\t" << Ly << "\t" << Lz << "\t";
      cout << force1.x << "\t" << force2.x << "\t" << force3.y << "\t" << force4.y << "\t";
      cout << force0.z << "\t" << force5.z << "\n";

      //  Output to shear data file

      if (testing == true) {
        stressStream << Lx/Lx0 << "\t" << Ly/Ly0 << "\t" << Lz/Lz0 << "\t";
        stressStream << force1.x / (Ly * Lz) << "\t" << force2.x / (Ly * Lz) << "\t";
        stressStream << force3.y / (Lx * Lz) << "\t" << force4.y / (Lx * Lz) << "\t";
        stressStream << force0.z / (Lx * Ly) << "\t" << force5.z / (Lx * Ly) << "\n";

        cout << Lx/Lx0 << "\t" << Ly/Ly0 << "\t" << Lz/Lz0 << "\t";
        cout << force1.x / (Ly * Lz) << "\t" << force2.x / (Ly * Lz) << "\t";
        cout << force3.y / (Lx * Lz) << "\t" << force4.y / (Lx * Lz) << "\t";
        cout << force0.z / (Lx * Ly) << "\t" << force5.z / (Lx * Ly) << "\n";
      }

      data_out_frame++;
    }

    //  Output to POV-Ray

    if (write_povray_data && my_system->GetChTime() >= visual_out_frame * visual_out_step) {
      char filename[100];
      sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), visual_out_frame + 1);
      utils::WriteShapesPovray(my_system, filename, false);

      visual_out_frame++;
    }
  }

  return 0;
}
