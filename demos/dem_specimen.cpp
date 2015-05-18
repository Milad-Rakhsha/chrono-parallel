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
// Soft-sphere (DEM/DEM-P) or hard-sphere (DVI/DEM-C) granular material specimen
// generator.
//
// The global reference frame has Z up.
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
// Specimen definition
// -----------------------------------------------------------------------------

bool visible_walls = true;

double gravity = 9.81; // m/s^2
const int maxParticleTypes = 5;
int numParticleTypes, i;

// Predefined specimen particle size distributions
enum SpecimenDistrib { UNIFORM_BEADS, OTTAWA_SAND };
SpecimenDistrib distrib = UNIFORM_BEADS;

// Predefined specimen materials
enum SpecimenMaterial { GLASS, QUARTZ };
SpecimenMaterial material = GLASS;

// Predefined specimen geometries
enum SpecimenGeom { HARTL_OOI, STANDARD_BOX, SMALL_BOX, JENIKE_SHEAR, STANDARD_TRIAXIAL, SMALL_TRIAXIAL };
SpecimenGeom geom = SMALL_BOX;

// Confining pre-stress properties (Pa)
//double sigma_a = 24.2e3;
//double sigma_b = 24.2e3;
//double sigma_c = 24.2e3;

//double sigma_a = 12.5e3;
//double sigma_b = 12.5e3;
//double sigma_c = 12.5e3;

//double sigma_a = 6.4e3;
//double sigma_b = 6.4e3;
//double sigma_c = 6.4e3;

double sigma_a = 3.1e3;
double sigma_b = 3.1e3;
double sigma_c = 3.1e3;

// Packing density
bool dense = false;

// Comment the following line to use DVI contact
#define USE_DEM

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 1;

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
double settling_time = 0.2;
double simulation_time = 0.3;
#else
double settling_time = 0.2;
double simulation_time = 0.3;
#endif

// Output
#ifdef USE_DEM
const std::string out_dir = "../SPECIMEN_DEM";
#else
const std::string out_dir = "../SPECIMEN_DVI";
#endif

const std::string pov_dir = out_dir + "/POVRAY";
const std::string stress_file = out_dir + "/specimen_stress.dat";
const std::string force_file = out_dir + "/specimen_force.dat";
const std::string stats_file = out_dir + "/stats.dat";

bool write_povray_data = true;

double data_out_step = 1e-3;       // time interval between data outputs
double visual_out_step = 1e-3;     // time interval between PovRay outputs

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

  // Particle size distribution (PSD) properties
  double particleDiameter[maxParticleTypes];
  double percentbyWeight[maxParticleTypes];

  switch (distrib) {
  case UNIFORM_BEADS: // Uniform size 6 mm diameter beads
	numParticleTypes = 1;
  	particleDiameter[0] = 0.006; // m
  	percentbyWeight[0] = 1.00; // x100 %
  	break;
  case OTTAWA_SAND: // ASTM C 778-06 standard graded Ottawa sand
	numParticleTypes = 4;
  	particleDiameter[0] = 0.0003; // m
  	particleDiameter[1] = 0.0004; // m
  	particleDiameter[2] = 0.0006; // m
  	particleDiameter[3] = 0.0008; // m

  	percentbyWeight[0] = 0.20; // x100 %
  	percentbyWeight[1] = 0.45; // x100 %
  	percentbyWeight[2] = 0.30; // x100 %
  	percentbyWeight[3] = 0.05; // x100 %
  	break;
  }

  // Particle material properties
  double particleDensity[maxParticleTypes];
  double particleMass[maxParticleTypes];
  float Y[maxParticleTypes];
  float nu[maxParticleTypes];
  float COR[maxParticleTypes];
  float mu[maxParticleTypes];

  switch (material) {
  case GLASS:
  	for (i = 0; i < numParticleTypes; i++) {
  		particleDensity[i] = 2550; // kg/m^3
  		particleMass[i] = particleDensity[i] * CH_C_PI * particleDiameter[i] * particleDiameter[i] * particleDiameter[i] / 6.0;
  		Y[i] = 4.0e7; // Pa (about 1000 times too soft)
  		nu[i] = 0.22f;
  		COR[i] = 0.87f;
  		mu[i] = 0.18f; // particle-on-particle friction
  	}
  	break;
  case QUARTZ:
  	for (i = 0; i < numParticleTypes; i++) {
  		particleDensity[i] = 2650; // kg/m^3
  		particleMass[i] = particleDensity[i] * CH_C_PI * particleDiameter[i] * particleDiameter[i] * particleDiameter[i] / 6.0;
  		Y[i] = 8.0e7; // Pa (about 1000 times too soft)
  		nu[i] = 0.3f;
  		COR[i] = 0.5f;
  		mu[i] = 0.5f; // particle-on-particle friction
  	}
  	break;
  }

  // Containing wall material properties
  float Y_ext = 1.0e7; // Pa
  float nu_ext = 0.3;
  float COR_ext = 0.5;
  float mu_ext = 0.01f; // particle-on-wall friction

  // Unconsolidated specimen dimensions
  double Lx0, Ly0, Lz0;
  bool cylinder = false;

  switch (geom) {
  case HARTL_OOI:
  	Lx0 = 0.12; // m
  	Ly0 = 0.12; // m
  	Lz0 = 0.24; // m
  	break;
  case STANDARD_BOX:
  	Lx0 = 0.06; // m
  	Ly0 = 0.06; // m
  	Lz0 = 0.12; // m
  	break;
  case SMALL_BOX:
  	Lx0 = 0.01; // m
  	Ly0 = 0.01; // m
  	Lz0 = 0.10; // m
  	break;
  case JENIKE_SHEAR:
  	Lx0 = Ly0 = 0.06; // m
  	Lz0 = 0.12; // m
  	cylinder = true;
  	break;
  case STANDARD_TRIAXIAL:
  	Lx0 = Ly0 = 0.06; // m
  	Lz0 = 0.24; // m
  	cylinder = true;
  	break;
  case SMALL_TRIAXIAL:
  	Lx0 = Ly0 = 0.02; // m
  	Lz0 = 0.06; // m
  	cylinder = true;
  	break;
  }

  // Packing properties
  double percentbyNumber[maxParticleTypes];

  for (i = 0; i < numParticleTypes; i++) {
	  percentbyNumber[i] = percentbyWeight[i]; // Not true at all...
  }

  int particle_Id = 1;  // first particle id
  int wall_1Id = -1;
  int wall_2Id = -2;
  int wall_3Id = -3;
  int wall_4Id = -4;
  int wall_5Id = -5;
  int wall_6Id = -6;
  int wall_7Id = -7;
  int wall_8Id = -8;
  int wall_9Id = -9;
  int wall_10Id = -10;
  int wall_11Id = -11;
  int wall_12Id = -12;
  int wall_13Id = -13;
  int wall_14Id = -14;

  double Lx, Ly, Lz, diam;
  double max_diameter = 0;
  double max_mass = 0;

  for (i = 0; i < numParticleTypes; i++) {
	  if (particleDiameter[i] > max_diameter) max_diameter = particleDiameter[i];
	  if (particleMass[i] > max_mass) max_mass = particleMass[i];
  }
  double thickness = 1.0*max_diameter;
  double wall_mass = 1.0*max_mass;

  // Define five quaternions representing:
  // - a rotation of -90 degrees around x (z2y)
  // - a rotation of +90 degrees around y (z2x)
  // - a rotation of +30 degrees around z (z30)
  // - a rotation of +45 degrees around z (z45)
  // - a rotation of +60 degrees around z (z60)

  ChQuaternion<> z2y;
  ChQuaternion<> z2x;
  ChQuaternion<> z30;
  ChQuaternion<> z45;
  ChQuaternion<> z60;
  z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));
  z2x.Q_from_AngAxis(CH_C_PI / 2, ChVector<>(0, 1, 0));
  z30.Q_from_AngAxis(CH_C_PI / 6, ChVector<>(0, 0, 1));
  z45.Q_from_AngAxis(CH_C_PI / 4, ChVector<>(0, 0, 1));
  z60.Q_from_AngAxis(CH_C_PI / 3, ChVector<>(0, 0, 1));

  // Create the system

#ifdef USE_DEM
  cout << "Create DEM system" << endl;
  const std::string title = "soft-sphere (DEM/DEM-P) granular material specimen";
  ChBody::ContactMethod contact_method = ChBody::DEM;
  ChSystemParallelDEM* my_system = new ChSystemParallelDEM();
#else
  cout << "Create DVI system" << endl;
  const std::string title = "hard-sphere (DVI/DEM-C) granular material specimen";
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

  // Create wall material

#ifdef USE_DEM
  ChSharedPtr<ChMaterialSurfaceDEM> mat_ext;
  mat_ext = ChSharedPtr<ChMaterialSurfaceDEM>(new ChMaterialSurfaceDEM);
  mat_ext->SetYoungModulus(Y_ext);
  mat_ext->SetPoissonRatio(nu_ext);
  mat_ext->SetRestitution(COR_ext);
  mat_ext->SetFriction(mu_ext);
#else
  ChSharedPtr<ChMaterialSurface> mat_ext;
  mat_ext = ChSharedPtr<ChMaterialSurface>(new ChMaterialSurface);
  mat_ext->SetRestitution(COR);
  mat_ext->SetFriction(mu_ext);
#endif

  // Create wall 1 (bottom)

  ChSharedPtr<ChBody> wall_1(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_1->SetIdentifier(wall_1Id);
  wall_1->SetMass(wall_mass);
  wall_1->SetPos(ChVector<>(0, 0, -thickness / 2));
  wall_1->SetBodyFixed(true);
  wall_1->SetCollide(true);

  wall_1->SetMaterialSurface(mat_ext);

  wall_1->GetCollisionModel()->ClearModel();
  wall_1->GetCollisionModel()->AddBox(Lx0, Ly0, thickness / 2, ChVector<>(0, 0, 0));
  wall_1->GetCollisionModel()->SetFamily(1);
  wall_1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_1->GetCollisionModel()->BuildModel();

  // Create wall 2 (top)

  ChSharedPtr<ChBody> wall_2(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_2->SetIdentifier(wall_2Id);
  wall_2->SetMass(wall_mass);
  wall_2->SetPos(ChVector<>(0, 0, Lz0 + thickness / 2));
  wall_2->SetBodyFixed(false);
  wall_2->SetCollide(true);

  wall_2->SetMaterialSurface(mat_ext);

  wall_2->GetCollisionModel()->ClearModel();
  wall_2->GetCollisionModel()->AddBox(Lx0, Ly0, thickness / 2, ChVector<>(0, 0, 0));
  wall_2->GetCollisionModel()->SetFamily(1);
  wall_2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_2->GetCollisionModel()->BuildModel();

  // Create wall 3 (side)

  ChSharedPtr<ChBody> wall_3(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_3->SetIdentifier(wall_3Id);
  wall_3->SetMass(wall_mass);
  wall_3->SetPos(ChVector<>(-(Lx0 / 2 + thickness / 2), 0, Lz0 / 2));
  wall_3->SetBodyFixed(false);
  wall_3->SetCollide(true);

  wall_3->SetMaterialSurface(mat_ext);

  wall_3->GetCollisionModel()->ClearModel();
  wall_3->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
  wall_3->GetCollisionModel()->SetFamily(1);
  wall_3->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_3->GetCollisionModel()->BuildModel();

  // Create wall 4 (side)

  ChSharedPtr<ChBody> wall_4(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_4->SetIdentifier(wall_4Id);
  wall_4->SetMass(wall_mass);
  wall_4->SetPos(ChVector<>(Lx0 / 2 + thickness / 2, 0, Lz0 / 2));
  wall_4->SetBodyFixed(false);
  wall_4->SetCollide(true);

  wall_4->SetMaterialSurface(mat_ext);

  wall_4->GetCollisionModel()->ClearModel();
  wall_4->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
  wall_4->GetCollisionModel()->SetFamily(1);
  wall_4->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_4->GetCollisionModel()->BuildModel();

  // Create wall 5 (side)

  ChSharedPtr<ChBody> wall_5(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_5->SetIdentifier(wall_5Id);
  wall_5->SetMass(wall_mass);
  wall_5->SetPos(ChVector<>(0, -(Ly0 / 2 + thickness / 2), Lz0 / 2));
  wall_5->SetBodyFixed(false);
  wall_5->SetCollide(true);

  wall_5->SetMaterialSurface(mat_ext);

  wall_5->GetCollisionModel()->ClearModel();
  wall_5->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
  wall_5->GetCollisionModel()->SetFamily(1);
  wall_5->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_5->GetCollisionModel()->BuildModel();

  // Create wall 6 (side)

  ChSharedPtr<ChBody> wall_6(new ChBody(new ChCollisionModelParallel, contact_method));

  wall_6->SetIdentifier(wall_6Id);
  wall_6->SetMass(wall_mass);
  wall_6->SetPos(ChVector<>(0, Ly0 / 2 + thickness / 2, Lz0 / 2));
  wall_6->SetBodyFixed(false);
  wall_6->SetCollide(true);

  wall_6->SetMaterialSurface(mat_ext);

  wall_6->GetCollisionModel()->ClearModel();
  wall_6->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
  wall_6->GetCollisionModel()->SetFamily(1);
  wall_6->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
  wall_6->GetCollisionModel()->BuildModel();

  // For cylinder only

	  // Create wall 7 (side)

	  ChSharedPtr<ChBody> wall_7(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_7->SetIdentifier(wall_7Id);
	  wall_7->SetMass(wall_mass);
	  wall_7->SetPos(ChVector<>(-(Lx0 / 2 + thickness / 2) * cos(CH_C_PI / 6),
			  -(Ly0 / 2 + thickness / 2) * sin(CH_C_PI / 6),
			  Lz0 / 2));
	  wall_7->SetRot(z30);
	  wall_7->SetBodyFixed(false);
	  wall_7->SetCollide(true);

	  wall_7->SetMaterialSurface(mat_ext);

	  wall_7->GetCollisionModel()->ClearModel();
	  wall_7->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
	  wall_7->GetCollisionModel()->SetFamily(1);
	  wall_7->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_7->GetCollisionModel()->BuildModel();

	  // Create wall 8 (side)

	  ChSharedPtr<ChBody> wall_8(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_8->SetIdentifier(wall_8Id);
	  wall_8->SetMass(wall_mass);
	  wall_8->SetPos(ChVector<>((Lx0 / 2 + thickness / 2) * cos(CH_C_PI / 6),
			  (Ly0 / 2 + thickness / 2) * sin(CH_C_PI / 6),
			  Lz0 / 2));
	  wall_8->SetRot(z30);
	  wall_8->SetBodyFixed(false);
	  wall_8->SetCollide(true);

	  wall_8->SetMaterialSurface(mat_ext);

	  wall_8->GetCollisionModel()->ClearModel();
	  wall_8->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
	  wall_8->GetCollisionModel()->SetFamily(1);
	  wall_8->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_8->GetCollisionModel()->BuildModel();

	  // Create wall 9 (side)

	  ChSharedPtr<ChBody> wall_9(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_9->SetIdentifier(wall_9Id);
	  wall_9->SetMass(wall_mass);
	  wall_9->SetPos(ChVector<>(-(Lx0 / 2 + thickness / 2) * cos(CH_C_PI / 3),
			  -(Ly0 / 2 + thickness / 2) * sin(CH_C_PI / 3),
			  Lz0 / 2));
	  wall_9->SetRot(z60);
	  wall_9->SetBodyFixed(false);
	  wall_9->SetCollide(true);

	  wall_9->SetMaterialSurface(mat_ext);

	  wall_9->GetCollisionModel()->ClearModel();
	  wall_9->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
	  wall_9->GetCollisionModel()->SetFamily(1);
	  wall_9->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_9->GetCollisionModel()->BuildModel();

	  // Create wall 10 (side)

	  ChSharedPtr<ChBody> wall_10(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_10->SetIdentifier(wall_10Id);
	  wall_10->SetMass(wall_mass);
	  wall_10->SetPos(ChVector<>((Lx0 / 2 + thickness / 2) * cos(CH_C_PI / 3),
			  (Ly0 / 2 + thickness / 2) * sin(CH_C_PI / 3),
			  Lz0 / 2));
	  wall_10->SetRot(z60);
	  wall_10->SetBodyFixed(false);
	  wall_10->SetCollide(true);

	  wall_10->SetMaterialSurface(mat_ext);

	  wall_10->GetCollisionModel()->ClearModel();
	  wall_10->GetCollisionModel()->AddBox(thickness / 2, Ly0, Lz0, ChVector<>(0, 0, 0));
	  wall_10->GetCollisionModel()->SetFamily(1);
	  wall_10->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_10->GetCollisionModel()->BuildModel();

	  // Create wall 11 (side)

	  ChSharedPtr<ChBody> wall_11(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_11->SetIdentifier(wall_11Id);
	  wall_11->SetMass(wall_mass);
	  wall_11->SetPos(ChVector<>((Lx0 / 2 + thickness / 2) * sin(CH_C_PI / 6),
			  -(Ly0 / 2 + thickness / 2) * cos(CH_C_PI / 6),
			  Lz0 / 2));
	  wall_11->SetRot(z30);
	  wall_11->SetBodyFixed(false);
	  wall_11->SetCollide(true);

	  wall_11->SetMaterialSurface(mat_ext);

	  wall_11->GetCollisionModel()->ClearModel();
	  wall_11->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
	  wall_11->GetCollisionModel()->SetFamily(1);
	  wall_11->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_11->GetCollisionModel()->BuildModel();

	  // Create wall 12 (side)

	  ChSharedPtr<ChBody> wall_12(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_12->SetIdentifier(wall_12Id);
	  wall_12->SetMass(wall_mass);
	  wall_12->SetPos(ChVector<>(-(Lx0 / 2 + thickness / 2) * sin(CH_C_PI / 6),
			  (Ly0 / 2 + thickness / 2) * cos(CH_C_PI / 6),
			  Lz0 / 2));
	  wall_12->SetRot(z30);
	  wall_12->SetBodyFixed(false);
	  wall_12->SetCollide(true);

	  wall_12->SetMaterialSurface(mat_ext);

	  wall_12->GetCollisionModel()->ClearModel();
	  wall_12->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
	  wall_12->GetCollisionModel()->SetFamily(1);
	  wall_12->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_12->GetCollisionModel()->BuildModel();

	  // Create wall 13 (side)

	  ChSharedPtr<ChBody> wall_13(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_13->SetIdentifier(wall_13Id);
	  wall_13->SetMass(wall_mass);
	  wall_13->SetPos(ChVector<>((Lx0 / 2 + thickness / 2) * sin(CH_C_PI / 3),
			  -(Ly0 / 2 + thickness / 2) * cos(CH_C_PI / 3),
			  Lz0 / 2));
	  wall_13->SetRot(z60);
	  wall_13->SetBodyFixed(false);
	  wall_13->SetCollide(true);

	  wall_13->SetMaterialSurface(mat_ext);

	  wall_13->GetCollisionModel()->ClearModel();
	  wall_13->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
	  wall_13->GetCollisionModel()->SetFamily(1);
	  wall_13->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_13->GetCollisionModel()->BuildModel();

	  // Create wall 14 (side)

	  ChSharedPtr<ChBody> wall_14(new ChBody(new ChCollisionModelParallel, contact_method));

	  wall_14->SetIdentifier(wall_14Id);
	  wall_14->SetMass(wall_mass);
	  wall_14->SetPos(ChVector<>(-(Lx0 / 2 + thickness / 2) * sin(CH_C_PI / 3),
			  (Ly0 / 2 + thickness / 2) * cos(CH_C_PI / 3),
			  Lz0 / 2));
	  wall_14->SetRot(z60);
	  wall_14->SetBodyFixed(false);
	  wall_14->SetCollide(true);

	  wall_14->SetMaterialSurface(mat_ext);

	  wall_14->GetCollisionModel()->ClearModel();
	  wall_14->GetCollisionModel()->AddBox(Lx0, thickness / 2, Lz0, ChVector<>(0, 0, 0));
	  wall_14->GetCollisionModel()->SetFamily(1);
	  wall_14->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
	  wall_14->GetCollisionModel()->BuildModel();

  // Add all walls to system

  my_system->AddBody(wall_1);
  my_system->AddBody(wall_2);
  my_system->AddBody(wall_3);
  my_system->AddBody(wall_4);
  my_system->AddBody(wall_5);
  my_system->AddBody(wall_6);

  if (cylinder == true) {
	  my_system->AddBody(wall_7);
	  my_system->AddBody(wall_8);
	  my_system->AddBody(wall_9);
	  my_system->AddBody(wall_10);
	  my_system->AddBody(wall_11);
	  my_system->AddBody(wall_12);
	  my_system->AddBody(wall_13);
	  my_system->AddBody(wall_14);
  }

  if (visible_walls == true) {
	  ChSharedPtr<ChBoxShape> box_1(new ChBoxShape);
	  ChSharedPtr<ChBoxShape> box_2(new ChBoxShape);
	  ChSharedPtr<ChBoxShape> box_3(new ChBoxShape);

	  box_1->GetBoxGeometry().Size = ChVector<>(Lx0, Ly0, thickness / 2);
	  box_1->Pos = ChVector<>(0, 0, 0);
	  box_2->GetBoxGeometry().Size = ChVector<>(thickness / 2, Ly0, Lz0);
	  box_2->Pos = ChVector<>(0, 0, 0);
	  box_3->GetBoxGeometry().Size = ChVector<>(Lx0, thickness / 2, Lz0);
	  box_3->Pos = ChVector<>(0, 0, 0);

	  wall_1->AddAsset(box_1);
	  wall_2->AddAsset(box_1);
	  wall_3->AddAsset(box_2);
	  wall_4->AddAsset(box_2);
	  wall_5->AddAsset(box_3);
	  wall_6->AddAsset(box_3);

	  if (cylinder == true) {
		  wall_7->AddAsset(box_2);
		  wall_8->AddAsset(box_2);
		  wall_9->AddAsset(box_2);
		  wall_10->AddAsset(box_2);
		  wall_11->AddAsset(box_3);
		  wall_12->AddAsset(box_3);
		  wall_13->AddAsset(box_3);
		  wall_14->AddAsset(box_3);
	  }
  }

  // Create the particle generator

  utils::Generator gen(my_system);

#ifdef USE_DEM
  ChSharedPtr<ChMaterialSurfaceDEM> mat[maxParticleTypes];
#else
  ChSharedPtr<ChMaterialSurfaceDVI> mat[maxParticleTypes];
#endif

  for (i = 0; i < numParticleTypes; i++) {
	  utils::MixtureIngredientPtr& type = gen.AddMixtureIngredient(utils::SPHERE, percentbyNumber[i]);
#ifdef USE_DEM
	  mat[i] = ChSharedPtr<ChMaterialSurfaceDEM>(new ChMaterialSurfaceDEM);
	  mat[i]->SetYoungModulus(Y[i]);
	  mat[i]->SetPoissonRatio(nu[i]);
	  mat[i]->SetRestitution(COR[i]);
	  mat[i]->SetFriction(mu[i]);
	  type->setDefaultMaterialDEM(mat[i]);
#else
	  mat[i] = ChSharedPtr<ChMaterialSurfaceDVI>(new ChMaterialSurfaceDVI);
	  mat[i]->SetRestitution(COR[i]);
	  mat[i]->SetFriction(mu[i]);
	  type->setDefaultMaterialDVI(mat[i]);
#endif
	  type->setDefaultDensity(particleDensity[i]);
	  type->setDefaultSize(particleDiameter[i] / 2);
  }

  // Ensure that all generated particle bodies will have positive IDs.
  gen.setBodyIdentifier(particle_Id);

  // Generate the particles

  if (cylinder == true) {
    gen.createObjectsCylinderZ(utils::POISSON_DISK, max_diameter,
		ChVector<>(0, 0, Lz0 / 2),
		(Lx0 - max_diameter) / 2, (Lz0 - max_diameter) / 2);
  } else {
    gen.createObjectsBox(utils::POISSON_DISK, max_diameter,
    	ChVector<>(0, 0, Lz0 / 2),
    	ChVector<>((Lx0 - max_diameter) / 2, (Ly0 - max_diameter) / 2, (Lz0 - max_diameter) / 2));
  }

  // Create prismatic (translational) joint between the plate and the ground.
  // The translational axis of a prismatic joint is along the Z axis of the
  // specified joint coordinate system.  Here, we apply the 'z2y' rotation to
  // align it with the Y axis of the global reference frame.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_2_1(new ChLinkLockPrismatic);
  prismatic_wall_2_1->SetName("prismatic_wall_2_1");
  prismatic_wall_2_1->Initialize(wall_2, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), QUNIT));
  my_system->AddLink(prismatic_wall_2_1);

  // Create prismatic (translational) joints between each of the four walls and the ground.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_3_1(new ChLinkLockPrismatic);
  prismatic_wall_3_1->SetName("prismatic_wall_3_1");
  prismatic_wall_3_1->Initialize(wall_3, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));
  my_system->AddLink(prismatic_wall_3_1);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_4_1(new ChLinkLockPrismatic);
  prismatic_wall_4_1->SetName("prismatic_wall_4_1");
  prismatic_wall_4_1->Initialize(wall_4, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));
  my_system->AddLink(prismatic_wall_4_1);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_5_1(new ChLinkLockPrismatic);
  prismatic_wall_5_1->SetName("prismatic_wall_5_1");
  prismatic_wall_5_1->Initialize(wall_5, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));
  my_system->AddLink(prismatic_wall_5_1);

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_6_1(new ChLinkLockPrismatic);
  prismatic_wall_6_1->SetName("prismatic_wall_6_1");
  prismatic_wall_6_1->Initialize(wall_6, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));
  my_system->AddLink(prismatic_wall_6_1);

  ChQuaternion<> z2wall_7_8 = z30%z2x;
  ChQuaternion<> z2wall_9_10 = z60%z2x;
  ChQuaternion<> z2wall_11_12 = z30%z2y;
  ChQuaternion<> z2wall_13_14 = z60%z2y;

  if (cylinder == true) {
	  // Create prismatic (translational) joints between an additional eight walls and the ground.

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_7_1(new ChLinkLockPrismatic);
	  prismatic_wall_7_1->SetName("prismatic_wall_7_1");
	  prismatic_wall_7_1->Initialize(wall_7, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_7_8));
	  my_system->AddLink(prismatic_wall_7_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_8_1(new ChLinkLockPrismatic);
	  prismatic_wall_8_1->SetName("prismatic_wall_8_1");
	  prismatic_wall_8_1->Initialize(wall_8, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_7_8));
	  my_system->AddLink(prismatic_wall_8_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_9_1(new ChLinkLockPrismatic);
	  prismatic_wall_9_1->SetName("prismatic_wall_9_1");
	  prismatic_wall_9_1->Initialize(wall_9, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_9_10));
	  my_system->AddLink(prismatic_wall_9_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_10_1(new ChLinkLockPrismatic);
	  prismatic_wall_10_1->SetName("prismatic_wall_10_1");
	  prismatic_wall_10_1->Initialize(wall_10, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_9_10));
	  my_system->AddLink(prismatic_wall_10_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_11_1(new ChLinkLockPrismatic);
	  prismatic_wall_11_1->SetName("prismatic_wall_11_1");
	  prismatic_wall_11_1->Initialize(wall_11, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_11_12));
	  my_system->AddLink(prismatic_wall_11_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_12_1(new ChLinkLockPrismatic);
	  prismatic_wall_12_1->SetName("prismatic_wall_12_1");
	  prismatic_wall_12_1->Initialize(wall_12, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_11_12));
	  my_system->AddLink(prismatic_wall_12_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_13_1(new ChLinkLockPrismatic);
	  prismatic_wall_13_1->SetName("prismatic_wall_13_1");
	  prismatic_wall_13_1->Initialize(wall_13, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_13_14));
	  my_system->AddLink(prismatic_wall_13_1);

	  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_14_1(new ChLinkLockPrismatic);
	  prismatic_wall_14_1->SetName("prismatic_wall_14_1");
	  prismatic_wall_14_1->Initialize(wall_14, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_13_14));
	  my_system->AddLink(prismatic_wall_14_1);
  }
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
  gl_window.SetCamera(ChVector<>(0, 0, 3 * Lz0), ChVector<>(0, 0, 0), ChVector<>(0, 1, 0), max_diameter, max_diameter);
  gl_window.SetRenderMode(opengl::SOLID);
#endif

  // Begin simulation

  bool settled = false;
  int data_out_frame = 0;
  int visual_out_frame = 0;
  real3 force1, force2, force3, force4, force5;
  real3 force6, force7, force8, force9, force10;
  real3 force11, force12, force13, force14;

  while (my_system->GetChTime() < simulation_time) {
    if (dense == true)
    	for (i = 0; i < numParticleTypes; i++) mat[i]->SetFriction(0.01f);

    if (my_system->GetChTime() > settling_time && settled == false) {
        wall_3->SetBodyFixed(false);
        wall_4->SetBodyFixed(false);
        wall_5->SetBodyFixed(false);
        wall_6->SetBodyFixed(false);
        if (cylinder == true) {
            wall_7->SetBodyFixed(false);
            wall_8->SetBodyFixed(false);
            wall_9->SetBodyFixed(false);
            wall_10->SetBodyFixed(false);
            wall_11->SetBodyFixed(false);
            wall_12->SetBodyFixed(false);
            wall_13->SetBodyFixed(false);
            wall_14->SetBodyFixed(false);
        }
    	settled = true;
    }

    Lz = wall_2->GetPos().z - wall_1->GetPos().z - thickness;
    Lx = wall_4->GetPos().x - wall_3->GetPos().x - thickness;
    Ly = wall_6->GetPos().y - wall_5->GetPos().y - thickness;

    if (settled == true) {
    	wall_2->Empty_forces_accumulators();
    	wall_3->Empty_forces_accumulators();
    	wall_4->Empty_forces_accumulators();
    	wall_5->Empty_forces_accumulators();
    	wall_6->Empty_forces_accumulators();
    	if (cylinder == true) {
    		wall_7->Empty_forces_accumulators();
    		wall_8->Empty_forces_accumulators();
    		wall_9->Empty_forces_accumulators();
    		wall_10->Empty_forces_accumulators();
    		wall_11->Empty_forces_accumulators();
    		wall_12->Empty_forces_accumulators();
    		wall_13->Empty_forces_accumulators();
    		wall_14->Empty_forces_accumulators();

    		diam = (Lx + Ly) / 2.0;

    		wall_2->Accumulate_force(ChVector<>(0, 0, -sigma_a*CH_C_PI*diam*diam/4.0),wall_2->GetPos(),false);
    		wall_3->Accumulate_force(ChVector<>(sigma_b*Lz*CH_C_PI*diam/12.0, 0, 0),wall_3->GetPos(),false);
    		wall_4->Accumulate_force(ChVector<>(-sigma_b*Lz*CH_C_PI*diam/12.0, 0, 0),wall_4->GetPos(),false);
    		wall_5->Accumulate_force(ChVector<>(0, sigma_b*Lz*CH_C_PI*diam/12.0, 0),wall_5->GetPos(),false);
    		wall_6->Accumulate_force(ChVector<>(0, -sigma_b*Lz*CH_C_PI*diam/12.0, 0),wall_6->GetPos(),false);
    		wall_7->Accumulate_force(ChVector<>(sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/6),
    				sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/6), 0),wall_7->GetPos(),false);
    		wall_8->Accumulate_force(ChVector<>(-sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/6),
    				-sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/6), 0),wall_8->GetPos(),false);
    		wall_9->Accumulate_force(ChVector<>(sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/3),
    				sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/3), 0),wall_9->GetPos(),false);
    		wall_10->Accumulate_force(ChVector<>(-sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/3),
    				-sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/3), 0),wall_10->GetPos(),false);
    		wall_11->Accumulate_force(ChVector<>(-sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/6),
    				sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/6), 0),wall_11->GetPos(),false);
    		wall_12->Accumulate_force(ChVector<>(sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/6),
    				-sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/6), 0),wall_12->GetPos(),false);
    		wall_13->Accumulate_force(ChVector<>(-sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/3),
    				sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/3), 0),wall_13->GetPos(),false);
    		wall_14->Accumulate_force(ChVector<>(sigma_b*Lz*CH_C_PI*diam/12.0*sin(CH_C_PI/3),
    				-sigma_b*Lz*CH_C_PI*diam/12.0*cos(CH_C_PI/3), 0),wall_14->GetPos(),false);
    	} else {
    		wall_2->Accumulate_force(ChVector<>(0, 0, -sigma_a*Lx*Ly),wall_2->GetPos(),false);
    		wall_3->Accumulate_force(ChVector<>(sigma_b*Ly*Lz, 0, 0),wall_3->GetPos(),false);
    		wall_4->Accumulate_force(ChVector<>(-sigma_b*Ly*Lz, 0, 0),wall_4->GetPos(),false);
    		wall_5->Accumulate_force(ChVector<>(0, sigma_c*Lx*Lz, 0),wall_5->GetPos(),false);
    		wall_6->Accumulate_force(ChVector<>(0, -sigma_c*Lx*Lz, 0),wall_6->GetPos(),false);
    	}
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

    //  Output to files and screen

    if (my_system->GetChTime() >= data_out_frame * data_out_step) {
#ifndef USE_DEM
      my_system->CalculateContactForces();
#endif
      force1 = my_system->GetBodyContactForce(0);
      force2 = my_system->GetBodyContactForce(1);
      force3 = my_system->GetBodyContactForce(2);
      force4 = my_system->GetBodyContactForce(3);
      force5 = my_system->GetBodyContactForce(4);
      force6 = my_system->GetBodyContactForce(5);
      if (cylinder == true) {
          force7 = my_system->GetBodyContactForce(6);
          force8 = my_system->GetBodyContactForce(7);
          force9 = my_system->GetBodyContactForce(8);
          force10 = my_system->GetBodyContactForce(9);
          force11 = my_system->GetBodyContactForce(10);
          force12 = my_system->GetBodyContactForce(11);
          force13 = my_system->GetBodyContactForce(12);
          force14 = my_system->GetBodyContactForce(13);
      }

      forceStream << my_system->GetChTime() << "\t" << Lx << "\t" << Ly << "\t" << Lz << "\t";
      forceStream << force3.x << "\t" << force4.x << "\t" << force5.y << "\t" << force6.y << "\t";
      if (cylinder == true) {
          forceStream << sqrt(force7.x*force7.x+force7.y*force7.y) << "\t";
          forceStream << sqrt(force8.x*force8.x+force8.y*force8.y) << "\t";
          forceStream << sqrt(force9.x*force9.x+force9.y*force9.y) << "\t";
          forceStream << sqrt(force10.x*force10.x+force10.y*force10.y) << "\t";
          forceStream << sqrt(force11.x*force11.x+force11.y*force11.y) << "\t";
          forceStream << sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
          forceStream << sqrt(force13.x*force13.x+force13.y*force13.y) << "\t";
          forceStream << sqrt(force14.x*force14.x+force14.y*force14.y) << "\t";
      }
      forceStream << force1.z << "\t" << force2.z << "\n";

      cout << my_system->GetChTime() << "\t" << Lx << "\t" << Ly << "\t" << Lz << "\t";
      cout << force3.x << "\t" << force4.x << "\t" << force5.y << "\t" << force6.y << "\t";
      if (cylinder == true) {
          cout << sqrt(force7.x*force7.x+force7.y*force7.y) << "\t";
          cout << sqrt(force8.x*force8.x+force8.y*force8.y) << "\t";
          cout << sqrt(force9.x*force9.x+force9.y*force9.y) << "\t";
          cout << sqrt(force10.x*force10.x+force10.y*force10.y) << "\t";
          cout << sqrt(force11.x*force11.x+force11.y*force11.y) << "\t";
          cout << sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
          cout << sqrt(force13.x*force13.x+force13.y*force13.y) << "\t";
          cout << sqrt(force14.x*force14.x+force14.y*force14.y) << "\t";
      }
      cout << force1.z << "\t" << force2.z << "\n";

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
