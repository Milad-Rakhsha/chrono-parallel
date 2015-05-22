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
// Soft-sphere (DEM/DEM-P) or hard-sphere (DVI/DEM-C) triaxial test simulation.
// Uses checkpoint file from dem_specimen granular material specimen generator.
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

using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::flush;
using std::endl;

// -----------------------------------------------------------------------------
// Specimen definition
// -----------------------------------------------------------------------------

bool fix_constrained_walls = true; // for debugging
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
SpecimenGeom geom = STANDARD_TRIAXIAL;

// Compressive stress (Pa)
// (negative means that stress is not prescribed)
double sigma_a = -1;
double sigma_b = 3.1e3;
double sigma_c = 3.1e3;

// Compressive strain-rate (1/s)
// (prescribed whenever stress is not prescribed)
double epsdot_a = 0.1;
double epsdot_b = -1;
double epsdot_c = -1;

// Comment the following line to use DVI contact
#define USE_DEM

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 1;

// Perform dynamic tuning of number of threads?
bool thread_tuning = false;

// Solver settings
#ifdef USE_DEM
double time_step = 1e-5;
double tolerance = 0.1;
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
double simulation_time = 1.0;
#else
double simulation_time = 1.0;
#endif

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
const std::string specimen_ckpnt_file = out_dir + "/specimen.dat";

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
  const std::string title = "soft-sphere (DEM/DEM-P) triaxial test simulation";
  ChBody::ContactMethod contact_method = ChBody::DEM;
  ChSystemParallelDEM* my_system = new ChSystemParallelDEM();
#else
  cout << "Create DVI system" << endl;
  const std::string title = "hard-sphere (DVI/DEM-C) triaxial test simulation";
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

  // Define all walls

  ChSharedPtr<ChBody> wall_1;
  ChSharedPtr<ChBody> wall_2;
  ChSharedPtr<ChBody> wall_3;
  ChSharedPtr<ChBody> wall_4;
  ChSharedPtr<ChBody> wall_5;
  ChSharedPtr<ChBody> wall_6;
  ChSharedPtr<ChBody> wall_7;
  ChSharedPtr<ChBody> wall_8;
  ChSharedPtr<ChBody> wall_9;
  ChSharedPtr<ChBody> wall_10;
  ChSharedPtr<ChBody> wall_11;
  ChSharedPtr<ChBody> wall_12;
  ChSharedPtr<ChBody> wall_13;
  ChSharedPtr<ChBody> wall_14;

  // Create bodies from checkpoint file

  cout << "Read checkpoint data from " << specimen_ckpnt_file;
  utils::ReadCheckpoint(my_system, specimen_ckpnt_file);
  cout << "  done.  Read " << my_system->Get_bodylist()->size() << " bodies." << endl;

  // Grab handles to bodies (must increase ref counts)

  wall_1 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(0));
  wall_2 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(1));
  wall_3 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(2));
  wall_4 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(3));
  wall_5 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(4));
  wall_6 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(5));
  my_system->Get_bodylist()->at(0)->AddRef();
  my_system->Get_bodylist()->at(1)->AddRef();
  my_system->Get_bodylist()->at(2)->AddRef();
  my_system->Get_bodylist()->at(3)->AddRef();
  my_system->Get_bodylist()->at(4)->AddRef();
  my_system->Get_bodylist()->at(5)->AddRef();

  if (cylinder == true) {
	  wall_7 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(6));
	  wall_8 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(7));
  	  wall_9 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(8));
  	  wall_10 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(9));
  	  wall_11 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(10));
  	  wall_12 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(11));
  	  wall_13 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(12));
  	  wall_14 = ChSharedPtr<ChBody>(my_system->Get_bodylist()->at(13));
  	  my_system->Get_bodylist()->at(6)->AddRef();
  	  my_system->Get_bodylist()->at(7)->AddRef();
  	  my_system->Get_bodylist()->at(8)->AddRef();
  	  my_system->Get_bodylist()->at(9)->AddRef();
  	  my_system->Get_bodylist()->at(10)->AddRef();
  	  my_system->Get_bodylist()->at(11)->AddRef();
  	  my_system->Get_bodylist()->at(12)->AddRef();
  	  my_system->Get_bodylist()->at(13)->AddRef();
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

  // Create prismatic (translational) joint between the plate and the ground.
  // The translational axis of a prismatic joint is along the Z axis of the
  // specified joint coordinate system.  Here, we apply the 'z2y' rotation to
  // align it with the Y axis of the global reference frame.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_2_1(new ChLinkLockPrismatic);
  prismatic_wall_2_1->SetName("prismatic_wall_2_1");
  prismatic_wall_2_1->Initialize(wall_2, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), QUNIT));

  // Create prismatic (translational) joints between each of the four walls and the ground.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_3_1(new ChLinkLockPrismatic);
  prismatic_wall_3_1->SetName("prismatic_wall_3_1");
  prismatic_wall_3_1->Initialize(wall_3, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_4_1(new ChLinkLockPrismatic);
  prismatic_wall_4_1->SetName("prismatic_wall_4_1");
  prismatic_wall_4_1->Initialize(wall_4, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2x));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_5_1(new ChLinkLockPrismatic);
  prismatic_wall_5_1->SetName("prismatic_wall_5_1");
  prismatic_wall_5_1->Initialize(wall_5, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_6_1(new ChLinkLockPrismatic);
  prismatic_wall_6_1->SetName("prismatic_wall_6_1");
  prismatic_wall_6_1->Initialize(wall_6, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2y));

  ChQuaternion<> z2wall_7_8 = z30%z2x;
  ChQuaternion<> z2wall_9_10 = z60%z2x;
  ChQuaternion<> z2wall_11_12 = z30%z2y;
  ChQuaternion<> z2wall_13_14 = z60%z2y;

  // For cylinder only

  // Create prismatic (translational) joints between an additional eight walls and the ground.

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_7_1(new ChLinkLockPrismatic);
  prismatic_wall_7_1->SetName("prismatic_wall_7_1");
  prismatic_wall_7_1->Initialize(wall_7, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_7_8));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_8_1(new ChLinkLockPrismatic);
  prismatic_wall_8_1->SetName("prismatic_wall_8_1");
  prismatic_wall_8_1->Initialize(wall_8, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_7_8));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_9_1(new ChLinkLockPrismatic);
  prismatic_wall_9_1->SetName("prismatic_wall_9_1");
  prismatic_wall_9_1->Initialize(wall_9, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_9_10));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_10_1(new ChLinkLockPrismatic);
  prismatic_wall_10_1->SetName("prismatic_wall_10_1");
  prismatic_wall_10_1->Initialize(wall_10, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_9_10));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_11_1(new ChLinkLockPrismatic);
  prismatic_wall_11_1->SetName("prismatic_wall_11_1");
  prismatic_wall_11_1->Initialize(wall_11, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_11_12));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_12_1(new ChLinkLockPrismatic);
  prismatic_wall_12_1->SetName("prismatic_wall_12_1");
  prismatic_wall_12_1->Initialize(wall_12, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_11_12));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_13_1(new ChLinkLockPrismatic);
  prismatic_wall_13_1->SetName("prismatic_wall_13_1");
  prismatic_wall_13_1->Initialize(wall_13, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_13_14));

  ChSharedPtr<ChLinkLockPrismatic> prismatic_wall_14_1(new ChLinkLockPrismatic);
  prismatic_wall_14_1->SetName("prismatic_wall_14_1");
  prismatic_wall_14_1->Initialize(wall_14, wall_1, ChCoordsys<>(ChVector<>(0, 0, 0), z2wall_13_14));

  // Add prismatic joints to system

  if (cylinder == true) {
  	if (sigma_a < 0 && fix_constrained_walls) {
  		wall_2->SetBodyFixed(true);
  	} else {
  		my_system->AddLink(prismatic_wall_2_1);
  	}
  	if (sigma_b < 0 && fix_constrained_walls) {
  		wall_3->SetBodyFixed(true);
  		wall_4->SetBodyFixed(true);
  		wall_5->SetBodyFixed(true);
  		wall_6->SetBodyFixed(true);
  		wall_7->SetBodyFixed(true);
  		wall_8->SetBodyFixed(true);
  		wall_9->SetBodyFixed(true);
  		wall_10->SetBodyFixed(true);
  		wall_11->SetBodyFixed(true);
  		wall_12->SetBodyFixed(true);
  		wall_13->SetBodyFixed(true);
  		wall_14->SetBodyFixed(true);
  	} else {
        my_system->AddLink(prismatic_wall_3_1);
        my_system->AddLink(prismatic_wall_4_1);
        my_system->AddLink(prismatic_wall_5_1);
        my_system->AddLink(prismatic_wall_6_1);
        my_system->AddLink(prismatic_wall_7_1);
        my_system->AddLink(prismatic_wall_8_1);
        my_system->AddLink(prismatic_wall_9_1);
        my_system->AddLink(prismatic_wall_10_1);
        my_system->AddLink(prismatic_wall_11_1);
        my_system->AddLink(prismatic_wall_12_1);
        my_system->AddLink(prismatic_wall_13_1);
        my_system->AddLink(prismatic_wall_14_1);
  	}
  } else {
	  	if (sigma_a < 0 && fix_constrained_walls) {
	  		wall_2->SetBodyFixed(true);
	  	} else {
	  		my_system->AddLink(prismatic_wall_2_1);
	  	}
	  	if (sigma_b < 0 && fix_constrained_walls) {
	  		wall_3->SetBodyFixed(true);
	  		wall_4->SetBodyFixed(true);
	  	} else {
	      my_system->AddLink(prismatic_wall_3_1);
	      my_system->AddLink(prismatic_wall_4_1);
	  	}
	  	if (sigma_c < 0 && fix_constrained_walls) {
	  		wall_5->SetBodyFixed(true);
	  		wall_6->SetBodyFixed(true);
	  	} else {
	      my_system->AddLink(prismatic_wall_5_1);
	      my_system->AddLink(prismatic_wall_6_1);
	  	}
  }

  // Setup output

  ChStreamOutAsciiFile stressStream(stress_file.c_str());
  ChStreamOutAsciiFile forceStream(force_file.c_str());
  ChStreamOutAsciiFile statsStream(stats_file.c_str());
  stressStream.SetNumFormat("%16.4e");
  forceStream.SetNumFormat("%16.4e");

  if (cylinder == true) {
      stressStream << "time" << "\t";
      stressStream << "strain_a" << "\t" << "strain_b" << "\t";
      stressStream << "stress_a" << "\t" << "stress_b" << "\n";
  } else {
      stressStream << "time" << "\t";
      stressStream << "strain_a" << "\t" << "strain_b" << "\t" << "strain_c" << "\t";
      stressStream << "stress_a" << "\t" << "stress_b" << "\t" << "stress_c" << "\n";
  }

  // Create the OpenGL visualization window

#ifdef CHRONO_PARALLEL_HAS_OPENGL
  opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
  gl_window.Initialize(800, 600, title.c_str(), my_system);
  gl_window.SetCamera(ChVector<>(10 * Lx0, 0, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), max_diameter, max_diameter);
  gl_window.SetRenderMode(opengl::WIREFRAME);
#endif

  // Begin simulation

  int data_out_frame = 0;
  int visual_out_frame = 0;
  real3 force1, force2, force3, force4, force5;
  real3 force6, force7, force8, force9, force10;
  real3 force11, force12, force13, force14;
  double Lx, Ly, Lz;
  double L7, L8, L9, L10, L11, L12, L13, L14, diam;

  Lz0 = wall_2->GetPos().z - wall_1->GetPos().z - thickness;
  Lx0 = wall_4->GetPos().x - wall_3->GetPos().x - thickness;
  Ly0 = wall_6->GetPos().y - wall_5->GetPos().y - thickness;

  while (my_system->GetChTime() < simulation_time) {

	Lz = wall_2->GetPos().z - wall_1->GetPos().z - thickness;
	Lx = wall_4->GetPos().x - wall_3->GetPos().x - thickness;
	Ly = wall_6->GetPos().y - wall_5->GetPos().y - thickness;

    if (cylinder == true) {
        L7 = sqrt(wall_7->GetPos().x*wall_7->GetPos().x+wall_7->GetPos().y*wall_7->GetPos().y) - thickness / 2;
        L8 = sqrt(wall_8->GetPos().x*wall_8->GetPos().x+wall_8->GetPos().y*wall_8->GetPos().y) - thickness / 2;
        L9 = sqrt(wall_9->GetPos().x*wall_9->GetPos().x+wall_9->GetPos().y*wall_9->GetPos().y) - thickness / 2;
        L10 = sqrt(wall_10->GetPos().x*wall_10->GetPos().x+wall_10->GetPos().y*wall_10->GetPos().y) - thickness / 2;
        L11 = sqrt(wall_11->GetPos().x*wall_11->GetPos().x+wall_11->GetPos().y*wall_11->GetPos().y) - thickness / 2;
        L12 = sqrt(wall_12->GetPos().x*wall_12->GetPos().x+wall_12->GetPos().y*wall_12->GetPos().y) - thickness / 2;
        L13 = sqrt(wall_13->GetPos().x*wall_13->GetPos().x+wall_13->GetPos().y*wall_13->GetPos().y) - thickness / 2;
        L14 = sqrt(wall_14->GetPos().x*wall_14->GetPos().x+wall_14->GetPos().y*wall_14->GetPos().y) - thickness / 2;
        diam = (L7 + L8 + L9 + L10 + L11 + L12 + L13 + L14) / 4;
    }

    // Apply prescribed compressive stresses or strain-rates (strain-rate is prescribed whenever stress is negative)

    if (cylinder == true) {
    	if (sigma_a < 0) {
    	    wall_2->SetPos_dt(ChVector<>(0, 0, -epsdot_a*Lz));
    	    if (fix_constrained_walls)
    	    	wall_2->SetPos(ChVector<>(wall_2->GetPos().x, wall_2->GetPos().y, wall_2->GetPos().z-epsdot_a*Lz*time_step));
    	} else {
    		wall_2->Empty_forces_accumulators();
        	wall_2->Accumulate_force(ChVector<>(0, 0, -sigma_a*CH_C_PI*diam*diam/4.0),wall_2->GetPos(),false);
    	}
    	if (sigma_b < 0) {
    		wall_3->SetPos_dt(ChVector<>(epsdot_b*Lx, 0, 0));
    		wall_4->SetPos_dt(ChVector<>(-epsdot_b*Lx, 0, 0));
    		wall_5->SetPos_dt(ChVector<>(0, epsdot_b*Ly, 0));
    		wall_6->SetPos_dt(ChVector<>(0, -epsdot_b*Ly, 0));
    		wall_7->SetPos_dt(ChVector<>(epsdot_b*(L7+L8)*cos(CH_C_PI/6),
    				epsdot_b*(L7+L8)*sin(CH_C_PI/6), 0));
    		wall_8->SetPos_dt(ChVector<>(-epsdot_b*(L7+L8)*cos(CH_C_PI/6),
    				-epsdot_b*(L7+L8)*sin(CH_C_PI/6), 0));
    		wall_9->SetPos_dt(ChVector<>(epsdot_b*(L9+L10)*cos(CH_C_PI/3),
    				epsdot_b*(L9+L10)*sin(CH_C_PI/3), 0));
    		wall_10->SetPos_dt(ChVector<>(-epsdot_b*(L9+L10)*cos(CH_C_PI/3),
    				-epsdot_b*(L9+L10)*sin(CH_C_PI/3), 0));
    		wall_11->SetPos_dt(ChVector<>(-epsdot_b*(L11+L12)*sin(CH_C_PI/6),
    				epsdot_b*(L11+L12)*cos(CH_C_PI/6), 0));
    		wall_12->SetPos_dt(ChVector<>(epsdot_b*(L11+L12)*sin(CH_C_PI/6),
    				-epsdot_b*(L11+L12)*cos(CH_C_PI/6), 0));
    		wall_13->SetPos_dt(ChVector<>(-epsdot_b*(L13+L14)*sin(CH_C_PI/3),
    				epsdot_b*(L13+L14)*cos(CH_C_PI/3), 0));
    		wall_14->SetPos_dt(ChVector<>(epsdot_b*(L13+L14)*sin(CH_C_PI/3),
    				-epsdot_b*(L13+L14)*cos(CH_C_PI/3), 0));
    		if (fix_constrained_walls) {
    			wall_3->SetPos(ChVector<>(wall_3->GetPos().x+epsdot_b*Lx*time_step, wall_3->GetPos().y, wall_3->GetPos().z));
    			wall_4->SetPos(ChVector<>(wall_4->GetPos().x-epsdot_b*Lx*time_step, wall_4->GetPos().y, wall_4->GetPos().z));
    			wall_5->SetPos(ChVector<>(wall_5->GetPos().x, wall_5->GetPos().y+epsdot_b*Ly*time_step, wall_5->GetPos().z));
    			wall_6->SetPos(ChVector<>(wall_6->GetPos().x, wall_6->GetPos().y-epsdot_b*Ly*time_step, wall_6->GetPos().z));
    			wall_7->SetPos(ChVector<>(wall_7->GetPos().x+epsdot_b*(L7+L8)*cos(CH_C_PI/6)*time_step,
    					wall_7->GetPos().y+epsdot_b*(L7+L8)*sin(CH_C_PI/6)*time_step, wall_7->GetPos().z));
    			wall_8->SetPos(ChVector<>(wall_8->GetPos().x-epsdot_b*(L7+L8)*cos(CH_C_PI/6)*time_step,
    					wall_8->GetPos().y-epsdot_b*(L7+L8)*sin(CH_C_PI/6)*time_step, wall_8->GetPos().z));
    			wall_9->SetPos(ChVector<>(wall_9->GetPos().x+epsdot_b*(L9+L10)*cos(CH_C_PI/3)*time_step,
    					wall_9->GetPos().y+epsdot_b*(L9+L10)*sin(CH_C_PI/3)*time_step, wall_9->GetPos().z));
    			wall_10->SetPos(ChVector<>(wall_10->GetPos().x-epsdot_b*(L9+L10)*cos(CH_C_PI/3)*time_step,
    					wall_10->GetPos().y-epsdot_b*(L9+L10)*sin(CH_C_PI/3)*time_step, wall_10->GetPos().z));
    			wall_11->SetPos(ChVector<>(wall_11->GetPos().x-epsdot_b*(L11+L12)*sin(CH_C_PI/6)*time_step,
    					wall_11->GetPos().y+epsdot_b*(L11+L12)*cos(CH_C_PI/6)*time_step, wall_11->GetPos().z));
    			wall_12->SetPos(ChVector<>(wall_12->GetPos().x+epsdot_b*(L11+L12)*sin(CH_C_PI/6)*time_step,
    					wall_12->GetPos().y-epsdot_b*(L11+L12)*cos(CH_C_PI/6)*time_step, wall_12->GetPos().z));
    			wall_13->SetPos(ChVector<>(wall_13->GetPos().x-epsdot_b*(L13+L14)*sin(CH_C_PI/3)*time_step,
    					wall_13->GetPos().y+epsdot_b*(L13+L14)*cos(CH_C_PI/3)*time_step, wall_13->GetPos().z));
    			wall_14->SetPos(ChVector<>(wall_14->GetPos().x+epsdot_b*(L13+L14)*sin(CH_C_PI/3)*time_step,
    					wall_14->GetPos().y-epsdot_b*(L13+L14)*cos(CH_C_PI/3)*time_step, wall_14->GetPos().z));
    		}
    	} else {
    	   	wall_3->Empty_forces_accumulators();
    	   	wall_4->Empty_forces_accumulators();
        	wall_5->Empty_forces_accumulators();
        	wall_6->Empty_forces_accumulators();
    		wall_7->Empty_forces_accumulators();
    		wall_8->Empty_forces_accumulators();
    		wall_9->Empty_forces_accumulators();
    		wall_10->Empty_forces_accumulators();
    		wall_11->Empty_forces_accumulators();
    		wall_12->Empty_forces_accumulators();
    		wall_13->Empty_forces_accumulators();
    		wall_14->Empty_forces_accumulators();
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
    	}
    } else {
    	if (sigma_a < 0) {
    	    wall_2->SetPos_dt(ChVector<>(0, 0, -epsdot_a*Lz));
    	    if (fix_constrained_walls)
    	    	wall_2->SetPos(ChVector<>(wall_2->GetPos().x, wall_2->GetPos().y, wall_2->GetPos().z-epsdot_a*Lz*time_step));
    	} else {
    	    wall_2->Empty_forces_accumulators();
    		wall_2->Accumulate_force(ChVector<>(0, 0, -sigma_a*Lx*Ly),wall_2->GetPos(),false);
    	}
    	if (sigma_b < 0) {
    		wall_3->SetPos_dt(ChVector<>(epsdot_b*Lx, 0, 0));
    		wall_4->SetPos_dt(ChVector<>(-epsdot_b*Lx, 0, 0));
    		if (fix_constrained_walls) {
    			wall_3->SetPos(ChVector<>(wall_3->GetPos().x+epsdot_b*Lx*time_step, wall_3->GetPos().y, wall_3->GetPos().z));
    			wall_4->SetPos(ChVector<>(wall_4->GetPos().x-epsdot_b*Lx*time_step, wall_4->GetPos().y, wall_4->GetPos().z));
    		}
    	} else {
    	   	wall_3->Empty_forces_accumulators();
    	   	wall_4->Empty_forces_accumulators();
    		wall_3->Accumulate_force(ChVector<>(sigma_b*Ly*Lz, 0, 0),wall_3->GetPos(),false);
    		wall_4->Accumulate_force(ChVector<>(-sigma_b*Ly*Lz, 0, 0),wall_4->GetPos(),false);
    	}
    	if (sigma_c < 0) {
    		wall_5->SetPos_dt(ChVector<>(0, epsdot_c*Ly, 0));
    		wall_6->SetPos_dt(ChVector<>(0, -epsdot_c*Ly, 0));
    		if (fix_constrained_walls) {
    			wall_5->SetPos(ChVector<>(wall_5->GetPos().x, wall_5->GetPos().y+epsdot_c*Ly*time_step, wall_5->GetPos().z));
    			wall_6->SetPos(ChVector<>(wall_6->GetPos().x, wall_6->GetPos().y-epsdot_c*Ly*time_step, wall_6->GetPos().z));
    		}
    	} else {
    	   	wall_5->Empty_forces_accumulators();
    	   	wall_6->Empty_forces_accumulators();
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

      forceStream << my_system->GetChTime() << "\t";
      forceStream << Lx << "\t" << Ly << "\t";
      if (cylinder == true) {
    	  forceStream << L7+L8 << "\t" << L9+L10 << "\t" << L11+L12 << "\t" << L13+L14 << "\t";
      }
      forceStream << Lz << "\n";
      forceStream << "\t" << force4.x << "\t" << force6.y << "\t";
      if (cylinder == true) {
          forceStream << sqrt(force8.x*force8.x+force8.y*force8.y) << "\t";
          forceStream << sqrt(force10.x*force10.x+force10.y*force10.y) << "\t";
          forceStream << sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
          forceStream << sqrt(force14.x*force14.x+force14.y*force14.y) << "\t";
      }
      forceStream << force2.z << "\n";
      forceStream << "\t" << force4.x+force3.x << "\t" << force6.y+force5.y << "\t";
      if (cylinder == true) {
          forceStream << sqrt(force8.x*force8.x+force8.y*force8.y)-sqrt(force7.x*force7.x+force7.y*force7.y) << "\t";
          forceStream << sqrt(force10.x*force10.x+force10.y*force10.y)-sqrt(force9.x*force9.x+force9.y*force9.y) << "\t";
          forceStream << sqrt(force12.x*force12.x+force12.y*force12.y)-sqrt(force11.x*force11.x+force11.y*force11.y) << "\t";
          forceStream << sqrt(force14.x*force14.x+force14.y*force14.y)-sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
      }
      forceStream << force2.z+force1.z << "\n";

      cout << my_system->GetChTime() << "\t";
	  cout << Lx << "\t" << Ly << "\t";
      if (cylinder == true) {
    	  cout << L7+L8 << "\t" << L9+L10 << "\t" << L11+L12 << "\t" << L13+L14 << "\t";
      }
      cout << Lz << "\n";
      cout << "\t" << force4.x << "\t" << force6.y << "\t";
      if (cylinder == true) {
          cout << sqrt(force8.x*force8.x+force8.y*force8.y) << "\t";
          cout << sqrt(force10.x*force10.x+force10.y*force10.y) << "\t";
          cout << sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
          cout << sqrt(force14.x*force14.x+force14.y*force14.y) << "\t";
      }
      cout << force2.z << "\n";
      cout << "\t" << force4.x+force3.x << "\t" << force6.y+force5.y << "\t";
      if (cylinder == true) {
          cout << sqrt(force8.x*force8.x+force8.y*force8.y)-sqrt(force7.x*force7.x+force7.y*force7.y) << "\t";
          cout << sqrt(force10.x*force10.x+force10.y*force10.y)-sqrt(force9.x*force9.x+force9.y*force9.y) << "\t";
          cout << sqrt(force12.x*force12.x+force12.y*force12.y)-sqrt(force11.x*force11.x+force11.y*force11.y) << "\t";
          cout << sqrt(force14.x*force14.x+force14.y*force14.y)-sqrt(force12.x*force12.x+force12.y*force12.y) << "\t";
      }
      cout << force2.z+force1.z << "\n";

      stressStream << my_system->GetChTime() << "\t";
      stressStream << (Lz0-Lz)/Lz0 << "\t";
      if (cylinder == true) {
          stressStream << ((Lx0-Lx)/Lx0+(Ly0-Ly)/Ly0)/2.0 << "\t";
          stressStream << force2.z/(CH_C_PI*diam*diam/4.0) << "\t";
          stressStream << (force4.x-force3.x+force6.y-force5.y)/(Lz*CH_C_PI*diam/3.0) << "\n";
      } else {
    	  stressStream << (Lx0-Lx)/Lx0 << "\t" << (Ly0-Ly)/Ly0 << "\t";
    	  stressStream << force2.z/(Lx*Ly) << "\t";
    	  stressStream << (force4.x-force3.x)/(2.0*Ly*Lz) << "\t" << (force6.y-force5.y)/(2.0*Lx*Lz) << "\n";
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

  // Create a checkpoint file for the prepared granular material specimen

  cout << "Write checkpoint data to " << flush;
  utils::WriteCheckpoint(my_system, specimen_ckpnt_file);
  cout << my_system->Get_bodylist()->size() << " bodies" << endl;

  return 0;
}
