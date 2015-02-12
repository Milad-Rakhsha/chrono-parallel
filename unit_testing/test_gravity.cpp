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
// Authors: Hammad Mazhar
// =============================================================================
//
// ChronoParallel unit test for MPR collision detection
// =============================================================================

//not used but prevents compilation errors with cuda 7 RC
#include <thrust/transform.h>


#include <stdio.h>
#include <vector>
#include <cmath>

#include "collision/ChCCollisionModel.h"
#include "core/ChMathematics.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "unit_testing.h"

using namespace chrono;
using namespace chrono::collision;
int main(
      int argc,
      char* argv[]) {
   double time_step = .00001;
   ChVector<> gravity = ChVector<>(0, -9.80665, 0);
   ChSystemParallelDVI msystem;
   msystem.Set_G_acc(gravity);
   msystem.SetStep(time_step);

   ChSharedBodyPtr ball(new ChBody(new ChCollisionModelParallel));
   double mass = 1;
   ChVector<> pos = ChVector<>(0, 0, 0);
   ChVector<> vel = ChVector<>(2, 2, 0);
   ball->SetMass(mass);
   ball->SetPos(pos);
   ball->SetPos_dt(vel);
   msystem.AddBody(ball);

   real t = .01;
   real3 a = ToReal3(gravity);
   real3 p_final = ToReal3(vel) * t + .5 * a * t * t;

   double time = 0;
   for (int i = 0; i < 1000; i++) {
      msystem.DoStepDynamics(time_step);
      time = time + time_step;
   }
   WeakEqual(ToReal3(ball->GetPos()), p_final, 1e-6);
   return 0;
}

