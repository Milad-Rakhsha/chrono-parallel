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
// Implementation of an iterative Steepest Descent solver.
// =============================================================================

#ifndef CHSOLVERSD_H
#define CHSOLVERSD_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverSD : public ChSolverParallel {
 public:
  ChSolverSD() : ChSolverParallel() {}
  ~ChSolverSD() {}

  void Solve() {
    if (data_container->num_constraints == 0) {
      return;
    }
    data_container->system_timer.start("ChSolverParallel_Solve");
    data_container->measures.solver.total_iteration += SolveSD(
        max_iteration, data_container->num_constraints, data_container->host_data.R, data_container->host_data.gamma);
    data_container->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using the steepest descent method
  uint SolveSD(const uint max_iter,            // Maximum number of iterations
               const uint size,                // Number of unknowns
               blaze::DynamicVector<real>& b,  // Rhs vector
               blaze::DynamicVector<real>& x   // The vector of unknowns
               );

  blaze::DynamicVector<real> r, temp;
};
}

#endif
