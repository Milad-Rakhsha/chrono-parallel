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
// Implementation of an iterative Conjugate Gradient solver.
// =============================================================================

#ifndef CHSOLVERPGS_H
#define CHSOLVERPGS_H

#include "chrono_parallel/solver/ChSolverParallel.h"

namespace chrono {

class CH_PARALLEL_API ChSolverPGS : public ChSolverParallel {
 public:
  ChSolverPGS() : ChSolverParallel() {}
  ~ChSolverPGS() {}

  void Solve() {
    if (data_container->num_constraints == 0) {
      return;
    }
    data_container->system_timer.start("ChSolverParallel_Solve");
    data_container->measures.solver.total_iteration += SolvePGS(
        max_iteration, data_container->num_constraints, data_container->host_data.R, data_container->host_data.gamma);
    data_container->system_timer.stop("ChSolverParallel_Solve");
  }

  // Solve using an iterative projected gradient method
  uint SolvePGS(const uint max_iter,            // Maximum number of iterations
                const uint size,                // Number of unknowns
                blaze::DynamicVector<real>& b,  // Rhs vector
                blaze::DynamicVector<real>& x   // The vector of unknowns
                );

  blaze::DynamicVector<real> diagonal;
};
}

#endif
