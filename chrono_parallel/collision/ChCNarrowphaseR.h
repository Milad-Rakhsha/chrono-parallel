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
// Authors: Radu Serban
// =============================================================================
//
// Header file for ChCNarrowphaseR.
// This narrow-phase collision detection relies on specialized functions for
// each pair of collision shapes. Only a subset of collision shapes and of
// pair-wise interactions are currently supported:
//
//          |  sphere   box   rbox   capsule   cylinder   rcyl   trimesh
// ---------+----------------------------------------------------------
// sphere   |    Y       Y      Y       Y         Y        Y        Y
// box      |           WIP     N       Y         N        N        N
// rbox     |                   N       N         N        N        N
// capsule  |                           Y         N        N        N
// cylinder |                                     N        N        N
// rcyl     |                                              N        N
// trimesh  |                                                       N
//
// Note that some pairs may return more than one contact (e.g., box-box).
//
// =============================================================================

#ifndef CHC_NARROWPHASE_R_H
#define CHC_NARROWPHASE_R_H

#include "chrono_parallel/collision/ChCNarrowphase.h"

namespace chrono {
namespace collision {

static const real edge_radius = 0.1;

// Primitive collision functions
bool sphere_sphere(const real3& pos1, const real& radius1,
                   const real3& pos2, const real& radius2,
                   real3& norm, real& depth,
                   real3& pt1, real3& pt2,
                   real& eff_radius);

bool capsule_sphere(const real3& pos1, const real4& rot1, const real& radius1, const real& hlen1,
                    const real3& pos2, const real& radius2,
                    real3& norm, real& depth,
                    real3& pt1, real3& pt2,
                    real& eff_radius);

bool cylinder_sphere(const real3& pos1, const real4& rot1, const real& radius1, const real& hlen1,
                     const real3& pos2, const real& radius2,
                     real3& norm, real& depth,
                     real3& pt1, real3& pt2,
                     real& eff_radius);

bool roundedcyl_sphere(const real3& pos1, const real4& rot1, const real& radius1, const real& hlen1, const real& srad1,
                       const real3& pos2, const real& radius2,
                       real3& norm, real& depth,
                       real3& pt1, real3& pt2,
                       real& eff_radius);

bool box_sphere(const real3& pos1, const real4& rot1, const real3& hdims1,
                const real3& pos2, const real& radius2,
                real3& norm, real& depth,
                real3& pt1, real3& pt2,
                real& eff_radius);

bool roundedbox_sphere(const real3& pos1, const real4& rot1, const real3& hdims1, const real& srad1,
                       const real3& pos2, const real& radius2,
                       real3& norm, real& depth,
                       real3& pt1, real3& pt2,
                       real& eff_radius);

bool face_sphere(const real3& A1, const real3& B1, const real3& C1,
                 const real3& pos2, const real& radius2,
                 real3& norm, real& depth,
                 real3& pt1, real3& pt2,
                 real& eff_radius);

int capsule_capsule(const real3& pos1, const real4& rot1, const real& radius1, const real& hlen1,
                    const real3& pos2, const real4& rot2, const real& radius2, const real& hlen2,
                    real3* norm, real* depth,
                    real3* pt1, real3* pt2,
                    real* eff_radius);

int box_capsule(const real3& pos1, const real4& rot1, const real3& hdims1,
                const real3& pos2, const real4& rot2, const real& radius2, const real& hlen2,
                real3* norm, real* depth,
                real3* pt1, real3* pt2,
                real* eff_radius);

int box_box(const real3& pos1, const real4& rot1, const real3& hdims1,
            const real3& pos2, const real4& rot2, const real3& hdims2,
            real3* norm, real* depth,
            real3* pt1, real3* pt2,
            real* eff_radius);

CH_PARALLEL_API
bool RCollision(uint icoll,
                const ConvexShape &shapeA,
                const ConvexShape &shapeB,
                int body1,
                int body2,
                uint* ct_flag,
                real3* ct_norm,
                real3* ct_pt1,
                real3* ct_pt2,
                real* ct_depth,
                real* ct_eff_rad,
                int2* ct_body_ids,
                int& nC);


} // end namespace collision
} // end namespace chrono


#endif

