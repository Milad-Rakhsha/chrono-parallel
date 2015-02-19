#ifndef CHC_AABBGENERATOR_H
#define CHC_AABBGENERATOR_H

#include "collision/ChCCollisionModel.h"

#include "chrono_parallel/math/ChParallelMath.h"
#include "chrono_parallel/ChParallelDefines.h"

#include <thrust/host_vector.h>

namespace chrono {
namespace collision {

class CH_PARALLEL_API ChCAABBGenerator {
 public:
  // functions
  ChCAABBGenerator();

  void GenerateAABB(const custom_vector<shape_type>& obj_data_T,  // Shape Type
                    const custom_vector<real3>& obj_data_A,  // Data A
                    const custom_vector<real3>& obj_data_B,  // Data B
                    const custom_vector<real3>& obj_data_C,  // Data C
                    const custom_vector<real4>& obj_data_R,  // Data D
                    const custom_vector<uint>& obj_data_ID,  // Body ID
                    const custom_vector<real3>& convex_data,  // Convex object data
                    const custom_vector<real3>& body_pos,  // Position global
                    const custom_vector<real4>& body_rot,  // Rotation global
                    const real collision_envelope,
                    custom_vector<real3>& aabb_data);

 private:
  void host_ComputeAABB(const shape_type* obj_data_T,
                        const real3* obj_data_A,
                        const real3* obj_data_B,
                        const real3* obj_data_C,
                        const real4* obj_data_R,
                        const uint* obj_data_ID,
                        const real3* convex_data,
                        const real3* body_pos,
                        const real4* body_rot,
                        const real collision_envelope,
                        real3* aabb_data);

  // variables
  uint numAABB, numAABB_fluid;
};
}
}

#endif
