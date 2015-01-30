#include <algorithm>
#include "chrono_parallel/collision/ChCBroadphase.h"

namespace chrono {
namespace collision {

typedef thrust::pair<real3, real3> bbox;
// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction : public thrust::binary_function<bbox, bbox, bbox> {
   bbox __host__ __device__ operator()(bbox a,
                                       bbox b) {
      real3 ll = R3(std::fmin(a.first.x, b.first.x), std::fmin(a.first.y, b.first.y), std::fmin(a.first.z, b.first.z));     // lower left corner
      real3 ur = R3(std::fmax(a.second.x, b.second.x), std::fmax(a.second.y, b.second.y), std::fmax(a.second.z, b.second.z));     // upper right corner
      return bbox(ll, ur);
   }
};

// convert a point to a bbox containing that point, (point) -> (point, point)
struct bbox_transformation : public thrust::unary_function<real3, bbox> {
   bbox __host__ __device__ operator()(real3 point) {
      return bbox(point, point);
   }
};

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ChCBroadphase::ChCBroadphase() {
   number_of_contacts_possible = 0;
   val = 0;
   last_active_bin = 0;
   number_of_bin_intersections = 0;
   numAABB = 0;
   grid_size = I3(20, 20, 20);
   min_body_per_bin = 25;
   max_body_per_bin = 50;
   // TODO: Should make aabb_data organization less confusing, compiler should switch depending on if the user passes a host/device vector
   // TODO: Should be able to tune bins_per_axis, it's nice to have as a parameter though!
   // TODO: As the collision detection is progressing, we should free up vectors that are no longer being used! For example, Bin_Intersections is only used in steps 4&5
   // TODO: Make sure that aabb_data isn't being duplicated for this code (should be a reference to user's code to conserve space)
   // TODO: Fix debug mode to work with compiler settings, add more debugging features (a drawing function would be nice!)
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void ChCBroadphase::setBinsPerAxis(int3 binsPerAxis) {
   grid_size = binsPerAxis;
#if PRINT_LEVEL==1
   cout << "Set BPA: " << grid_size.x << " " << grid_size.y << " " << grid_size.z << endl;
#endif
}
int3 ChCBroadphase::getBinsPerAxis() {
   return grid_size;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//template<class T>
//inline int3 __host__ __device__ HashMax(     //CHANGED: For maximum point, need to check if point lies on edge of bin (TODO: Hmm, fmod still doesn't work completely)
//		const T &A,
//		const real3 & bin_size_vec) {
//	int3 temp;
//	temp.x = A.x / bin_size_vec.x;
//	if (!fmod(A.x, bin_size_vec.x) && temp.x != 0)
//		temp.x--;
//	temp.y = A.y / bin_size_vec.y;
//	if (!fmod(A.y, bin_size_vec.y) && temp.y != 0)
//		temp.y--;
//	temp.z = A.z / bin_size_vec.z;
//	if (!fmod(A.z, bin_size_vec.z) && temp.z != 0)
//		temp.z--;
//
//	//cout << temp.x << " " << temp.y << " " << temp.z << endl;
//	return temp;
//}

template<class T>
inline int3 __host__ __device__ HashMin(const T &A,
                                        const real3 &bin_size_vec) {
   int3 temp;
   temp.x = A.x * bin_size_vec.x;
   temp.y = A.y * bin_size_vec.y;
   temp.z = A.z * bin_size_vec.z;
   //cout << temp.x << " " << temp.y << " " << temp.z << endl;
   return temp;
}

template<class T>
inline uint __host__ __device__ Hash_Index(const T &A,
                                           int3 grid_size) {
   //return ((A.x * 73856093) ^ (A.y * 19349663) ^ (A.z * 83492791));
   return ((A.z * grid_size.y) * grid_size.x) + (A.y * grid_size.x) + A.x;
}

//Function to Count AABB Bin intersections
inline void __host__ __device__ function_Count_AABB_BIN_Intersection(uint index,
                                                                     const real3 *aabb_data,
                                                                     const real3 &bin_size_vec,
                                                                     uint num_shapes,
                                                                     uint *Bins_Intersected) {
   int3 gmin = HashMin(aabb_data[index], bin_size_vec);
   int3 gmax = HashMin(aabb_data[index + num_shapes], bin_size_vec);
   Bins_Intersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
}

//--------------------------------------------------------------------------
void ChCBroadphase::host_Count_AABB_BIN_Intersection(const real3 *aabb_data,
                                                     uint *Bins_Intersected) {
#pragma omp parallel for schedule(guided)
   for (int i = 0; i < numAABB; i++) {
      function_Count_AABB_BIN_Intersection(i, aabb_data, bin_size_vec, numAABB, Bins_Intersected);
   }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Function to Store AABB Bin Intersections

inline void __host__ __device__ function_Store_AABB_BIN_Intersection(uint index,
                                                                     const real3 *aabb_data,
                                                                     const uint *Bins_Intersected,
                                                                     const real3 &bin_size_vec,
                                                                     const int3 &grid_size,
                                                                     uint num_shapes,
                                                                     uint *bin_number,
                                                                     uint *shape_number) {
   uint count = 0, i, j, k;
   int3 gmin = HashMin(aabb_data[index], bin_size_vec);
   int3 gmax = HashMin(aabb_data[index + num_shapes], bin_size_vec);
   uint mInd = (index == 0) ? 0 : Bins_Intersected[index - 1];
   for (i = gmin.x; i <= gmax.x; i++) {
      for (j = gmin.y; j <= gmax.y; j++) {
         for (k = gmin.z; k <= gmax.z; k++) {
            uint3 location;
            location.x = i;
            location.y = j;
            location.z = k;
            bin_number[mInd + count] = Hash_Index(location, grid_size);
            shape_number[mInd + count] = index;
            count++;
         }
      }
   }
}

//--------------------------------------------------------------------------

void ChCBroadphase::host_Store_AABB_BIN_Intersection(const real3 *aabb_data,
                                                     const uint *Bins_Intersected,
                                                     uint *bin_number,
                                                     uint *shape_number) {
#pragma omp parallel for
   for (int i = 0; i < numAABB; i++) {
      function_Store_AABB_BIN_Intersection(i, aabb_data, Bins_Intersected, bin_size_vec, grid_size, numAABB, bin_number, shape_number);
   }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Check if two bodies interact using their collision family data.
inline bool __host__ __device__ collide(short2 fam_data_A,
                                        short2 fam_data_B)
{
  // Return true only if the bit corresponding to family of B is set in the mask
  // of A and vice-versa.
  return (fam_data_A.y & fam_data_B.x) && (fam_data_B.y & fam_data_A.x);
}

// Check if two AABBs overlap using their min/max corners.
inline bool __host__ __device__ overlap(real3 Amin,
                                        real3 Amax,
                                        real3 Bmin,
                                        real3 Bmax)
{
  // Return true only if the two AABBs overlap in all 3 directions.
  return (Amin.x <= Bmax.x && Bmin.x <= Amax.x) && (Amin.y <= Bmax.y && Bmin.y <= Amax.y) && (Amin.z <= Bmax.z && Bmin.z <= Amax.z);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Function to count AABB AABB intersection

inline void __host__ __device__ function_Count_AABB_AABB_Intersection(uint index,
                                                                      const real3 *aabb_data,
                                                                      uint num_shapes,
                                                                      const uint *bin_number,
                                                                      const uint *shape_number,
                                                                      const uint *bin_start_index,
                                                                      const short2 *fam_data,
                                                                      uint *Num_ContactD) {
  uint start = (index == 0) ? 0 : bin_start_index[index - 1];
  uint end = bin_start_index[index];
  uint count = 0;

  for (uint i = start; i < end; i++) {
    uint shapeA = shape_number[i];
    real3 Amin = aabb_data[shapeA];
    real3 Amax = aabb_data[shapeA + num_shapes];
    short2 famA = fam_data[shapeA];

    for (uint k = i + 1; k < end; k++) {
      uint shapeB = shape_number[k];
      if (shapeA == shapeB) continue;
      if (!collide(famA, fam_data[shapeB])) continue;
      if (!overlap(Amin, Amax, aabb_data[shapeB], aabb_data[shapeB + num_shapes])) continue;
      count++;
    }
  }

  Num_ContactD[index] = count;
}

//--------------------------------------------------------------------------
void ChCBroadphase::host_Count_AABB_AABB_Intersection(const real3 *aabb_data,
                                                      const uint *bin_number,
                                                      const uint *shape_number,
                                                      const uint *bin_start_index,
                                                      const short2 *fam_data,
                                                      uint *Num_ContactD) {
#pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < last_active_bin; i++) {
      function_Count_AABB_AABB_Intersection(i, aabb_data, numAABB, bin_number, shape_number, bin_start_index, fam_data, Num_ContactD);
   }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Function to store AABB-AABB intersections
inline void __host__ __device__ function_Store_AABB_AABB_Intersection(uint index,
                                                                      const real3 *aabb_data,
                                                                      uint num_shapes,
                                                                      const uint *bin_number,
                                                                      const uint *shape_number,
                                                                      const uint *bin_start_index,
                                                                      const uint *Num_ContactD,
                                                                      const short2 *fam_data,
                                                                      long long *potential_contacts) {

  uint start = (index == 0) ? 0 : bin_start_index[index - 1];
  uint end = bin_start_index[index];

  if (end - start == 1) return;

  uint Bin = bin_number[index];
  uint offset = (index == 0) ? 0 : Num_ContactD[index - 1];
  uint count = 0;

  for (uint i = start; i < end; i++) {
    uint shapeA = shape_number[i];
    real3 Amin = aabb_data[shapeA];
    real3 Amax = aabb_data[shapeA + num_shapes];
    short2 famA = fam_data[shapeA];

    for (int k = i + 1; k < end; k++) {
      uint shapeB = shape_number[k];
      if (shapeA == shapeB) continue;
      if (!collide(famA, fam_data[shapeB])) continue;
      if (!overlap(Amin, Amax, aabb_data[shapeB], aabb_data[shapeB + num_shapes])) continue;

      if (shapeB < shapeA) {
        uint t = shapeA;
        shapeA = shapeB;
        shapeB = t;
      }
      potential_contacts[offset + count] = ((long long)shapeA << 32 | (long long)shapeB);     //the two indicies of the shapes that make up the contact
      count++;
    }
  }
}

//--------------------------------------------------------------------------

void ChCBroadphase::host_Store_AABB_AABB_Intersection(const real3 *aabb_data,
                                                      const uint *bin_number,
                                                      const uint *shape_number,
                                                      const uint *bin_start_index,
                                                      const uint *Num_ContactD,
                                                      const short2 *fam_data,
                                                      long long *potential_contacts) {
#pragma omp parallel for schedule (dynamic)
   for (int index = 0; index < last_active_bin; index++) {
      function_Store_AABB_AABB_Intersection(index, aabb_data, numAABB, bin_number, shape_number, bin_start_index, Num_ContactD, fam_data, potential_contacts);
   }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// use spatial subdivision to detect the list of POSSIBLE collisions (let user define their own narrow-phase collision detection)
int ChCBroadphase::detectPossibleCollisions(ChParallelDataManager* data_container)
{
  custom_vector<real3>& aabb_data = data_container->host_data.aabb_rigid;
  custom_vector<long long>& potentialCollisions = data_container->host_data.pair_rigid_rigid;

  const custom_vector<short2>& fam_data = data_container->host_data.fam_rigid;
  const custom_vector<bool>& obj_active = data_container->host_data.active_data;
  const custom_vector<uint>& obj_data_ID = data_container->host_data.id_rigid;

   double startTime = omp_get_wtime();
   numAABB = aabb_data.size() / 2;

#if PRINT_LEVEL==2
   cout << "Number of AABBs: "<<numAABB<<endl;
#endif
   // STEP 1: Initialization TODO: this could be put in the constructor
#ifdef SIM_ENABLE_GPU_MODE
   // set the default cache configuration on the device to prefer a larger L1 cache and smaller shared memory
   cudaFuncSetCacheConfig(device_Count_AABB_BIN_Intersection, cudaFuncCachePreferL1);
   cudaFuncSetCacheConfig(device_Store_AABB_BIN_Intersection, cudaFuncCachePreferL1);
   cudaFuncSetCacheConfig(device_Count_AABB_AABB_Intersection, cudaFuncCachePreferL1);
   cudaFuncSetCacheConfig(device_Store_AABB_AABB_Intersection, cudaFuncCachePreferL1);
   COPY_TO_CONST_MEM(numAABB);
#endif
   potentialCollisions.clear();
   // END STEP 1
   // STEP 2: determine the bounds on the total space and subdivide based on the bins per axis
   bbox init = bbox(aabb_data[0], aabb_data[0]);		// create a zero volume bounding box using the first set of aabb_data (??)
   bbox_transformation unary_op;
   bbox_reduction binary_op;
   bbox result = thrust::transform_reduce(thrust_parallel,aabb_data.begin(), aabb_data.end(), unary_op, init, binary_op);
   min_bounding_point = result.first;
   max_bounding_point = result.second;
   global_origin = (min_bounding_point);		//CHANGED: removed abs
   bin_size_vec = (fabs(max_bounding_point - global_origin));
   bin_size_vec = R3(grid_size.x, grid_size.y, grid_size.z) / bin_size_vec;//CHANGED: this was supposed to be reversed, CHANGED BACK this is just the inverse for convenience (saves us the divide later)
   thrust::transform(aabb_data.begin(), aabb_data.end(), thrust::constant_iterator<real3>(global_origin), aabb_data.begin(), thrust::minus<real3>());
#if PRINT_LEVEL==2
   cout << "Global Origin: (" << global_origin.x << ", " << global_origin.y << ", " << global_origin.z << ")" << endl;
   cout << "Maximum bounding point: (" << max_bounding_point.x << ", " << max_bounding_point.y << ", " << max_bounding_point.z << ")" << endl;
   cout << "Bin size vector: (" << bin_size_vec.x << ", " << bin_size_vec.y << ", " << bin_size_vec.z << ")" << endl;
#endif
   // END STEP 2
   // STEP 3: Count the number AABB's that lie in each bin, allocate space for each AABB
   Bins_Intersected.resize(numAABB);		// TODO: how do you know how large to make this vector?
   // TODO: I think there is something wrong with the hash function...
#ifdef SIM_ENABLE_GPU_MODE
   COPY_TO_CONST_MEM(bin_size_vec);
   device_Count_AABB_BIN_Intersection __KERNEL__(BLOCKS(numAABB), THREADS)(CASTR3(aabb_data), CASTU1(Bins_Intersected));
#else
   host_Count_AABB_BIN_Intersection(aabb_data.data(), Bins_Intersected.data());
#endif
   thrust::inclusive_scan(Bins_Intersected.begin(), Bins_Intersected.end(), Bins_Intersected.begin());
   number_of_bin_intersections = Bins_Intersected.back();
#if PRINT_LEVEL==2
   cout << "Number of bin intersections: " << number_of_bin_intersections << endl;
#endif
   bin_number.resize(number_of_bin_intersections);
   shape_number.resize(number_of_bin_intersections);
   bin_start_index.resize(number_of_bin_intersections);
   // END STEP 3
   // STEP 4: Indicate what bin each AABB belongs to, then sort based on bin number
#ifdef SIM_ENABLE_GPU_MODE
   device_Store_AABB_BIN_Intersection __KERNEL__(BLOCKS(numAABB), THREADS)(CASTR3(aabb_data), CASTU1(Bins_Intersected), CASTU1(bin_number), CASTU1(shape_number));
#else
   host_Store_AABB_BIN_Intersection(aabb_data.data(), Bins_Intersected.data(), bin_number.data(), shape_number.data());
#endif
#if PRINT_LEVEL==2
   cout<<"DONE WID DAT (device_Store_AABB_BIN_Intersection)"<<endl;
#endif
//    for(int i=0; i<bin_number.size(); i++){
//    	cout<<bin_number[i]<<" "<<shape_number[i]<<endl;
//    }

   //Thrust_Sort_By_Key(bin_number, shape_number);
   thrust::sort_by_key(bin_number.begin(), bin_number.end(), shape_number.begin());
//    for(int i=0; i<bin_number.size(); i++){
//    	cout<<bin_number[i]<<" "<<shape_number[i]<<endl;
//    }

#if PRINT_LEVEL==2
#endif

   last_active_bin = (thrust::reduce_by_key(bin_number.begin(), bin_number.end(), thrust::constant_iterator<uint>(1), bin_number.begin(), bin_start_index.begin()).second)
         - bin_start_index.begin();

//    host_vector<uint> bin_number_t=bin_number;
//    host_vector<uint> bin_start_index_t(number_of_bin_intersections);
//    host_vector<uint> Output(number_of_bin_intersections);
//    thrust::pair<uint*,uint*> new_end;
//    last_active_bin= thrust::reduce_by_key(bin_number_t.begin(),bin_number_t.end(),thrust::constant_iterator<uint>(1),Output.begin(),bin_start_index_t.begin()).first-Output.begin();
//
//
//    bin_number=Output;
//    bin_start_index=bin_start_index_t;

#if PRINT_LEVEL==2
#endif
////      //QUESTION: I have no idea what is going on here
   if (last_active_bin <= 0) {
      number_of_contacts_possible = 0;
      return 0;
   }
		//val = bin_start_index[thrust::max_element(bin_start_index.begin(), bin_start_index.begin() + last_active_bin)- bin_start_index.begin()];
		//std::cout<<"max: "<<val<<std::endl;
//		if (val > max_body_per_bin) {
//			grid_size =  I3(grid_size.x+1,grid_size.y+1,grid_size.z+1);
//		} else if (val < min_body_per_bin && val > 10) {
//			grid_size =  I3(grid_size.x-1,grid_size.y-1,grid_size.z-1);
//		}

   bin_start_index.resize(last_active_bin);
#if PRINT_LEVEL==2
   cout <<val<<" "<<grid_size.x<<" "<<grid_size.y<<" "<<grid_size.z<<endl;
   cout << "Last active bin: " << last_active_bin << endl;
#endif
   thrust::inclusive_scan(bin_start_index.begin(), bin_start_index.end(), bin_start_index.begin());
   Num_ContactD.resize(last_active_bin);
   // END STEP 4
   // STEP 5: Count the number of AABB collisions
#ifdef SIM_ENABLE_GPU_MODE
   COPY_TO_CONST_MEM(last_active_bin);
   device_Count_AABB_AABB_Intersection __KERNEL__(BLOCKS(last_active_bin), THREADS)(
         CASTR3(aabb_data),
         CASTU1(bin_number),
         CASTU1(shape_number),
         CASTU1(bin_start_index),
         CASTS2(fam_data),
         CASTU1(Num_ContactD));
#else
   host_Count_AABB_AABB_Intersection(aabb_data.data(), bin_number.data(), shape_number.data(), bin_start_index.data(), fam_data.data(), Num_ContactD.data());
#endif
   thrust::inclusive_scan(Num_ContactD.begin(), Num_ContactD.end(), Num_ContactD.begin());
   number_of_contacts_possible = Num_ContactD.back();
   potentialCollisions.resize(number_of_contacts_possible);
#if PRINT_LEVEL==2
   cout << "Number of possible collisions: " << number_of_contacts_possible << endl;
#endif
   // END STEP 5
   // STEP 6: Store the possible AABB collision pairs
#ifdef SIM_ENABLE_GPU_MODE
   device_Store_AABB_AABB_Intersection __KERNEL__(BLOCKS(last_active_bin), THREADS)(
         CASTR3(aabb_data),
         CASTU1(bin_number),
         CASTU1(shape_number),
         CASTU1(bin_start_index),
         CASTU1(Num_ContactD),
         CASTS2(fam_data),
         CASTLL(potentialCollisions));
#else
   host_Store_AABB_AABB_Intersection(aabb_data.data(), bin_number.data(), shape_number.data(), bin_start_index.data(), Num_ContactD.data(), fam_data.data(),
                                     potentialCollisions.data());
#endif
   thrust::stable_sort(thrust_parallel,potentialCollisions.begin(), potentialCollisions.end());
   number_of_contacts_possible = thrust::unique(potentialCollisions.begin(), potentialCollisions.end()) - potentialCollisions.begin();

   potentialCollisions.resize(number_of_contacts_possible);
#if PRINT_LEVEL==2
   cout << "Number of possible collisions: " << number_of_contacts_possible << endl;
#endif
   // END STEP 6
   double endTime = omp_get_wtime();
#if PRINT_LEVEL==2
   printf("Time to detect: %lf seconds\n", (endTime - startTime));
#endif
   return 0;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

}
}
