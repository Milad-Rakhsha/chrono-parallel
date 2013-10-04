#ifndef CHC_COLLISIONSYSTEMBULLETGPUA_H
#define CHC_COLLISIONSYSTEMBULLETGPUA_H
//////////////////////////////////////////////////
//
//   ChCCollisionSystemBulletGPU.h
//
//   Header for class for collision engine based on
//   spatial subdivision method, performed on GPU.
//
//   HEADER file for CHRONO,
//   Multibody dynamics engine
//
// ------------------------------------------------
//   Copyright:Alessandro Tasora / DeltaKnowledge
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "../ChParallelDefines.h"
#include "ChCCollisionModelParallel.h"
#include "physics/ChBody.h"

#include "../ChLcpSystemDescriptorParallel.h"
#include "ChContactContainerParallel.h"
#include "collision/ChCCollisionSystem.h"
#include "physics/ChProximityContainerBase.h"
#include "physics/ChBody.h"
#include "ChCAABBGenerator.h"
#include "ChCNarrowphase.h"
#include "ChCBroadphase.h"
#include "collision/ChCCollisionSystem.h"
#include "collision/bullet/btBulletCollisionCommon.h"

#include "core/ChApiCE.h"
#include "collision/ChCCollisionSystem.h"
#include "collision/bullet/btBulletCollisionCommon.h"

#include "collision/ChCModelBullet.h"
#include "collision/gimpact/GIMPACT/Bullet/btGImpactCollisionAlgorithm.h"
#include "physics/ChBody.h"
#include "physics/ChContactContainerBase.h"
#include "physics/ChProximityContainerBase.h"
#include "LinearMath/btPoolAllocator.h"
#include "ChDataManager.h"


namespace chrono {
	namespace collision {
///
/// Class for collision engine based on the spatial subdivision method.
/// Contains both the broadphase and the narrow phase methods.
///

		class ChApiGPU ChCollisionSystemBulletParallel: public ChCollisionSystem {
			public:

				ChCollisionSystemBulletParallel(unsigned int max_objects = 16000, double scene_size = 500);
				virtual ~ChCollisionSystemBulletParallel();

				/// Clears all data instanced by this algorithm
				/// if any (like persistent contact manifolds)
				virtual void Clear(void);

				/// Adds a collision model to the collision
				/// engine (custom data may be allocated).
				virtual void Add(ChCollisionModel* model);

				/// Removes a collision model from the collision
				/// engine (custom data may be deallocated).
				virtual void Remove(ChCollisionModel* model);

				/// Removes all collision models from the collision
				/// engine (custom data may be deallocated).
				//virtual void RemoveAll();

				/// Run the algorithm and finds all the contacts.
				/// (Contacts will be managed by the Bullet persistent contact cache).
				virtual void Run();

				/// After the Run() has completed, you can call this function to
				/// fill a 'contact container', that is an object inherited from class
				/// ChContactContainerBase. For instance ChSystem, after each Run()
				/// collision detection, calls this method multiple times for all contact containers in the system,
				/// The basic behavior of the implementation is the following: collision system
				/// will call in sequence the functions BeginAddContact(), AddContact() (x n times),
				/// EndAddContact() of the contact container.
				virtual void ReportContacts(ChContactContainerBase* mcontactcontainer);

				/// After the Run() has completed, you can call this function to
				/// fill a 'proximity container' (container of narrow phase pairs), that is
				/// an object inherited from class ChProximityContainerBase. For instance ChSystem, after each Run()
				/// collision detection, calls this method multiple times for all proximity containers in the system,
				/// The basic behavior of the implementation is  the following: collision system
				/// will call in sequence the functions BeginAddProximities(), AddProximity() (x n times),
				/// EndAddProximities() of the proximity container.
				virtual void ReportProximities(ChProximityContainerBase* mproximitycontainer){}

				/// Perform a raycast (ray-hit test with the collision models).
				virtual bool RayHit(const ChVector<>& from, const ChVector<>& to, ChRayhitResult& mresult){}

				// For Bullet related stuff
				btCollisionWorld* GetBulletCollisionWorld() {
					return bt_collision_world;
				}
				ChGPUDataManager* data_container;

			private:
				btCollisionConfiguration* bt_collision_configuration;
				btCollisionDispatcher* bt_dispatcher;
				btBroadphaseInterface* bt_broadphase;
				btCollisionWorld* bt_collision_world;

				uint counter;

		};
	} // END_OF_NAMESPACE____
} // END_OF_NAMESPACE____

#endif
