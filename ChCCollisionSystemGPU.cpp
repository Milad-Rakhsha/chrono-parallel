//////////////////////////////////////////////////
//  
//   ChCCollisionSystemGPU->cpp
//
// ------------------------------------------------
// 	 Copyright:Alessandro Tasora / DeltaKnowledge
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "ChCCollisionSystemGPU.h"

namespace chrono {
	namespace collision {
		ChCollisionSystemGPU::ChCollisionSystemGPU(ChLcpSystemDescriptorGPU* mdescriptor, float mEnvelope){
			mSystemDescriptor=mdescriptor;
			mGPU=mSystemDescriptor->gpu_collision;
			mGPU->collision_envelope=mEnvelope;
			mGPU->number_of_objects=0;
		}
		ChCollisionSystemGPU::~ChCollisionSystemGPU(){
			for(int i=0; i<mGPU->number_of_objects; i++){delete colModels[i];}
			colModels.clear();
			mGPU->number_of_objects=0;
		}
		void ChCollisionSystemGPU::Add(ChCollisionModel* model){
			if(model->GetPhysicsItem()->GetCollide()==true){
				colModels.push_back((ChCollisionModelGPU*)model);
			}
		}
		void ChCollisionSystemGPU::Run(){
			updateDataStructures();
			mGPU->Run();
			mSystemDescriptor->number_of_contacts=mGPU->number_of_contacts;
		}
		void ChCollisionSystemGPU::updateDataStructures(){
			mGPU->object_data_host.clear();
			for(int i=0; i<colModels.size(); i++){
				body = colModels[i];
				int type=body->GetShapeType();
				if(type==SPHERE){
					object temp_obj;
					float4 B=body->mData[0].B;
					localPos=body->GetBody()->GetPos();
					ChQuaternion<> quat=body->GetBody()->GetRot();
					quat.Normalize();
					temp_obj.A=F4(localPos.x,localPos.y,localPos.z,i);
					temp_obj.B=F4(B.x,0,0,0);
					temp_obj.C=F4(quat.e1,quat.e2,quat.e3,quat.e0);
					temp_obj.family=I2(i,i);
					mGPU->object_data_host.push_back(temp_obj);
				}
				else if(type==COMPOUND){
					for(int j=0; j<body->GetNObjects(); j++){
						float4 A=body->mData[j].A;
						float4 B=body->mData[j].B;
						float4 C=body->mData[j].C;
						ChQuaternion<> quat=body->GetBody()->GetRot();//+ChQuaternion<>(C.x,C.y,C.z,C.w);
						quat.Normalize();
						object temp_obj;
						gPos = body->GetBody()->GetCoord().TrasformLocalToParent(ChVector<>(A.x,A.y,A.z));
						temp_obj.A=F4(gPos.x,gPos.y,gPos.z,i);
						temp_obj.B=F4(B.x,B.y,B.z,B.w);
						
						quat=quat%ChQuaternion<>(C.x,C.y,C.z,C.w);
						
						temp_obj.C=F4(quat.e1,quat.e2,quat.e3,quat.e0);
						temp_obj.family=I2(i,i);
						mGPU->object_data_host.push_back(temp_obj);
					}
				}
				else if(type==BOX){
					localPos=body->GetBody()->GetPos();
					float4 B=body->mData[0].B;
					object temp_obj;
					ChQuaternion<> quat=body->GetBody()->GetRot();
					quat.Normalize();
					temp_obj.A=F4(localPos.x,localPos.y,localPos.z,i);
					temp_obj.B=F4(B.x,B.y,B.z,2);
					temp_obj.C=F4(quat.e1,quat.e2,quat.e3,quat.e0);
					temp_obj.family=I2(body->GetFamily(),body->GetNoCollFamily());
					mGPU->object_data_host.push_back(temp_obj);
				}
				else if(type==ELLIPSOID){
					localPos=body->GetBody()->GetPos();
					float4 B=body->mData[0].B;
					object temp_obj;
					ChQuaternion<> quat=body->GetBody()->GetRot();
					quat.Normalize();
					temp_obj.A=F4(localPos.x,localPos.y,localPos.z,i);
					temp_obj.B=F4(B.x,B.y,B.z,3);
					temp_obj.C=F4(quat.e1,quat.e2,quat.e3,quat.e0);
					temp_obj.family=I2(body->GetFamily(),body->GetNoCollFamily());
					mGPU->object_data_host.push_back(temp_obj);
				}
				else if(type==CYLINDER){
					localPos=body->GetBody()->GetPos();
					float4 B=body->mData[0].B;
					object temp_obj;
					ChQuaternion<> quat=body->GetBody()->GetRot();
					quat.Normalize();
					temp_obj.A=F4(localPos.x,localPos.y,localPos.z,i);
					temp_obj.B=F4(B.x,B.y,B.z,4);
					temp_obj.C=F4(quat.e1,quat.e2,quat.e3,quat.e0);
					temp_obj.family=I2(body->GetFamily(),body->GetNoCollFamily());
					mGPU->object_data_host.push_back(temp_obj);
				}
				else if(type==TRIANGLEMESH){
					for(int j=0; j<body->GetNObjects(); j++){
						localPos=body->GetBody()->GetPos();
						ChMatrix33<>* rotA=body->GetBody()->GetA();
						ChVector<> pA=localPos+rotA->Matr_x_Vect(ChVector<>(body->mData[j].A.x,body->mData[j].A.y,body->mData[j].A.z));
						ChVector<> pB=localPos+rotA->Matr_x_Vect(ChVector<>(body->mData[j].B.x,body->mData[j].B.y,body->mData[j].B.z));
						ChVector<> pC=localPos+rotA->Matr_x_Vect(ChVector<>(body->mData[j].C.x,body->mData[j].C.y,body->mData[j].C.z));
						object temp_obj;
						float mu=body->GetBody()->GetKfriction();
						temp_obj.A=F4(pA.x,pA.y,pA.z,i);
						temp_obj.B=F4(pB.x,pB.y,pB.z,1);
						temp_obj.C=F4(pC.x,pC.y,pC.z,0);
						temp_obj.family=I2(body->GetFamily(),body->GetNoCollFamily());
						mGPU->object_data_host.push_back(temp_obj);
					}
				}
			}
			mGPU->number_of_objects=mGPU->object_data_host.size();
		}
		void ChCollisionSystemGPU::Clear(void){}   
		void ChCollisionSystemGPU::Remove(ChCollisionModel* model){}
		void ChCollisionSystemGPU::ReportContacts(ChContactContainerBase* mcontactcontainer){}
		void ChCollisionSystemGPU::ReportProximities(ChProximityContainerBase* mproximitycontainer){mproximitycontainer->BeginAddProximities();mproximitycontainer->EndAddProximities();}
		bool ChCollisionSystemGPU::RayHit(const ChVector<>& from, const ChVector<>& to, ChRayhitResult& mresult){return false;}
	} // END_OF_NAMESPACE____
} // END_OF_NAMESPACE____
