


#include "fluid_system.cuh"
//#include "device_atomic_functions.hpp"
#include <vector>
#include <thrust\reduce.h>
#include <thrust\functional.h>
#include <thrust\execution_policy.h>
#include"timer.h"

#include <iostream>
using namespace std;
extern vector<float3> boundaryWall;
extern Timer timer;


//#ifndef __CUDACC__
//#define __CUDACC__

#define UNDEF_GRID -1
#define ITERATION_MAX_NUM 20000000

extern struct cudaGraphicsResource *cuda_vbo_resource[2];


namespace IISPH {
	float radius = 0.02;
	float smoothRadius = radius * 4;
	float densityRatio = 1;   //control neighborNum
	float GridSize = smoothRadius / densityRatio;
	float3 minGridCorner = { -0.5,-0.5,-0.5 };
	float3 maxGridCorner = { 0.5,0.5,0.5 };
	float3 OuterGridRange = maxGridCorner - minGridCorner + (2.0 * smoothRadius)*make_float3(1.0, 1.0, 1.0); //FLOAT3_ADD( FLOAT3_SUB(maxGridCorner,minGridCorner),make_float3(smoothRadius*2, smoothRadius*2, smoothRadius*2));
	float3 minOuterBound;
	float3 minWaterCorner;
	float3 maxWaterCorner;
	float3 waterRange;
	float restDensity = 1000.0;
	uint influcedParticleNum = 50;
	float mass = 4.0 * M_PI / (3 * influcedParticleNum)*pow(smoothRadius, 3.0f)*restDensity;
	float initialDistance = pow(mass / restDensity, 1.0 / 3.0);

	//initialized in setup function
	int3 outerGridDim;
	uint particleNum;
	int outerGridNum; //including ghost particles


	int ghostnum;
	dim3 blocksize_p;
	dim3 gridsize_p; //1 dimension

					 //dim3 blocksize_grid(outerGridDim.x, outerGridDim.y);
	dim3 gridsize_grid;
	bufflist Ibuf;
	uint Location[2];
	uint ParticleVAO[2];
	__constant__ ParticleParams _param;
}


#define ErrorBound 0.02

using namespace IISPH;
inline __device__ __host__ int getGrid(float3& pos) {
	int gridx = (pos.x - _param.minOuterBound.x) / _param._GridSize;
	int gridy = (pos.y - _param.minOuterBound.y) / _param._GridSize;
	int gridz = ((pos.z - _param.minOuterBound.z) / _param._GridSize);
	if (gridx >= 0 && gridx < _param.outerGridDim.x&&gridy >= 0 && gridy < _param.outerGridDim.y&&gridz >= 0 && gridz < _param.outerGridDim.z)
		return  gridz + gridy * _param.outerGridDim.z + gridx * _param.outerGridDim.z*_param.outerGridDim.y;
	else
		return UNDEF_GRID;

}
__device__ inline void boxBoundaryForce(const float3& position, float3& force)
{
}
int getGridCpuI(float3& pos) {
	int gridx = (pos.x - IISPH::minOuterBound.x) / GridSize;
	int gridy = (pos.y - IISPH::minOuterBound.y) / GridSize;
	int gridz = ((pos.z - IISPH::minOuterBound.z) / GridSize);
	if (gridx >= 0 && gridx < IISPH::outerGridDim.x&&gridy >= 0 && gridy < IISPH::outerGridDim.y&&gridz >= 0 && gridz < IISPH::outerGridDim.z)
		return  gridz + gridy * IISPH::outerGridDim.z + gridx * IISPH::outerGridDim.z*IISPH::outerGridDim.y;
	else {
		cout << "error" << pos.x << ' ' << pos.y << ' ' << pos.z << ' ' << endl;
		return UNDEF_GRID;
	}


}
inline __host__ __device__ float poly6kernelVal(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernel*(tmp*tmp*tmp);
}
inline __host__ __device__ float poly6kernelGradient(float dist) {
	/*if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernelGradient*(-ratio * tmp*tmp);*/
	if (dist > _param.smooth_radius)
		return 0;
    float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.spikykernelGradient*(-1.0 * tmp*tmp);
}

inline __host__ __device__ float spikykernelGradient(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.spikykernelGradient*(-1.0 * tmp*tmp);
}

__global__ void CountParticleInGridI(float3* p, bufflist Ibuf) {
	//int i = threadIdx.z * 25 + threadIdx.y * 5 + threadIdx.x;
	int i = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 point = p[i];
	const float3 vec_bound_min = _param._minGridCorner;
	const float3 vec_bound_max = _param._maxGridCorner;
	{
		int gridIndex = getGrid(point);
		if (gridIndex == UNDEF_GRID)
			printf("error grid\n");
		Ibuf.particle_grid_cell_index[i] = gridIndex;
		Ibuf.grid_particle_offset[i] = atomicAdd(&Ibuf.grid_particles_num[gridIndex], 1);
	}
}
__global__ void computeDensity(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;

	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;


	float particle_density = 0.0;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int3 cell_neighbor = make_int3(cell_x, cell_y, cell_z) + _param._neighbor_off[cell];
		if (cell_neighbor.x < 0 || cell_neighbor.x >= _param.outerGridDim.x || cell_neighbor.y < 0 || cell_neighbor.y >= _param.outerGridDim.y || cell_neighbor.z < 0 || cell_neighbor.z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor.z + cell_neighbor.y * GridnumRange.z + cell_neighbor.x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];

		for (int cndx = cell_start; cndx < cell_end; cndx++)
		{
			int j = cndx;
			if (tid == j)
				continue;

			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.pos_update[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius && dist > 0)
			{
				float kernelValue = poly6kernelVal(dist);
				particle_density += kernelValue * _param.mass;
				//predictedSPHDensity += 1;
			}
		}

		//ghost particles
		 cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		 cell_end = cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int cndx = cell_start; cndx < cell_end; cndx++)
		{
			int j = cndx;
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.ghost_pos[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius )
			{
				float kernelValue =  poly6kernelVal(dist) *Ibuf.ghost_volum[j];
				particle_density += kernelValue * _param.mass;

			}
		}
	}

	particle_density += _param.poly6kernel* _param.mass;
	Ibuf.particle_density[tid] = particle_density;
}
__global__ void AdvectionAndComputeDii(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 acceleration = (1.0 / _param.mass)*(Ibuf.force[tid] + Ibuf.correction_pressure_force[tid]);
	Ibuf.vel_update[tid]  += acceleration * _param.time_step;
	float3 dii = make_float3(0.0);

	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;

	float3& ipos = Ibuf.pos_update[tid];
	float idensity = Ibuf.particle_density[tid];
	float idensity_square = idensity*idensity;
	const float diiCoefficient = _param.time_step* _param.time_step*(_param.mass / idensity_square);
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int3 cell_neighbor = make_int3(cell_x, cell_y, cell_z) + _param._neighbor_off[cell];
		if (cell_neighbor.x < 0 || cell_neighbor.x >= _param.outerGridDim.x || cell_neighbor.y < 0 || cell_neighbor.y >= _param.outerGridDim.y || cell_neighbor.z < 0 || cell_neighbor.z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor.z + cell_neighbor.y * GridnumRange.z + cell_neighbor.x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];

		for (int j = cell_start; j < cell_end; j++)
		{
			if (j == tid)
				continue;
			float3& jpos = Ibuf.pos_update[j];
			float3 vector_i_minus_j = (ipos-jpos);
			const float dist = length(vector_i_minus_j);
		    
			if (dist <= _param.smooth_radius )
			{
				float3& jvel = Ibuf.vel_update[j];
				dii -= diiCoefficient *poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				//predictedSPHDensity += 1;
			}
		}

		//ghost particles
		 cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		 cell_end = cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.ghost_pos[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius )
			{
				
				dii -=  Ibuf.ghost_volum[j]* diiCoefficient * poly6kernelGradient(dist)/dist*(vector_i_minus_j);

			}
		}
	}
	Ibuf.Dii[tid] = dii;
}
__global__ void ComputeAdvectionDensityAndAii(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid > _param.particleNum) return;
	float densityAdv = Ibuf.particle_density[tid];
	float aii = 0.0;
	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;

	float3& ivel = Ibuf.vel_update[tid];
	float3& ipos = Ibuf.pos_update[tid];
	float3& dii = Ibuf.Dii[tid];
	float idensity = Ibuf.particle_density[tid];
	float idensity_square = idensity * idensity;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int3 cell_neighbor = make_int3(cell_x, cell_y, cell_z) + _param._neighbor_off[cell];
		if (cell_neighbor.x < 0 || cell_neighbor.x >= _param.outerGridDim.x || cell_neighbor.y < 0 || cell_neighbor.y >= _param.outerGridDim.y || cell_neighbor.z < 0 || cell_neighbor.z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor.z + cell_neighbor.y * GridnumRange.z + cell_neighbor.x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];

		for (int j = cell_start; j < cell_end; j++)
		{
			if (j == tid)
				continue;
			float3& jpos = Ibuf.pos_update[j];
			float3 vector_i_minus_j = (ipos - jpos);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius)
			{
				float3& jvel = Ibuf.vel_update[j];
				float3 gradVec = poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				densityAdv += _param.time_step* _param.mass*
					dot(ivel - jvel, gradVec);
				float jdensity_square = Ibuf.particle_density[j] * Ibuf.particle_density[j];
				float3 dji = (_param.time_step*_param.time_step)*_param.mass / idensity_square *gradVec;
				aii +=  _param.mass*dot(dii - dji, gradVec);
				//predictedSPHDensity += 1;
			}
		}

		//ghost particles
		 cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		cell_end = cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.ghost_pos[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius)
			{
				float3 jvel = { 0.0,0.0,0.0 };
				float3 gradVec = poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				densityAdv += _param.time_step* _param.mass*Ibuf.ghost_volum[j] *
					dot(ivel , gradVec);
				float jdensity_square = _param.rest_density*_param.rest_density;
				//float3 dji = (_param.time_step*_param.time_step)*_param.mass / idensity_square * gradVec;
				aii += _param.mass*Ibuf.ghost_volum[j]*dot(dii, gradVec);
			}
		}
	}
	Ibuf.advect_density[tid] = densityAdv;
	Ibuf.aii[tid] = aii;
	Ibuf.correction_pressure[tid] = 0.0;
}
__global__ void ComputeDij_Pj(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 dij_pj = make_float3( 0.0,0.0,0.0);
	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;


	float particle_density = 0.0;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int3 cell_neighbor = make_int3(cell_x, cell_y, cell_z) + _param._neighbor_off[cell];
		if (cell_neighbor.x < 0 || cell_neighbor.x >= _param.outerGridDim.x || cell_neighbor.y < 0 || cell_neighbor.y >= _param.outerGridDim.y || cell_neighbor.z < 0 || cell_neighbor.z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor.z + cell_neighbor.y * GridnumRange.z + cell_neighbor.x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			if (j == tid)
				continue;
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.pos_update[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius)
			{
				float3 gradVec = poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				float jdensity_square = Ibuf.particle_density[j] * Ibuf.particle_density[j];
				dij_pj -= (_param.time_step*_param.time_step)*_param.mass*Ibuf.correction_pressure[j]/ jdensity_square* gradVec;
				//predictedSPHDensity += 1;
			}
		}
	}
	//if (tid == 4)
	//	printf("%f %f %f sumDijPj\n",Ibuf.pos_update[tid].x, Ibuf.pos_update[tid].y, Ibuf.pos_update[tid].z,dij_pj);
	Ibuf.sumDij_Pj[tid] = dij_pj;
}
__global__ void ComputeNewPressure(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float aii = Ibuf.aii[tid];
	float idensity_sqaure = Ibuf.particle_density[tid]* Ibuf.particle_density[tid];
	float ipressure = Ibuf.correction_pressure[tid];
	float sum = 0.0;
	float omega = 0.452;
	float3& isumDij_Pj = Ibuf.sumDij_Pj[tid];
	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;


	float particle_density = 0.0;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int3 cell_neighbor = make_int3(cell_x, cell_y, cell_z) + _param._neighbor_off[cell];
		if (cell_neighbor.x < 0 || cell_neighbor.x >= _param.outerGridDim.x || cell_neighbor.y < 0 || cell_neighbor.y >= _param.outerGridDim.y || cell_neighbor.z < 0 || cell_neighbor.z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor.z + cell_neighbor.y * GridnumRange.z + cell_neighbor.x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			if (j == tid)
				continue;
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.pos_update[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius)
			{
				float3 gradVec = poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				float jdensity_square = Ibuf.particle_density[j] * Ibuf.particle_density[j];
				float3 dji = (_param.time_step*_param.time_step)*_param.mass / idensity_sqaure * gradVec;
				sum += _param.mass*dot(isumDij_Pj-Ibuf.Dii[j]*Ibuf.correction_pressure[j]
					-Ibuf.sumDij_Pj[j]+dji* ipressure, gradVec);
				//dij_pj -= (_param.time_step*_param.time_step)*_param.mass*Ibuf.correction_pressure[j] / jdensity_square * gradVec;
				//predictedSPHDensity += 1;
			}
		}
		 cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		cell_end = cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			float3 vector_i_minus_j = (Ibuf.pos_update[tid] - Ibuf.ghost_pos[j]);
			const float dist = length(vector_i_minus_j);
			if (dist <= _param.smooth_radius)
			{
				float3 gradVec = poly6kernelGradient(dist)/dist*(vector_i_minus_j);
				sum += Ibuf.ghost_volum[j] * _param.mass*dot(isumDij_Pj, gradVec);
			}
		}
	}
	float denom = aii ;
	float newPressure;
	float pressureDifference = _param.rest_density - Ibuf.advect_density[tid];
	if (fabs(denom) > 0.0) {
		newPressure = MAX( (1.0 - omega)*ipressure + (omega / aii) * (pressureDifference - sum),0.0);
		Ibuf.densityError[tid] = aii * newPressure+ sum - pressureDifference;
	}
	else {
		newPressure = 0.0;
		Ibuf.densityError[tid] = 0.0;
	}
	Ibuf.correction_pressure_update[tid] = newPressure;
}

//

void IISPH_solver::setUpParameter()
{
	radius = this->radius;
	smoothRadius = this->smooth_radius;
	 GridSize = smoothRadius / densityRatio;
	minGridCorner = { -0.5,-0.5,-0.5 };
	maxGridCorner = { 0.5,0.5,0.5 };
	OuterGridRange = maxGridCorner - minGridCorner + (2.0 * smoothRadius)*make_float3(1.0, 1.0, 1.0); //FLOAT3_ADD( FLOAT3_SUB(maxGridCorner,minGridCorner),make_float3(smoothRadius*2, smoothRadius*2, smoothRadius*2));
	minOuterBound = make_float3(minGridCorner.x - smoothRadius,minGridCorner.y - smoothRadius,minGridCorner.z - smoothRadius );
	minWaterCorner = { -0.0f,-0.5,-0.5f };
	maxWaterCorner = { 0.5,0.5,0.5f };
	waterRange = maxWaterCorner - minWaterCorner;
	restDensity = 1000.0;
	influcedParticleNum = 50;
	mass = 4.0 * M_PI / (3 * influcedParticleNum)*pow(smoothRadius, 3.0f)*restDensity;
	initialDistance = pow(mass / restDensity, 1.0 / 3.0);
	outerGridDim = { (int)ceil(OuterGridRange.x / GridSize),(int)ceil(OuterGridRange.y / GridSize) ,(int)ceil(OuterGridRange.z / GridSize) };
	particleNum = (int)ceil(waterRange.x / initialDistance)*(int)ceil(waterRange.y / initialDistance)*(int)ceil(waterRange.z / initialDistance);  //a particle per grid
	outerGridNum = outerGridDim.x*outerGridDim.y*outerGridDim.z; //including ghost particles
	blocksize_p = dim3((uint)ceil(waterRange.x / initialDistance), (uint)ceil(waterRange.y / initialDistance));
	gridsize_p = dim3((uint)ceil(waterRange.z / initialDistance)); //1 dimension
	gridsize_grid = dim3((uint)outerGridDim.z);
	float gravity = 9.0;
	float poly6kernel = 315.0f / (64.0f*M_PI*pow(smoothRadius, 3.0f));
	float poly6kernelGrad = 945.0f / (32.0f*M_PI*pow(smoothRadius, 4.0f));
	float boudary_force_factor = 25.0;
	float time_step = this->time_step;
	float spikykernelGrad = 45.0f / (M_PI*pow(smoothRadius, 4.0f));

	_param._minGridCorner = minGridCorner;
	_param._maxGridCorner = maxGridCorner;
	_param._GridSize = GridSize;
	_param.outerGridDim = outerGridDim;
	_param.particleNum = particleNum;
	_param.gravity = gravity;
	_param.mass = mass;
	_param.time_step = time_step;
	_param.smooth_radius = smoothRadius;
	_param.rest_density = restDensity;
	_param.poly6kernel = poly6kernel;
	_param.poly6kernelGradient = poly6kernelGrad;
	_param.minOuterBound = minOuterBound;
	_param.spikykernelGradient = spikykernelGrad;
	uint index = 0;
	for (int i = -1; i < 2; i++)
		for (int j = -1; j < 2; j++)
			for (int k = -1; k < 2; k++)
				_param._neighbor_off[index++] = { k,j,i };

}
void IISPH_solver::particleSetUp()
{
	//CUDA_SAFE_CALL(cudaMalloc((void**)&testdata, 9*sizeof(float)));
	//memset(testdata, 0, 9 * sizeof(float));
	setUpParameter();
	//dim3 blocksize_grid(outerGridDim.x, outerGridDim.y);

	glGenBuffers(2, &Location[0]);
	glGenVertexArrays(2, &ParticleVAO[0]);
	vector<float3> initialPos = vector<float3>();
	vector<float3> ghostPos = vector<float3>();
	int total = particleNum;
	int numX = (int)ceil(waterRange.x / initialDistance);
	int numY = (int)ceil(waterRange.y / initialDistance);
	int numZ = (int)ceil(waterRange.z / initialDistance);
	float offsetX = (waterRange.x - (numX - 1)*initialDistance) / 2.0f;
	float offsetY = (waterRange.y - (numY - 1)*initialDistance) / 2.0f;
	float offsetZ = (waterRange.z - (numZ - 1)*initialDistance) / 2.0f;
	float3 startPos = make_float3(-offsetX + maxWaterCorner.x, -offsetY + maxWaterCorner.y, -offsetZ + maxWaterCorner.z);
	for (int i = 0; i<numX; i++)
		for (int j = 0; j<numY; j++)
			for (int k = 0; k < numZ; k++) {
				int index = k + numZ * j + i * numZ*numY;
				float3 move = { i*initialDistance,j*initialDistance,k*initialDistance };
				float3 pos = (startPos - move);

				initialPos.push_back(pos);
			}
	float posx = startPos.x;
	float posy = startPos.y;
	float posz = startPos.z;


	//ComputeDensityErrorFactor(initialPos, initialPos.size() / 2);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(_param, &_param, sizeof(ParticleParams), 0, cudaMemcpyHostToDevice));
	ghostPos = boundaryWall;
	for (int i = 0; i < 2; i++) {
		glBindVertexArray(ParticleVAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, Location[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*initialPos.size(), &initialPos[0], GL_STATIC_DRAW);
		//glVertexAttribPointer(1,)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(0);
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource[i], Location[i], cudaGraphicsRegisterFlagsNone);
	}

	cudaDeviceSynchronize();
	//	uint index = 0;

	//{ minGridCorner ,maxGridCorner,GridIndexRange,GridSize,particleNum,gravity,mass,time_step,smoothRadius,restDensity,poly6kernel,poly6kernelGrad,density_error_factor,boudary_force_factor} ;

	//	CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceArray, hostArray, sizeof(float)*10, 0, cudaMemcpyHostToDevice));

	//memset(initialPos,0.2, total*sizeof(float3));

	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.grid_particle_offset, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.particle_grid_cell_index, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.particle_grid_cell_index_update, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.vel_old, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.vel_update, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.pos_update, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.sort_index, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.force, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemset(Ibuf.vel_old, 0, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.correction_pressure_force, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.correction_pressure, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.max_predicted_density, sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.densityError, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(Ibuf.densityError, 0, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.grid_off, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.grid_particles_num, outerGridNum * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(Ibuf.grid_particles_num, 0, outerGridNum * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.ghost_grid_off, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.ghost_grid_particles_num, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.ghost_volum, ghostPos.size() * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.ghost_pos, ghostPos.size() * sizeof(float3)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.Dii, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.aii, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.sumDij_Pj, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.particle_density, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.advect_density, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Ibuf.correction_pressure_update, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(Ibuf.correction_pressure,0.0, particleNum * sizeof(float)));

	ghostnum = ghostPos.size();
	vector<int> ghost_particle_index_grid(ghostPos.size());
	vector<int> ghost_particle_grid_index(ghostPos.size());
	vector<int> sorted_index(ghostPos.size());
	vector<float3> pos_tmp(ghostPos.size());
	vector<int> ghost_grid_particles_num(outerGridNum, 0);
	vector<int> ghost_grid_off(outerGridNum + 1);
	vector<float> ghost_vol(ghostnum);
	for (int i = 0; i < ghostPos.size(); i++) {
		float3 pos = ghostPos[i];
		int index = getGridCpuI(pos);
		if (index != UNDEF_GRID) {
			ghost_particle_grid_index[i] = index;
			ghost_particle_index_grid[i] = ghost_grid_particles_num[index];
			ghost_grid_particles_num[index]++;
		}
	}
	int grid_off = 0;
	for (int i = 0; i < outerGridNum; i++) {
		ghost_grid_off[i] = grid_off;
		grid_off += ghost_grid_particles_num[i];
	}
	ghost_grid_off.push_back(grid_off);
	for (int i = 0; i < ghostPos.size(); i++) {
		int cell_index = ghost_particle_grid_index[i];
		sorted_index[i] = ghost_grid_off[cell_index] + ghost_particle_index_grid[i];
	}
	for (int i = 0; i < ghostPos.size(); i++) {
		int index = sorted_index[i];
		pos_tmp[index] = ghostPos[i];
	}
	for (int i = 0; i < ghostPos.size(); i++) {
		int i_cell_index = getGridCpuI(pos_tmp[i]);
		int3 GridnumRange = _param.outerGridDim;
		int cell_z = i_cell_index % (GridnumRange.z);
		i_cell_index /= GridnumRange.z;
		int cell_y = i_cell_index % (GridnumRange.y);
		int cell_x = i_cell_index / GridnumRange.y;
		//int index = getGridCpuI(ghostPos[i]);
		float Wsum = 0.0;
		for (int cell = 0; cell < neighborGridNum; cell++)
		{
			int cell_neighbor_x = cell_x + _param._neighbor_off[cell].x;
			int cell_neighbor_y = cell_y + _param._neighbor_off[cell].y;
			int cell_neighbor_z = cell_z + _param._neighbor_off[cell].z;
			if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_y < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
				continue;
			int neighbor_cell_index = cell_neighbor_z + cell_neighbor_y * GridnumRange.z + cell_neighbor_x * GridnumRange.z*GridnumRange.y;
			int ghost_cell_start = ghost_grid_off[neighbor_cell_index];
			int ghost_cell_end = ghost_grid_off[neighbor_cell_index + 1];
			for (int cndx = ghost_cell_start; cndx < ghost_cell_end; cndx++)
			{
				int j = cndx;
				float3 vector_i_minus_j = (pos_tmp[i] - pos_tmp[j]);
				const float jdist = length(vector_i_minus_j);
				if (jdist < _param.smooth_radius)
				{
					float kernel = poly6kernelVal(jdist);
					Wsum += kernel * _param.mass / _param.rest_density;
				}
			}
		}
		ghost_vol[i] = 1.5 / Wsum;
	}
	CUDA_SAFE_CALL(cudaMemcpy(Ibuf.ghost_volum, &ghost_vol[0], sizeof(float)*ghostnum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(Ibuf.ghost_grid_off, &ghost_grid_off[0], sizeof(int)*outerGridNum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(Ibuf.ghost_grid_particles_num, &ghost_grid_particles_num[0], sizeof(int)*outerGridNum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(Ibuf.ghost_pos, &pos_tmp[0], sizeof(float3)*ghostPos.size(), cudaMemcpyHostToDevice));
	ghostnum = ghostPos.size();

}


//helper function


GLuint IISPH_solver::getRenderVBO()
{
	return ParticleVAO[0];
}
void IISPH_solver::swapBuff() {
	std::swap(cuda_vbo_resource[0], cuda_vbo_resource[1]);
	std::swap(Location[0], Location[1]);
	std::swap(ParticleVAO[0], ParticleVAO[1]);
	std::swap(Ibuf.vel_old, Ibuf.vel_update);
	//swap(Ibuf.particle_grid_cell_index, Ibuf.particle_grid_cell_index_update);
}
void IISPH_solver::CountParticles(float3* input) {
	//dim3 threadperBlock(numRangeX,numRangeY ,numRangeZ);
	CUDA_SAFE_CALL(cudaMemset(Ibuf.grid_particles_num, 0, outerGridNum * sizeof(uint)));
	//cudaMemset(Ibuf.max_predicted_density, 0, sizeof(float));

	CountParticleInGridI << <gridsize_p, blocksize_p >> > (input, Ibuf);
	cudaDeviceSynchronize();
}

__global__ void rearrangeI(bufflist Ibuf, float3* pos_old) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	uint index = Ibuf.sort_index[tid];
	Ibuf.vel_update[index] = Ibuf.vel_old[tid];
	Ibuf.pos_update[index] = pos_old[tid];
	Ibuf.particle_grid_cell_index_update[index] = Ibuf.particle_grid_cell_index[tid];
}
__global__ void sortIndexI(bufflist Ibuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	uint cell_index = Ibuf.particle_grid_cell_index[tid];
	uint particle_index = Ibuf.grid_off[cell_index] + Ibuf.grid_particle_offset[tid];
	Ibuf.sort_index[tid] = particle_index;
}
__global__ void computeOtherForceI(bufflist Ibuf) {
	//Gravity
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 nonPressureforce = { 0,-_param.gravity,0 };
	//Viscosity
	float3 & ipos = Ibuf.pos_update[tid];
	float3 & ivel = Ibuf.vel_update[tid];
	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	float smooth_radius_square = _param.smooth_radius*_param.smooth_radius;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int cell_neighbor_x = cell_x + _param._neighbor_off[cell].x;
		int cell_neighbor_y = cell_y + _param._neighbor_off[cell].y;
		int cell_neighbor_z = cell_z + _param._neighbor_off[cell].z;
		if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_y < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor_z + cell_neighbor_y * GridnumRange.z + cell_neighbor_x * GridnumRange.z*GridnumRange.y;
		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			//force.y++;
			if (tid == j)
			{
				continue;
			}
			float3 vector_i_minus_j = (ipos - Ibuf.pos_update[j]);

			const float dist = length(vector_i_minus_j);
			if (dist < _param.smooth_radius)
			{
				float3 vel_i_minus_j = ivel - Ibuf.vel_update[j];
				//float jdist = sqrt(dist_square);
				float kernelGradientValue = poly6kernelGradient(dist);
				float3 kernelGradient = (vector_i_minus_j * kernelGradientValue / dist);
				nonPressureforce += 0.2*(_param.mass*_param.mass / _param.rest_density)*
					dot(vector_i_minus_j, vel_i_minus_j) / (0.01*smooth_radius_square + dist*dist)*kernelGradient;
			}
		}
		int ghost_cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		int ghost_cell_end = ghost_cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int j = ghost_cell_start; j < ghost_cell_end; j++)
		{

			float3 vector_i_minus_j = ipos - Ibuf.ghost_pos[j];
			const float dist = length(vector_i_minus_j);

			if (dist < _param.smooth_radius)
			{
				float3 vel_i_minus_j = ivel;
				//float jdist = sqrt(dist_square);
				float kernelGradientValue = poly6kernelGradient(dist);
				float3 kernelGradient = (vector_i_minus_j * kernelGradientValue / dist);
				nonPressureforce += 0.0*(_param.mass*_param.mass / _param.rest_density)*
					dot(vector_i_minus_j, vel_i_minus_j)*Ibuf.ghost_volum[j] / (0.01*smooth_radius_square + dist * dist)*kernelGradient;
			}
		}
	}
	Ibuf.force[tid] = nonPressureforce;
}




__global__ void ComputePressureForceI(bufflist Ibuf, float3* predicted_pos) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;

	uint i_cell_index = Ibuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;
	if (i_cell_index == UNDEF_GRID)
		return;

	const float3 ipos = Ibuf.pos_update[tid];
	const float  ipress = Ibuf.correction_pressure[tid];
	const float  mass_sqaure = _param.mass*_param.mass;
	const float idensity = Ibuf.particle_density[tid];
	const float idensity2 = idensity * idensity;
	const float  smooth_radius = _param.smooth_radius;
	float3 force = make_float3(0, 0, 0);
	float3 forceB = make_float3(0, 0, 0);
	for (int cell = 0; cell < neighborGridNum; cell++)
	{
		int cell_neighbor_x = cell_x + _param._neighbor_off[cell].x;
		int cell_neighbor_y = cell_y + _param._neighbor_off[cell].y;
		int cell_neighbor_z = cell_z + _param._neighbor_off[cell].z;
		if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_y < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index = cell_neighbor_z + cell_neighbor_y * GridnumRange.z + cell_neighbor_x * GridnumRange.z*GridnumRange.y;
		//water particles

		int cell_start = Ibuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + Ibuf.grid_particles_num[neighbor_cell_index];
		for (int cndx = cell_start; cndx < cell_end; cndx++)
		{
			//force.y++;
			int j = cndx;
			if (tid == j)
			{
				continue;
			}
			float3 vector_i_minus_j = (ipos - Ibuf.pos_update[j]);

			const float dist = length(vector_i_minus_j); //dot(vector_i_minus_j, vector_i_minus_j);
			if (dist < smooth_radius && dist > 0)
			{
				float jdensity_square = Ibuf.particle_density[j] * Ibuf.particle_density[j];
				float kernelGradientValue = poly6kernelGradient(dist);
				float3 kernelGradient = (vector_i_minus_j * kernelGradientValue / dist);
				float grad = mass_sqaure*(ipress/ idensity2 + Ibuf.correction_pressure[j]/jdensity_square);
				force -= kernelGradient * grad;
			}
		}
		int ghost_cell_start = Ibuf.ghost_grid_off[neighbor_cell_index];
		int ghost_cell_end = ghost_cell_start + Ibuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int cndx = ghost_cell_start; cndx < ghost_cell_end; cndx++)
		{
			int j = cndx;
			float3 vector_i_minus_j = ipos - Ibuf.ghost_pos[j];
			const float dist = length(vector_i_minus_j);

			if (dist < smooth_radius && dist > 0)
			{
				float kernelGradientValue = poly6kernelGradient(dist);
				float3 kernelGradient = vector_i_minus_j * ((kernelGradientValue / dist));
				float grad = mass_sqaure * (ipress)/idensity2* Ibuf.ghost_volum[j];
				forceB -= kernelGradient * grad;
			}
		}
	}

	Ibuf.correction_pressure_force[tid] = force + forceB;
	/*if (tid == 0)
	printf("pressure force fluid x: %f,y: %f,z: %f\n pressure force boudary x: %f,y: %f,z: %f\nself x:%f,y:%f,z:%f\n",
	force.x,force.y,force.z,forceB.x,forceB.y,forceB.z,ipos.x,ipos.y,ipos.z);*/

}
__global__ void scanSumIntI(int * input, int * output, int *aux, int numPerthread) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int offset = id * numPerthread;
	int prefix = 0;
	for (int i = offset; i < offset + numPerthread; i++) {
		int tmp = input[i];
		output[i] = prefix;
		prefix += tmp;
	}
	if (aux)
		aux[id] = prefix;

}
__global__ void reduceMaxI(float *g_idata, float *g_odata, int num) {

	extern __shared__ float sdata[];
	//// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	//unsigned int i =  threadIdx.x;
	if (tid<num)
		sdata[tid] = g_idata[tid];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = num; s > 1; s = (s + 1) / 2) {
		//s is odd
		if (s % 2)
		{
			if (tid < s / 2)
				sdata[tid] = MAX(sdata[tid], sdata[tid + s / 2]);
			else if (tid == s / 2)
				sdata[tid] = sdata[s - 1];
		}
		else
			if (tid < s / 2) {
				sdata[tid] = MAX(sdata[tid], sdata[tid + s / 2]);
			}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[0] = sdata[0];
}
__global__ void advanceParticlesI(bufflist Ibuf, float3* output) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;

	float3 acceleration = (1.0 / _param.mass)*(Ibuf.force[tid] + Ibuf.correction_pressure_force[tid]);
	float3 veval = Ibuf.vel_update[tid]+ acceleration * _param.time_step;
	float3 pos = Ibuf.pos_update[tid]+ veval * _param.time_step;
	output[tid] = pos;
	Ibuf.vel_update[tid] = (veval);
}
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
float IISPH_solver::ReduceMax(float* input, float* output, int num) {
	thrust::device_ptr<float> data(input);
	float res = thrust::reduce(data, data + num
		, -1.0,
		thrust::maximum<float>()
	);
	return res;
	dim3 blocksize(ceil(sqrt(num)), ceil(sqrt(num)));
	reduceMaxI << <1, blocksize, num * sizeof(float) >> > (input, output, num);
	cudaDeviceSynchronize();
	float max_density_error;
	cudaMemcpy(&max_density_error, Ibuf.max_predicted_density, sizeof(float), cudaMemcpyDeviceToHost);
	return max_density_error;

}
__global__ void addUpPrefixI(int* prefix, int* aux, int stride) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	prefix[id] += aux[id / stride];
}

void IISPH_solver::prescanInt(int* input, int* output, int len, int numPerThread, int numBlock, int numThread) {
	int * aux;
	//auto input1 = vector<int>(GridNum);

	int totalthread = (numBlock*numThread);
	CUDA_SAFE_CALL(cudaMalloc(&aux, totalthread * sizeof(int)));
	cudaDeviceSynchronize();
	//cudaMemcpy(&input1[0], input, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	scanSumIntI << <numBlock, numThread >> > (input, output, aux, numPerThread);


	scanSumIntI << <1, 1 >> > (aux, aux, NULL, totalthread);  //in place 
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	addUpPrefixI << <numBlock*numThread, numPerThread >> >(output, aux, numPerThread);
	CUDA_SAFE_CALL(cudaFree(aux));
	//cudaMemcpy(&input1[0], output, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	//
	//
	//for (int a : input1)
	//	cout << a << " ";
	//cudaDeviceSynchronize();
}
void IISPH_solver::IndexSort(float3* pos_old) {

	prescanInt(Ibuf.grid_particles_num, Ibuf.grid_off, outerGridNum, outerGridDim.x, outerGridDim.y, outerGridDim.z);
	//dim3 blocksize(particleNum);
	//sortIndex << <1, particleNum >> >(Ibuf.grid_off, Ibuf.grid_particle_offset, Ibuf.particle_grid_cell_index, Ibuf.sort_index);


	cudaDeviceSynchronize();
	sortIndexI << <gridsize_p, blocksize_p >> >(Ibuf);
	//Safe
	cudaDeviceSynchronize();

	rearrangeI << <gridsize_p, blocksize_p >> >(Ibuf, pos_old);


	cudaDeviceSynchronize();
}
//void IISPH_solver::ComputeOtherForce() {    //grivty
//							  //	dim3 blockSize();
//	computeOtherForceI << <gridsize_p, blocksize_p >> >(Ibuf);
//	//Safe
//	auto input1 = vector<float3>(particleNum);
//}
void IISPH_solver::RelaxedJacobiIteration(float3* output) {
	bool densityErrorLarge = true;
	int cnt = 0;
	//CUDA_SAFE_CALL(cudaMemset(Ibuf.correction_pressure, 0, sizeof(float)*particleNum));
	//ComputePredictedDensityAndPressure << <gridsize_p, blocksize_p >> >(Ibuf, output);
	timer.timeAvgStart("SimulationLoop");
	while (cnt < 2 || (densityErrorLarge&&cnt<ITERATION_MAX_NUM)) {
		densityErrorLarge = true;
		//cudaMemcpy(&input2[0], Ibuf.correction_pressure, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
#ifdef TEST
		cudaMemcpy(&input1[0], Ibuf.pos_update, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		for (auto a : input1)
			cout << a.y << " ";
		cout << endl;
		cudaDeviceSynchronize();
#endif // TEST
		ComputeDij_Pj <<<gridsize_p,blocksize_p>>> (Ibuf);
#ifdef TEST
#endif // TEST

		//cudaMemcpy(&input2[0], Ibuf.ghost_volum, ghostnum * sizeof(float), cudaMemcpyDeviceToHost);

		ComputeNewPressure <<<gridsize_p, blocksize_p >>>(Ibuf);
		//Safe
		//cudaMemcpy(&input2[0], Ibuf.densityError, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&input2[0], Ibuf.correction_pressure_update, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&input2[0], Ibuf.densityError, particleNum * sizeof(float), cudaMemcpyDeviceToHost);

		////cout << "density:" << endl;
		//int ind = -1; float maxError=000.0;
		//for (int i = 0; i < particleNum; i++) {
		//	if (input2[i] > maxError) {
		//		maxError = input2[i];
		//		ind = i;
		//	}
		//}
		/*for (auto a : input2)
		cout << a << " ";
		cout << endl;*/
		//cudaDeviceSynchronize();
		//CUDA_SAFE_CALL(cudaMemset(Ibuf.densityError, 0, sizeof(float)*particleNum));



		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		float max_density_error;
		max_density_error = ReduceMax(Ibuf.densityError, Ibuf.max_predicted_density, particleNum);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		//reduceMaxI << <1, particleNum, particleNum * sizeof(float) >> > (Ibuf.densityError, Ibuf.max_predicted_density);;
		//cudaMemcpy(&max_density_error, Ibuf.max_predicted_density, sizeof(float), cudaMemcpyDeviceToHost);
		max_density_error = MAX(0, max_density_error);
		if (max_density_error / restDensity < ErrorBound)
			densityErrorLarge = false;
		std::swap(Ibuf.correction_pressure,Ibuf.correction_pressure_update);
		cnt++;
	}
	timer.timeAvgEnd("SimulationLoop",cnt);
}

void IISPH_solver::ComputeBeforIteration() {
	computeDensity << <gridsize_p, blocksize_p >> > (Ibuf);
	//cudaMemcpy(&input1[0], Ibuf.particle_density, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
	
	computeOtherForceI << <gridsize_p, blocksize_p >> >(Ibuf);
	AdvectionAndComputeDii << <gridsize_p, blocksize_p >> > (Ibuf);
	//cudaMemcpy(&input2[0], Ibuf.Dii, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);

	ComputeAdvectionDensityAndAii << <gridsize_p, blocksize_p >> > (Ibuf);
	//cudaMemcpy(&input1[0], Ibuf.advect_density, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&input1[0], Ibuf.aii, particleNum * sizeof(float), cudaMemcpyDeviceToHost);

		
}
void IISPH_solver::Advance(float3* output) {
	ComputePressureForceI << <gridsize_p, blocksize_p >> > (Ibuf, Ibuf.pos_update);
	advanceParticlesI << <gridsize_p, blocksize_p >> > (Ibuf, output);
}
void IISPH_solver::step()
{
	float3 * input, *output;
	cudaGraphicsMapResources(1, &cuda_vbo_resource[0], 0);
	cudaGraphicsMapResources(1, &cuda_vbo_resource[1], 0);


	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&input, &num_bytes,
		cuda_vbo_resource[0]);
	cudaGraphicsResourceGetMappedPointer((void **)&output, &num_bytes,
		cuda_vbo_resource[1]);
	//advectParticles(input, output);
	//cudaDeviceSynchronize();
	CountParticles(input);

	//cudaMemcpy(&input1[0], Ibuf.grid_particles_num, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int a : input1)
	//	cout << a << " ";
	//cudaDeviceSynchronize();
	IndexSort(input);
	ComputeBeforIteration();
	//ComputeOtherForce();
	RelaxedJacobiIteration(output);
	Advance(output);


	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[0], 0);
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[1], 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	swapBuff();
}
uint IISPH_solver::getParticleNum()
{
	return particleNum;
}
float IISPH_solver::getRadius() {

	return radius;
}
float IISPH_solver::getSmoothRadius() {
	return smoothRadius;
}
uint IISPH_solver::getGhostParticleNum()
{
	return ghostnum;
}






//void testf() {
//	int data[6000];
//
//
//}


