


#include "fluid_system.cuh"
//#include "device_atomic_functions.hpp"
#include <vector>
#include <thrust\reduce.h>
#include <thrust\functional.h>
#include <thrust\execution_policy.h>
#include "timer.h"

#include <iostream>
using namespace std;
extern vector<float3> boundaryWall;
extern Timer timer;

bufflist fbuf;
uint Location[2];
uint ParticleVAO[2];
//#ifndef __CUDACC__
//#define __CUDACC__

#define UNDEF_GRID -1
#define ITERATION_MAX_NUM 20000000

extern struct cudaGraphicsResource *cuda_vbo_resource[2];
namespace PCISPH {
	__constant__ ParticleParams _param;

	float radius = 0.018;
	float smoothRadius = radius * 4;
	float densityRatio = 1;   //control neighborNum
	float GridSize = smoothRadius / densityRatio;
	float3 minGridCorner = { -0.5,-0.5,-0.5 };
	float3 maxGridCorner = { 0.5,0.5,0.5 };
	float3 OuterGridRange = maxGridCorner - minGridCorner + (2.0 * smoothRadius)*make_float3(1.0, 1.0, 1.0); //FLOAT3_ADD( FLOAT3_SUB(maxGridCorner,minGridCorner),make_float3(smoothRadius*2, smoothRadius*2, smoothRadius*2));
	float3 minOuterBound;
	float3 minWaterCorner;
	float3 maxWaterCorner;
	float3 waterRange = maxWaterCorner - minWaterCorner;
	float restDensity = 1000;
	uint influcedParticleNum = 50;
	float mass = 4 * M_PI / (3 * influcedParticleNum)*pow(smoothRadius, 3.0f)*restDensity;
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
}


#define ErrorBound 0.02

using namespace PCISPH;
inline __device__ __host__ int getGridPCI(float3& pos) {
	int gridx = (pos.x - _param.minOuterBound.x) / _param._GridSize;
	int gridy = (pos.y - _param.minOuterBound.y) / _param._GridSize;
	int gridz = ((pos.z - _param.minOuterBound.z) / _param._GridSize);
	if (gridx >= 0 && gridx < _param.outerGridDim.x&&gridy >= 0 && gridy < _param.outerGridDim.y&&gridz >= 0 && gridz < _param.outerGridDim.z)
		return  gridz+gridy*_param.outerGridDim.z+gridx*_param.outerGridDim.z*_param.outerGridDim.y;
	else
		return UNDEF_GRID;

}

int getGridCpu(float3& pos) {
	int gridx = (pos.x - minOuterBound.x) / GridSize;
	int gridy = (pos.y - minOuterBound.y) / GridSize;
	int gridz = ((pos.z - minOuterBound.z) / GridSize);
	if (gridx >= 0 && gridx < outerGridDim.x&&gridy >= 0 && gridy < outerGridDim.y&&gridz >= 0 && gridz < outerGridDim.z)
		return  gridz + gridy * outerGridDim.z + gridx * outerGridDim.z*outerGridDim.y;
	else {
		cout << gridx<<' ' << gridy << ' '<<gridz << endl;
		cout << "error" <<pos.x<<' '<< pos.y << ' '<< pos.z << ' ' << endl;
		return UNDEF_GRID;
	}
		

}
float poly6kernelGradientCpu(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernelGradient*(-ratio * tmp*tmp);
}
inline __host__ __device__ float poly6kernelValPCI(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernel*(tmp*tmp*tmp);
}
inline __host__ __device__ float poly6kernelGradientPCI(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.spikykernelGradient*(-1.0 * tmp*tmp);
	//if (dist > _param.smooth_radius)
	//	return 0;
	//float ratio = dist / _param.smooth_radius;
	//float tmp = 1 - ratio * ratio;
	//return _param.poly6kernelGradient*(-ratio * tmp*tmp);
}
inline __host__ __device__ float spikykernelGradient(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.spikykernelGradient*(-1.0 * tmp*tmp);
}

__global__ void CountParticleInGrid(float3* p,bufflist fbuf) {
	//int i = threadIdx.z * 25 + threadIdx.y * 5 + threadIdx.x;
	int i =  blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 point = p[i];
	const float3 vec_bound_min = _param._minGridCorner;
	const float3 vec_bound_max = _param._maxGridCorner;
	{
		int gridIndex = getGridPCI(point);
		if (gridIndex == UNDEF_GRID)
			printf("error grid\n");
		fbuf.particle_grid_cell_index[i] = gridIndex;
		fbuf.grid_particle_offset[i] = atomicAdd(&fbuf.grid_particles_num[gridIndex], 1);
	}
	

}

void ComputeDensityErrorFactor(vector<float3>& pos, int i) {
	float GradWDot = 0;
	float3 GradW = make_float3(0,0,0);
	for (int j = 0; j < pos.size(); j++) {
		if (i == j)
			continue;
		float3 pos_i_minus_j = (pos[i]-pos[j]);
		float dist_square = dot(pos_i_minus_j,pos_i_minus_j);
		float dist = sqrtf(dist_square);
		float3 gradVec = (pos_i_minus_j* poly6kernelGradientPCI(dist)/dist);
		GradWDot += dot(gradVec, gradVec);
		GradW = (GradW+ gradVec);
	}
	
	float factor = _param.mass * _param.mass * _param.time_step * _param.time_step/(_param.rest_density*_param.rest_density);
	float gradWTerm = -dot(GradW,GradW) - GradWDot;
	_param.param_density_error_factor = -1.0 / (factor*gradWTerm);
}

//
void addPointFormTriangle(float3 p1,float3 p2,float3 p3,int lod,vector<float3>&pointSet) {
	for (int i = 0; i <= lod; i++) {
		for (int j = 0; i + j <= lod; j++) {
			pointSet.push_back(p1*(float(i)/float(lod))+p2* (float(j) / float(lod)) +p3* (float(lod-i-j) / float(lod)));
		}
	}

}
void PCISPH_solver::setUpParameter()
{
	radius = this->radius;
	smoothRadius = this->smooth_radius;
	GridSize = smoothRadius / densityRatio;
	 minGridCorner = { -0.5,-0.5,-0.5 };
	 maxGridCorner = { 0.5,0.5,0.5 };
	 OuterGridRange = maxGridCorner - minGridCorner + (2.0 * smoothRadius)*make_float3(1.0, 1.0, 1.0); //FLOAT3_ADD( FLOAT3_SUB(maxGridCorner,minGridCorner),make_float3(smoothRadius*2, smoothRadius*2, smoothRadius*2));
	 minOuterBound = { minGridCorner.x - smoothRadius,minGridCorner.y - smoothRadius,minGridCorner.z - smoothRadius };
	 minWaterCorner = { -0,-0.5,-0.5 };
	 maxWaterCorner = { 0.5,0.5,0.5 };
	 waterRange = maxWaterCorner - minWaterCorner;
	 restDensity = 1000;
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
void PCISPH_solver::particleSetUp()
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
	float offsetX = (waterRange.x-(numX-1)*initialDistance) / 2.0f;
	float offsetY = (waterRange.y - (numY - 1)*initialDistance) / 2.0f;
	float offsetZ = (waterRange.z - (numZ - 1)*initialDistance) / 2.0f;
	float3 startPos = make_float3(offsetX+minWaterCorner.x,offsetY+minWaterCorner.y,offsetZ+minWaterCorner.z);
	for(int i=0;i<numX;i++)
		for (int j = 0; j<numY; j++)
			for (int k = 0; k < numZ; k++) {
				int index = k + numZ * j + i * numZ*numY;
				float3 move = {i*initialDistance,j*initialDistance,k*initialDistance};
				float3 pos = (startPos+move);
				
				initialPos.push_back(pos);
			}
	float posx = startPos.x;
	float posy = startPos.y;
	float posz = startPos.z;

	
	ComputeDensityErrorFactor(initialPos, initialPos.size() / 2);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(_param, &_param, sizeof(ParticleParams), 0, cudaMemcpyHostToDevice));
	ghostPos=boundaryWall;
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
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_particle_offset, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_grid_cell_index, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_grid_cell_index_update, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.vel_old, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.vel_update, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.pos_update, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.sort_index, particleNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.force, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.vel_old, 0, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.correction_pressure_force, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.correction_pressure, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.max_predicted_density, sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.densityError, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.densityError, 0, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_off, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_particles_num, outerGridNum * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.grid_particles_num,0,outerGridNum*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_grid_off, outerGridNum* sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_grid_particles_num, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_volum, ghostPos.size() * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_pos, ghostPos.size()* sizeof(float3)));
	ghostnum = ghostPos.size();
	vector<int> ghost_particle_index_grid(ghostPos.size());
	vector<int> ghost_particle_grid_index(ghostPos.size());
	vector<int> sorted_index(ghostPos.size());
	vector<float3> pos_tmp(ghostPos.size());
	vector<int> ghost_grid_particles_num(outerGridNum,0);
	vector<int> ghost_grid_off(outerGridNum+1);
	vector<float> ghost_vol(ghostnum);
	for (int i = 0; i < ghostPos.size();i++) {
		float3 pos = ghostPos[i];
		int index = getGridCpu(pos);
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
		int i_cell_index = getGridCpu(pos_tmp[i]);
		int3 GridnumRange = _param.outerGridDim;
		int cell_z = i_cell_index % (GridnumRange.z);
		i_cell_index /= GridnumRange.z;
		int cell_y = i_cell_index % (GridnumRange.y);
		int cell_x = i_cell_index / GridnumRange.y;
		//int index = getGridCpu(ghostPos[i]);
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
			int ghost_cell_end = ghost_grid_off[neighbor_cell_index+1];
			for (int cndx = ghost_cell_start; cndx < ghost_cell_end; cndx++)
			{
				//force.y++;
				int j = cndx;
				float3 vector_i_minus_j = (pos_tmp[i]- pos_tmp[j]);
				const float jdist = length(vector_i_minus_j);
				if (jdist < _param.smooth_radius)
				{
					//float jdist = sqrt(dist_square);
					float kernel = poly6kernelValPCI(jdist);
					Wsum += kernel*_param.mass/_param.rest_density;
				}
			}
		}
		ghost_vol[i] = 1.5/ Wsum;
	}
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_volum, &ghost_vol[0], sizeof(float)*ghostnum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_grid_off,&ghost_grid_off[0],sizeof(int)*outerGridNum,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_grid_particles_num, &ghost_grid_particles_num[0], sizeof(int)*outerGridNum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_pos, &pos_tmp[0], sizeof(float3)*ghostPos.size(), cudaMemcpyHostToDevice));
	ghostnum = ghostPos.size();

}


//helper function


GLuint PCISPH_solver::getRenderVBO()
{
	return ParticleVAO[0];
}
void swapBuff() {
	swap(cuda_vbo_resource[0], cuda_vbo_resource[1]);
	swap(Location[0], Location[1]);
	swap(ParticleVAO[0], ParticleVAO[1]);
	swap(fbuf.vel_old, fbuf.vel_update);
	//swap(fbuf.particle_grid_cell_index, fbuf.particle_grid_cell_index_update);
}
void CountParticles(float3* input) {
	//dim3 threadperBlock(numRangeX,numRangeY ,numRangeZ);
	CUDA_SAFE_CALL(cudaMemset(fbuf.grid_particles_num, 0, outerGridNum*sizeof(uint) ));
	//cudaMemset(fbuf.max_predicted_density, 0, sizeof(float));

	CountParticleInGrid <<<gridsize_p, blocksize_p >> > (input, fbuf);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaDeviceSynchronize();
}

__global__ void rearrange(bufflist fbuf,float3* pos_old) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	uint index = fbuf.sort_index[tid];
	fbuf.vel_update[index] = fbuf.vel_old[tid];
	fbuf.pos_update[index] = pos_old[tid];
	fbuf.particle_grid_cell_index_update[index] = fbuf.particle_grid_cell_index[tid];
}
__global__ void sortIndex(bufflist fbuf) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	uint cell_index = fbuf.particle_grid_cell_index[tid];
	uint particle_index = fbuf.grid_off[cell_index] + fbuf.grid_particle_offset[tid];
	fbuf.sort_index[tid] = particle_index;
}
__global__ void computeOtherForce(bufflist fbuf) {
	//Gravity
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 nonPressureforce = { 0,-_param.gravity,0 };
	//Viscosity
	float3 & ipos = fbuf.pos_update[tid];
	float3 & ivel = fbuf.vel_update[tid];
	uint i_cell_index = fbuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	float smooth_radius_square = _param.smooth_radius * _param.smooth_radius;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
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
		int cell_start = fbuf.grid_off[neighbor_cell_index];
		int cell_end = cell_start + fbuf.grid_particles_num[neighbor_cell_index];
		for (int j = cell_start; j < cell_end; j++)
		{
			////force.y++;
			if (tid == j)
			{
				continue;
			}
			float3 vector_i_minus_j = (ipos - fbuf.pos_update[j]);
			
			const float dist = length(vector_i_minus_j);
			if (dist < _param.smooth_radius)
			{
				float3 vel_i_minus_j = ivel - fbuf.vel_update[j];
				//float jdist = sqrt(dist_square);
				float kernelGradientValue = poly6kernelGradientPCI(dist);
				float3 kernelGradient = (vector_i_minus_j * kernelGradientValue / dist);
				nonPressureforce += .2*(_param.mass*_param.mass / _param.rest_density)*
					dot(vector_i_minus_j, vel_i_minus_j) / (0.01*smooth_radius_square + dist * dist)*kernelGradient;
			}
		}
		int ghost_cell_start = fbuf.ghost_grid_off[neighbor_cell_index];
		int ghost_cell_end = ghost_cell_start + fbuf.ghost_grid_particles_num[neighbor_cell_index];
		for (int j = ghost_cell_start; j < ghost_cell_end; j++)
		{
			
			float3 vector_i_minus_j = ipos - fbuf.ghost_pos[j];
			const float dist = length(vector_i_minus_j);

			if (dist < _param.smooth_radius)
			{
				float3 vel_i_minus_j = ivel;
				//float jdist = sqrt(dist_square);
				float kernelGradientValue = poly6kernelGradientPCI(dist);
				float3 kernelGradient = (vector_i_minus_j * kernelGradientValue / dist);
				nonPressureforce += 0.0*(_param.mass*_param.mass / _param.rest_density)*
				dot(vector_i_minus_j, vel_i_minus_j)*fbuf.ghost_volum[j] / (0.01*smooth_radius_square + dist * dist)*kernelGradient;
			}
		}
	}
	fbuf.force[tid] = nonPressureforce;
	
}


__global__ void PredictPosition(bufflist fbuf,float3 * output_pos) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;
	//const float	   sim_scale = simData.param_sim_scale;
	
	//float3 acceleration = FLOAT3_MUL_SCALAR(FLOAT3_ADD(fbuf.force[tid], fbuf.correction_pressure_force[tid]), (1.0f / _param.mass));
	float3 acceleration = (fbuf.force[tid]+ fbuf.correction_pressure_force[tid])* (1.0f / _param.mass);

	float3 predictedVelocity = (fbuf.vel_update[tid] + (acceleration* _param.time_step));
	float3 pos = (fbuf.pos_update[tid] + predictedVelocity * _param.time_step);

	output_pos[tid] = pos;
}
__global__ void ComputePredictedDensityAndPressure(bufflist fbuf,float3* predicted_pos) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;

	uint i_cell_index = fbuf.particle_grid_cell_index_update[tid];
	int3 GridnumRange = _param.outerGridDim;
	int cell_z = i_cell_index % (GridnumRange.z);
	i_cell_index /= GridnumRange.z;
	int cell_y = i_cell_index % (GridnumRange.y);
	int cell_x = i_cell_index / GridnumRange.y;

	const float3 ipredicted_pos = predicted_pos[tid];
	const float  smooth_radius = _param.smooth_radius;
	const float  smooth_radius_square = smooth_radius * smooth_radius;
	//const float  sim_scale_square = simData.param_sim_scale * simData.param_sim_scale;
	const float  mass = _param.mass;
	float predictedSPHDensity = 0.0;
	for (int cell = 0; cell < neighborGridNum; cell++)
	{	
		int cell_neighbor_x = cell_x + _param._neighbor_off[cell].x;
		int cell_neighbor_y = cell_y + _param._neighbor_off[cell].y;
		int cell_neighbor_z = cell_z + _param._neighbor_off[cell].z;
		if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_y < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
			continue;
		int neighbor_cell_index=cell_neighbor_z+ cell_neighbor_y* GridnumRange.z+ cell_neighbor_x* GridnumRange.z*GridnumRange.y;
		// real water particles
		if (fbuf.grid_particles_num[neighbor_cell_index] != 0)
		{
			int cell_start = fbuf.grid_off[neighbor_cell_index];
			int cell_end = cell_start + fbuf.grid_particles_num[neighbor_cell_index];

			for (int cndx = cell_start; cndx < cell_end; cndx++)
			{
				int j = cndx;
				if (tid == j)
				{
					continue;
				}
				float3 vector_i_minus_j = (ipredicted_pos- predicted_pos[j]);
				const float dx = vector_i_minus_j.x;
				const float dy = vector_i_minus_j.y;
				const float dz = vector_i_minus_j.z;
				const float dist_square_scale = dx * dx + dy * dy + dz * dz;
				if (dist_square_scale <= smooth_radius_square && dist_square_scale > 0)
				{
					//predictedSPHDensity += 1;
					const float dist = sqrt(dist_square_scale);
					float kernelValue = poly6kernelValPCI(dist);
					predictedSPHDensity += kernelValue * mass;
					//predictedSPHDensity += 1;
				}
			}
		}
		//ghost particles
		if (fbuf.ghost_grid_particles_num[neighbor_cell_index] > 0) {
			int cell_start = fbuf.ghost_grid_off[neighbor_cell_index];
			int cell_end = cell_start + fbuf.ghost_grid_particles_num[neighbor_cell_index];

			for (int cndx = cell_start; cndx < cell_end; cndx++)
			{
				int j = cndx;
				float3 vector_i_minus_j = (ipredicted_pos- fbuf.ghost_pos[j]);
				const float dist_square_scale = dot(vector_i_minus_j,vector_i_minus_j);
				if (dist_square_scale <= smooth_radius_square && dist_square_scale > 0)
				{
					//predictedSPHDensity += 1;
					/*if (tid == 0)
						printf("not correct x: %f,y: %f,z: %f\nself x:%f,y:%f,z:%f\n",
							fbuf.ghost_pos[j].x,fbuf.ghost_pos[j].y, fbuf.ghost_pos[j].z,ipredicted_pos.x,ipredicted_pos.y,ipredicted_pos.z);*/
					const float dist = sqrt(dist_square_scale);
					float kernelValue = poly6kernelValPCI(dist) *fbuf.ghost_volum[j] ;
					predictedSPHDensity += kernelValue * mass;
					//predictedSPHDensity = 10010.0;
					//predictedSPHDensity += 1;
				}
			}
		}

		
	}

	predictedSPHDensity += _param.poly6kernel* mass;

	 float densityError= MAX(predictedSPHDensity-_param.rest_density,0.0 );
	 fbuf.densityError[tid] = densityError;
	// fbuf.test_buff[tid].x = predictedSPHDensity;
	fbuf.correction_pressure[tid] += densityError*_param.param_density_error_factor;

}
__global__ void ComputePressureForce(bufflist fbuf,float3* predicted_pos) {
		int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
		if (tid >= _param.particleNum)
			return;

		uint i_cell_index = fbuf.particle_grid_cell_index_update[tid];
		int3 GridnumRange = _param.outerGridDim;
		int cell_z = i_cell_index % (GridnumRange.z);
		i_cell_index /= GridnumRange.z;
		int cell_y = i_cell_index % (GridnumRange.y);
		int cell_x = i_cell_index / GridnumRange.y;
		if (i_cell_index == UNDEF_GRID)
			return;

		const float3 ipos = fbuf.pos_update[tid];
		const float  ipress = fbuf.correction_pressure[tid];
		const float  mass = _param.mass;
		const float  smooth_radius = _param.smooth_radius;
		const float  smooth_radius_square = smooth_radius * smooth_radius;
		const float  rest_volume = mass / _param.rest_density;
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

			int cell_start = fbuf.grid_off[neighbor_cell_index];
			int cell_end = cell_start + fbuf.grid_particles_num[neighbor_cell_index];
			for (int cndx = cell_start; cndx < cell_end; cndx++)
			{
				//force.y++;
				int j = cndx;
				if (tid == j)
				{
					continue;
				}
				float3 vector_i_minus_j = (ipos - fbuf.pos_update[j]);

				const float dist_square = dot(vector_i_minus_j, vector_i_minus_j); 
				if (dist_square < smooth_radius_square && dist_square > 0)
				{
					float jdist = sqrt(dist_square);
					float kernelGradientValue = poly6kernelGradientPCI(jdist);
					float3 kernelGradient = ( vector_i_minus_j * kernelGradientValue/jdist);
					float grad = 0.5f * (ipress + fbuf.correction_pressure[j]) * rest_volume * rest_volume;
					force -= kernelGradient * grad;
				}
			}
			int ghost_cell_start = fbuf.ghost_grid_off[neighbor_cell_index];
			int ghost_cell_end = ghost_cell_start + fbuf.ghost_grid_particles_num[neighbor_cell_index];
			for (int cndx = ghost_cell_start; cndx < ghost_cell_end; cndx++)
			{
				int j = cndx;
				float3 vector_i_minus_j = ipos- fbuf.ghost_pos[j];
				const float dist = length(vector_i_minus_j);

				if (dist < smooth_radius && dist > 0)
				{
					float kernelGradientValue = poly6kernelGradientPCI(dist);
					float3 kernelGradient = vector_i_minus_j*(( kernelGradientValue/ dist)*fbuf.ghost_volum[j]);
					float grad = 0.5f * (ipress) * rest_volume * rest_volume;
					forceB -= kernelGradient * grad;
				}
			}
		}

		fbuf.correction_pressure_force[tid] = force+forceB;
	
}
__global__ void scanSumInt(int * input, int * output, int *aux, int numPerthread) {
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
__global__ void reduceMax(float *g_idata, float *g_odata,int num) {

	extern __shared__ float sdata[];
	//// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	//unsigned int i =  threadIdx.x;
	if(tid<num)
	sdata[tid] = g_idata[tid];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = num; s > 1; s = (s+1)/2) {
		//s is odd
		if (s % 2)
		{
			if (tid < s / 2)
				sdata[tid] = MAX(sdata[tid], sdata[tid + s / 2]);
			else if (tid == s / 2)
				sdata[tid] = sdata[s-1];
		}
		else
		if (tid < s/2) {
			sdata[tid] =MAX(sdata[tid], sdata[tid + s/2]);
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[0] = sdata[0];
}
__global__ void advanceParticles(bufflist fbuf,float3* output) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;
	float3 acceleration = (1.0 / _param.mass)*(fbuf.force[tid] + fbuf.correction_pressure_force[tid]);
	float3 veval = fbuf.vel_update[tid];
	veval += acceleration * _param.time_step;
	float3 pos = fbuf.pos_update[tid];
	pos += veval * _param.time_step;

	output[tid] = pos;
	fbuf.vel_update[tid] = (veval);
}
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
float ReduceMax(float* input, float* output, int num) {
	thrust::device_ptr<float> data(input);
	float res= thrust::reduce(data, data + num
		,-1.0,
		thrust::maximum<float>()
	);
	return res;
	dim3 blocksize(ceil(sqrt(num)), ceil(sqrt(num)));
	reduceMax << <1, blocksize, num * sizeof(float) >> > (input, output,num);
	cudaDeviceSynchronize();
	float max_density_error;
	cudaMemcpy(&max_density_error, fbuf.max_predicted_density, sizeof(float), cudaMemcpyDeviceToHost);
	////Safe
		//float res = *max_density_error;
	//delete max_density_error;
		return max_density_error;

}
__global__ void addUpPrefix(int* prefix, int* aux, int stride) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	prefix[id] += aux[id / stride];
}

void prescanInt(int* input, int* output, int len, int numPerThread, int numBlock, int numThread) {
	int * aux;
	//auto input1 = vector<int>(GridNum);

	int totalthread = (numBlock*numThread);
	CUDA_SAFE_CALL(cudaMalloc(&aux, totalthread * sizeof(int)));
	cudaDeviceSynchronize();
	//cudaMemcpy(&input1[0], input, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	scanSumInt << <numBlock, numThread >> > (input, output, aux, numPerThread);


	scanSumInt << <1, 1 >> > (aux, aux, NULL, totalthread);  //in place 
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	addUpPrefix << <numBlock*numThread, numPerThread >> >(output, aux, numPerThread);
	CUDA_SAFE_CALL(cudaFree(aux));
	//cudaMemcpy(&input1[0], output, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	//
	//
	//for (int a : input1)
	//	cout << a << " ";
	//cudaDeviceSynchronize();
}
void IndexSort(float3* pos_old) {
	
	prescanInt(fbuf.grid_particles_num, fbuf.grid_off,outerGridNum, outerGridDim.x, outerGridDim.y, outerGridDim.z);
	//dim3 blocksize(particleNum);

	auto input1 = vector<float3>(particleNum);
	auto input2 = vector<int>(particleNum);
	//sortIndex << <1, particleNum >> >(fbuf.grid_off, fbuf.grid_particle_offset, fbuf.particle_grid_cell_index, fbuf.sort_index);

	
	cudaDeviceSynchronize();
	sortIndex<<<gridsize_p, blocksize_p >>>(fbuf);
	//Safe
	cudaDeviceSynchronize();

	rearrange<<<gridsize_p, blocksize_p >>>(fbuf,pos_old);

	
	cudaDeviceSynchronize();
}
void ComputeOtherForce() {    //grivty
//	dim3 blockSize();
	computeOtherForce<<<gridsize_p,blocksize_p>>>(fbuf);
	//Safe
	auto input1 = vector<float3>(particleNum);
}
void RelaxedJacobiIteration(float3* output) {
	bool densityErrorLarge=true;
	int cnt = 0;
	CUDA_SAFE_CALL(cudaMemset(fbuf.correction_pressure,0, sizeof(float)*particleNum));
	ComputePredictedDensityAndPressure << <gridsize_p, blocksize_p >> >(fbuf, output);
	timer.timeAvgStart("SimulationLoop");
	while (cnt < 1||(densityErrorLarge&&cnt<ITERATION_MAX_NUM)) {
		//auto input1 = vector<float3>(particleNum);
		//auto input2 = vector<float>(particleNum);

		//cudaDeviceSynchronize();
#ifdef TEST
		cudaMemcpy(&input1[0], fbuf.pos_update, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		for (auto a : input1)
			cout << a.y << " ";
		cout << endl;
		cudaDeviceSynchronize();
#endif // TEST
		PredictPosition<<<gridsize_p,blocksize_p>>>(fbuf,output);
#ifdef TEST
#endif // TEST

		//cudaMemcpy(&input2[0], fbuf.ghost_volum, ghostnum * sizeof(float), cudaMemcpyDeviceToHost);

		ComputePredictedDensityAndPressure<<<gridsize_p,blocksize_p>>>(fbuf,output);
		//Safe
		//cudaMemcpy(&input2[0], fbuf.densityError, particleNum * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&input1[0], output, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
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
		//CUDA_SAFE_CALL(cudaMemset(fbuf.densityError, 0, sizeof(float)*particleNum));


		
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		float max_density_error;
		max_density_error = ReduceMax(fbuf.densityError, fbuf.max_predicted_density,particleNum);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		//reduceMax << <1, particleNum, particleNum * sizeof(float) >> > (fbuf.densityError, fbuf.max_predicted_density);;
		//cudaMemcpy(&max_density_error, fbuf.max_predicted_density, sizeof(float), cudaMemcpyDeviceToHost);
		max_density_error = MAX(0, max_density_error);
		if (max_density_error / restDensity < ErrorBound)
			densityErrorLarge = false;
		ComputePressureForce << <gridsize_p, blocksize_p >> >(fbuf,output);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		cnt++;
	}
	timer.timeAvgEnd("SimulationLoop",cnt);
}
void Advance(float3* output) {
	advanceParticles << <gridsize_p, blocksize_p >> > (fbuf,output);
}
void PCISPH_solver::step()
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
	auto input1 = vector<int>(outerGridNum);

	//cudaMemcpy(&input1[0], fbuf.grid_particles_num, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int a : input1)
	//	cout << a << " ";
	//cudaDeviceSynchronize();
	IndexSort(input);
	ComputeOtherForce();
	RelaxedJacobiIteration(output);
	Advance(output);


	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[0], 0);
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[1], 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	swapBuff();
}
uint PCISPH_solver::getParticleNum()
{
	return particleNum;
}
float PCISPH_solver::getRadius() {

	return radius;
}
float PCISPH_solver::getSmoothRadius() {
	return smoothRadius;
}
uint PCISPH_solver::getGhostParticleNum()
{
	return ghostnum;
}






void testf() {
	int data[6000];
	
}


