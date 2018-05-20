


#include "fluid_system.cuh"
//#include "device_atomic_functions.hpp"
#include <vector>
#include <thrust\reduce.h>
#include <thrust\functional.h>
#include <thrust\execution_policy.h>


#include <iostream>
using namespace std;



bufflist fbuf;
uint Location[2];
uint ParticleVAO[2];
//#ifndef __CUDACC__
//#define __CUDACC__

#define UNDEF_GRID -1
#define ITERATION_MAX_NUM 20000000
extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern struct cudaGraphicsResource *cuda_vbo_resource[2];
 __constant__ ParticleParams _param;
 struct ParticleParams cpuParam;
//__device__ __constant__ float deviceArray[10];

dim3 blocksize_p(ceil(waterRange.x / initialDistance), ceil(waterRange.y / initialDistance));
dim3 gridsize_p(ceil(waterRange.z / initialDistance)); //1 dimension
int ghostnum;
//dim3 blocksize_grid(outerGridDim.x, outerGridDim.y);
dim3 gridsize_grid(outerGridDim.z);

//#define TEST

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	Timer timer;
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;
	timer.start();
	cudaStatus = cudaSetDevice(0);
	timer.stop();
	printf("time: %d ms\n", timer.duration());
	// Choose which GPU to run on, change this on a multi-GPU system.
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	timer.start();

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	timer.stop();
	printf("time: %d ms\n", timer.duration());
	

	// Launch a kernel on the GPU with one thread for each element
	
	timer.start();
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	timer.stop();
	printf("time: %d ms\n", timer.duration());

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

__device__ int getGrid(float3& pos) {
	int gridx = (pos.x - _param.minOuterBound.x) / _param._GridSize;
	int gridy = (pos.y - _param.minOuterBound.y) / _param._GridSize;
	int gridz = ((pos.z - _param.minOuterBound.z) / _param._GridSize);
	if (gridx >= 0 && gridx < _param.outerGridDim.x&&gridy >= 0 && gridy < _param.outerGridDim.y&&gridz >= 0 && gridz < _param.outerGridDim.z)
		return  gridz+gridy*_param.outerGridDim.z+gridx*_param.outerGridDim.z*_param.outerGridDim.y;
	else
		return UNDEF_GRID;

}
__device__ inline void boxBoundaryForce(const float3& position, float3& force)
{
	const float  sim_scale = 1;
	const float3 vec_bound_min = _param._minGridCorner;
	const float3 vec_bound_max = _param._maxGridCorner ;
	float  param_force_distance = 0.015;
	float param_max_boundary_force = 2.0;
	float param_inv_force_distance = 1.0/param_force_distance;
	if (position.x < vec_bound_min.x + param_force_distance)
	{
		float3 boundForce = make_float3(1.0, 0.0, 0.0)*((position.z + param_force_distance - vec_bound_max.z)
			* param_inv_force_distance * 2.0 * param_max_boundary_force);
		force += boundForce;
	}
	if (position.x > vec_bound_max.x - param_force_distance)
	{
		float3 boundForce = make_float3(-1.0, 0.0, 0.0)*((position.z + param_force_distance - vec_bound_max.z)
			* param_inv_force_distance * 2.0 * param_max_boundary_force);
		force += boundForce;
	}

	if (position.y < vec_bound_min.y + param_force_distance)
	{
		float3 boundForce = make_float3(0.0, 1.0, 0.0)*((position.z + param_force_distance - vec_bound_max.z)
			* param_inv_force_distance * 2.0 * param_max_boundary_force);
		force += boundForce;
	}
	if (position.y > vec_bound_max.y - param_force_distance)
	{
		float3 boundForce = make_float3(0.0, -1.0, 0.0)*((position.z + param_force_distance - vec_bound_max.z)
			* param_inv_force_distance * 2.0 * param_max_boundary_force);
		force += boundForce;
	}

	if (position.z < vec_bound_min.z + param_force_distance)
	{
		float3 boundForce = make_float3(0.0, 0.0, 1.0)*((position.z + param_force_distance - vec_bound_max.z)
			* param_inv_force_distance * 2.0 * param_max_boundary_force); 
		force += boundForce;
	}

	if (position.z > vec_bound_max.z - param_force_distance)
	{
		float3 boundForce = make_float3(0.0,0.0,-1.0)*((position.z + param_force_distance - vec_bound_max.z) 
			* param_inv_force_distance * 2.0 * param_max_boundary_force); 
		force += boundForce;
	}
}
int getGridCpu(float3& pos) {
	int gridx = (pos.x - minOuterBound.x) / GridSize;
	int gridy = (pos.y - minOuterBound.y) / GridSize;
	int gridz = ((pos.z - minOuterBound.z) / GridSize);
	if (gridx >= 0 && gridx < outerGridDim.x&&gridy >= 0 && gridy < outerGridDim.y&&gridz >= 0 && gridz < outerGridDim.z)
		return  gridz + gridy * outerGridDim.z + gridx * outerGridDim.z*outerGridDim.y;
	else
		return UNDEF_GRID;

}
float poly6kernelGradientCpu(float dist) {
	if (dist > cpuParam.smooth_radius)
		return 0;
	float ratio = dist / cpuParam.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return cpuParam.poly6kernelGradient*(-ratio * tmp*tmp);
}

__global__ void CountParticleInGrid(float3* p,bufflist fbuf) {
	//int i = threadIdx.z * 25 + threadIdx.y * 5 + threadIdx.x;
	int i =  blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	float3 point = p[i];
	const float3 vec_bound_min = _param._minGridCorner;
	const float3 vec_bound_max = _param._maxGridCorner;
	if (point.x<vec_bound_min.x || point.x>vec_bound_max.x ||
		point.y<vec_bound_min.y || point.y>vec_bound_max.y ||
		point.z<vec_bound_min.z || point.x>vec_bound_max.z) {
		//atomicAdd(&fbuf.max_predicted_density[0], 1);
	}
	else {
		int gridIndex = getGrid(point);
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
		float3 gradVec = (pos_i_minus_j* poly6kernelGradientCpu(dist)/dist);
		GradWDot += dot(gradVec, gradVec);
		GradW = (GradW+ gradVec);
	}
	
	float factor = cpuParam.mass * cpuParam.mass * cpuParam.time_step * cpuParam.time_step/(cpuParam.rest_density*cpuParam.rest_density);
	float gradWTerm = -dot(GradW,GradW) - GradWDot;
	cpuParam.param_density_error_factor = -1.0 / (factor*gradWTerm);
}

//

void Setup() {
	//CUDA_SAFE_CALL(cudaMalloc((void**)&testdata, 9*sizeof(float)));
	//memset(testdata, 0, 9 * sizeof(float));
	glGenBuffers(2, &Location[0]);
	glGenVertexArrays(2, &ParticleVAO[0]);
	vector<float3> initialPos = vector<float3>();
	vector<float3> ghostPos = vector<float3>();
	int total = particleNum;
	int numX = ceil(waterRange.x / initialDistance);
	int numY = ceil(waterRange.y / initialDistance);
	int numZ = ceil(waterRange.z / initialDistance);
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
	posx = posx - initialDistance*(int)((posx - minGridCorner.x) / initialDistance);
	posy = posz - initialDistance * (int)((posy - minGridCorner.y) / initialDistance);
	posz = posz - initialDistance * (int)((posz - minGridCorner.z) / initialDistance);

	while (true)
	{
		if (posx - initialDistance > minOuterBound.x)
			posx -= initialDistance;
		else break;
	}
	while (true)
	{
		if (posy - initialDistance > minOuterBound.y)
			posy -= initialDistance;
		else break;
	}
	while (true)
	{
		if (posz - initialDistance > minOuterBound.z)
			posz -= initialDistance;
		else break;
	}
	for (float px=posx; px < maxGridCorner.x + smoothRadius; px += initialDistance) {
		for (float py=posy; py < maxGridCorner.y + smoothRadius; py += initialDistance) {
			for (float pz=posz ; pz < maxGridCorner.z + smoothRadius; pz += initialDistance) {
				if (px > minGridCorner.x  && px<maxGridCorner.x  &&
					py>minGridCorner.y  && py<maxGridCorner.x  &&
					pz>minGridCorner.z && pz < maxGridCorner.x)
					continue;

				ghostPos.push_back({ px,py,pz });
			}
		}
	}	
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
	//cudaMemcpyToSymbol(&_param, &cpuParam, sizeof(cpuParam), 0);
	//__constant__ ParticleParams* _param1;
	uint index = 0;
	float gravity=9;
	//float mass = 1.0;
	float poly6kernel = 315.0f / (64.0f*M_PI*pow(smoothRadius, 3));
	float poly6kernelGrad = 945.0f / (32.0f*M_PI*pow(smoothRadius, 4));
	float boudary_force_factor = 25;
	float time_step = 0.0015;
	
	//{ minGridCorner ,maxGridCorner,GridIndexRange,GridSize,particleNum,gravity,mass,time_step,smoothRadius,restDensity,poly6kernel,poly6kernelGrad,density_error_factor,boudary_force_factor} ;
	cpuParam._minGridCorner = minGridCorner;
	cpuParam._maxGridCorner = maxGridCorner;
	cpuParam._GridSize = GridSize;
	cpuParam.outerGridDim = outerGridDim;
	cpuParam.particleNum = particleNum;
	cpuParam.gravity = gravity;
	cpuParam.mass = mass;
	cpuParam.time_step = time_step;
	cpuParam.smooth_radius = smoothRadius;
	cpuParam.rest_density = restDensity;
	cpuParam.poly6kernel = poly6kernel;
	cpuParam.poly6kernelGradient = poly6kernelGrad;
	//cpuParam.param_density_error_factor = density_error_factor;
	cpuParam.minOuterBound =minOuterBound;
	//cpuParam.time_step = time_step;


	for (int i = -1; i < 2; i++)
		for (int j = -1; j < 2; j++)
			for (int k = -1; k < 2; k++)
				cpuParam._neighbor_off[index++] = { k,j,i };
	ComputeDensityErrorFactor(initialPos,initialPos.size()/2);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(_param, &cpuParam, sizeof(ParticleParams), 0, cudaMemcpyHostToDevice));
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
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.predicted_density, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.correction_pressure_force, particleNum * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.correction_pressure, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.max_predicted_density, sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.densityError, particleNum * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.densityError, 0, particleNum * sizeof(float)));
	

	//CUDA_SAFE_CALL(cudaMalloc())
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_off, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_particles_num, outerGridNum * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.grid_particles_num,0,outerGridNum*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_grid_off, outerGridNum* sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_grid_particles_num, outerGridNum * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.ghost_pos, ghostPos.size()* sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.test_buff, particleNum * sizeof(float3)))
	vector<int> ghost_particle_index_grid(ghostPos.size());
	vector<int> ghost_particle_grid_index(ghostPos.size());
	vector<int> sorted_index(ghostPos.size());
	vector<float3> pos_tmp(ghostPos.size());
	vector<int> ghost_grid_particles_num(outerGridNum,0);
	vector<int> ghost_grid_off(outerGridNum);
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
	for (int i = 0; i < ghostPos.size(); i++) {
		int cell_index = ghost_particle_grid_index[i];
		sorted_index[i] = ghost_grid_off[cell_index] + ghost_particle_index_grid[i];
	}
	for (int i = 0; i < ghostPos.size(); i++) {
		int index = sorted_index[i];
		pos_tmp[index] = ghostPos[i];
	}
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_grid_off,&ghost_grid_off[0],sizeof(int)*outerGridNum,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_grid_particles_num, &ghost_grid_particles_num[0], sizeof(int)*outerGridNum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.ghost_pos, &pos_tmp[0], sizeof(float3)*ghostPos.size(), cudaMemcpyHostToDevice));
	ghostnum = ghostPos.size();

}

void ClearSystem() {

}
void computeCUDAGridBlockSize(int numParticles, int blockSize, int &numBlocks, int &numThreads);
extern "C"
void ParticleSetupCUDA(int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk, float grid_cell_size, float param_kernel_self)
{
	

	//deallocBlockSumsInt();
	//preallocBlockSumsInt(fcudaParams.param_grid_total);
}
//helper function
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeCUDAGridBlockSize(int numParticles, int blockSize, int &numBlocks, int &numThreads)
{
	numThreads = min(blockSize, numParticles);
	numBlocks = iDivUp(numParticles, numThreads);
}
uint getLocation() {
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
	auto input2 = vector<int>(outerGridNum);
	float* input1 = new float(0);
#ifdef TEST
	cudaMemcpy(&input1[0], fbuf.max_predicted_density, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "error" << *input1 << endl;
	cudaMemcpy(&input2[0], fbuf.grid_particles_num, outerGridNum * sizeof(int), cudaMemcpyDeviceToHost);

	for (auto a : input2)
		cout << a << " ";
	cout << endl;
#endif // TEST

	
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
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	fbuf.force[tid] = { 0,-_param.gravity,0 };
	boxBoundaryForce(fbuf.pos_update[tid], fbuf.force[tid]);
}
__device__ void collisionHandling(float3* pos,float3* vel) {
	const float3 vec_bound_min = _param._minGridCorner;
	const float3 vec_bound_max = _param._maxGridCorner;
	float damping = 0.9;

	float reflect = 1.1;
		if (pos->x < vec_bound_min.x)
		{
			pos->x = vec_bound_min.x;
			if (vel) {
				float3 axis = make_float3(-1, 0, 0);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->x *= damping;
			}
		}
		if (pos->x > vec_bound_max.x)
		{
			pos->x = vec_bound_max.x;
			if (vel) {
				float3 axis = make_float3(1, 0, 0);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->x *= damping;
			}

		}
		if (pos->y < vec_bound_min.y)
		{
			pos->y = vec_bound_min.y;
			if (vel) {
				float3 axis = make_float3(0, -1, 0);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->y *= damping;
			}
		
		}
		if (pos->y > vec_bound_max.y)
		{
			pos->y = vec_bound_max.y;
			if (vel) {
				float3 axis = make_float3(0, 1, 0);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->y *= damping;
			}
			
		}
		if (pos->z < vec_bound_min.z)
		{
			pos->z = vec_bound_min.z;
			if (vel) {
				float3 axis = make_float3(0, 0, -1);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->z *= damping;
			}

		}
		if (pos->z > vec_bound_max.z)
		{
			pos->z = vec_bound_max.z;
			if (vel) {
				float3 axis = make_float3(0, 0, 1);
				*vel -= axis * (dot(axis, *vel)*reflect);
				vel->z *= damping;
			}
		}
		if (vel)
			*vel += 0.99;
	
}
__device__ float poly6kernel(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernel*(tmp*tmp*tmp);
}
__device__ float poly6kernelGradient(float dist) {
	if (dist > _param.smooth_radius)
		return 0;
	float ratio = dist / _param.smooth_radius;
	float tmp = 1 - ratio * ratio;
	return _param.poly6kernelGradient*(-ratio*tmp*tmp);
}
__global__ void PredictPosition(bufflist fbuf,float3 * output_pos) {
	int tid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tid >= _param.particleNum)
		return;
	//const float	   sim_scale = simData.param_sim_scale;
	;
	float3 acceleration = (fbuf.force[tid]+ fbuf.correction_pressure_force[tid])* (1.0f / _param.mass);
	float3 predictedVelocity = (fbuf.vel_update[tid] + (acceleration* _param.time_step));

	float3 pos = (fbuf.pos_update[tid] + (predictedVelocity* _param.time_step));

	collisionHandling(&pos, NULL);

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
		if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_x < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
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
					float kernelValue = poly6kernel(dist);
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
				const float dx = vector_i_minus_j.x;
				const float dy = vector_i_minus_j.y;
				const float dz = vector_i_minus_j.z;
				const float dist_square_scale = dx * dx + dy * dy + dz * dz;
				if (dist_square_scale <= smooth_radius_square && dist_square_scale > 0)
				{
					//predictedSPHDensity += 1;
					const float dist = sqrt(dist_square_scale);
					float kernelValue = poly6kernel(dist);
					predictedSPHDensity += kernelValue * mass;
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

	fbuf.predicted_density[tid] = predictedSPHDensity;
	//get max Error;
	//atomicMax((double*)fbuf.max_predicted_density,(double)densityError);
}
__global__ void ComputePressureForce(bufflist fbuf) {
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

		for (int cell = 0; cell < neighborGridNum; cell++)
		{
			int cell_neighbor_x = cell_x + _param._neighbor_off[cell].x;
			int cell_neighbor_y = cell_y + _param._neighbor_off[cell].y;
			int cell_neighbor_z = cell_z + _param._neighbor_off[cell].z;
			if (cell_neighbor_x < 0 || cell_neighbor_x >= _param.outerGridDim.x || cell_neighbor_x < 0 || cell_neighbor_y >= _param.outerGridDim.y || cell_neighbor_z < 0 || cell_neighbor_z >= _param.outerGridDim.z)
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
					float kernelGradientValue = poly6kernelGradient(jdist);
					float3 kernelGradient = ( vector_i_minus_j * kernelGradientValue/jdist);
					float grad = 0.5f * (ipress + fbuf.correction_pressure[j]) * rest_volume * rest_volume;
					//force =FLOAT3_SUB(force, FLOAT3_MUL_SCALAR(kernelGradient, grad));
					force -= kernelGradient * grad;
					//force.y++;
				}
			}
			int ghost_cell_start = fbuf.ghost_grid_off[neighbor_cell_index];
			int ghost_cell_end = cell_start + fbuf.ghost_grid_particles_num[neighbor_cell_index];
			for (int cndx = cell_start; cndx < cell_end; cndx++)
			{
				//force.y++;
				int j = cndx;
				float3 vector_i_minus_j = (ipos- fbuf.ghost_pos[j]);
				const float dx = vector_i_minus_j.x;
				const float dy = vector_i_minus_j.y;
				const float dz = vector_i_minus_j.z;
				const float dist_square = (dx*dx + dy * dy + dz * dz);

				if (dist_square < smooth_radius_square && dist_square > 0)
				{
					float jdist = sqrt(dist_square);
					float kernelGradientValue = poly6kernelGradient(jdist);
					float3 kernelGradient = (vector_i_minus_j* kernelGradientValue/jdist);
					float grad = 0.5f * (ipress) * rest_volume * rest_volume;
					//force = (force- FLOAT3_MUL_SCALAR(kernelGradient, grad));
					force -= kernelGradient * grad;
					
				}
			}
		}

		fbuf.correction_pressure_force[tid] = force;
	
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

	float3 acceleration =  (fbuf.force[tid] + fbuf.correction_pressure_force[tid])* 1.0/ _param.mass;

	float3 veval = fbuf.vel_update[tid];
	veval += acceleration * _param.time_step;
	float3 pos = fbuf.pos_update[tid];
	pos = (pos+ (veval*_param.time_step));

	collisionHandling(&pos, &veval);

	output[tid] = pos;
	fbuf.vel_update[tid] = (veval);
}
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
float ReduceMax(float* input, float* output, int num) {
	/*thrust::device_ptr<float> data(input);
	float res= thrust::reduce(data, data + num,
		-1.0,
		thrust::maximum<float>());
	return res;*/
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
void IndexSort(float3* pos_old) {
	
	prescanInt(fbuf.grid_particles_num, fbuf.grid_off,outerGridNum, outerGridDim.x, outerGridDim.y, outerGridDim.z);
	//dim3 blocksize(particleNum);

	auto input1 = vector<float3>(particleNum);
	auto input2 = vector<int>(particleNum);
	//sortIndex << <1, particleNum >> >(fbuf.grid_off, fbuf.grid_particle_offset, fbuf.particle_grid_cell_index, fbuf.sort_index);
#ifdef TEST
	cudaMemcpy(&input1[0], pos_old, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
	for (auto a : input1)
		cout << a.y << " ";
	cout << "pos_old" << endl;
#endif // TEST

	
	cudaDeviceSynchronize();
	sortIndex<<<gridsize_p, blocksize_p >>>(fbuf);
	//Safe
	cudaDeviceSynchronize();
#ifdef TEST
	cudaMemcpy(&input2[0], fbuf.sort_index, particleNum * sizeof(int), cudaMemcpyDeviceToHost);
	for (auto a : input2)
		cout << a << " ";
	cout << "sort_index" << endl;
#endif // TEST

	

	//CUDA_SAFE_CALL(cudaMemcpy(&input1[0], fbuf.sort_index, particleNum * sizeof(int), cudaMemcpyDeviceToHost));
	//for (auto a : input1)
	//	cout << a << " ";
	rearrange<<<gridsize_p, blocksize_p >>>(fbuf,pos_old);
	

#ifdef TEST
	cudaMemcpy(&input1[0], fbuf.pos_update, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
	for (auto a : input1)
		cout << a.y << " ";
	cout << "pos_update" << endl;
#endif // TEST

	
	cudaDeviceSynchronize();
}
void ComputeOtherForce() {    //grivty
//	dim3 blockSize();
	computeOtherForce<<<gridsize_p,blocksize_p>>>(fbuf);
	//Safe
	//auto input1 = vector<float3>(particleNum);

	//cudaMemcpy(&input1[0], fbuf.force, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
	//for (auto a : input1)
	//	cout << a.x << " "<<a.y<<" "<<a.z<<" ";
	//cout << endl;
	//cudaDeviceSynchronize();
}
void PredictonCorrection(float3* output) {
	bool densityErrorLarge=true;
	int cnt = 0;
	CUDA_SAFE_CALL(cudaMemset(fbuf.correction_pressure,0, sizeof(float)*particleNum));
	while (cnt < 1||(densityErrorLarge&&cnt<ITERATION_MAX_NUM)) {
		auto input1 = vector<float3>(particleNum);
		auto input2 = vector<float>(particleNum);

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
		cudaMemcpy(&input1[0], output, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		for (auto a : input1)
			cout << a.y << " ";
		cout << endl;
		cudaDeviceSynchronize();
#endif // TEST

		

		ComputePredictedDensityAndPressure<<<gridsize_p,blocksize_p>>>(fbuf,output);
		//Safe
		//cudaMemcpy(&input1[0], fbuf.test_buff, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		//cout << "density:" << endl;
		//for (auto a : input1)
		//	cout << a.x << " ";
		//cout << endl;
		//cudaDeviceSynchronize();
		//CUDA_SAFE_CALL(cudaMemset(fbuf.densityError, 0, sizeof(float)*particleNum));
//#ifdef TEST
			//cudaMemcpy(&input1[0], fbuf.correction_pressure_force, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		/*	cout << "correction_pressure_force" << endl;
		for (auto a : input1)
			cout << a.y << " ";
		cout << endl;
		cudaDeviceSynchronize();
		cudaMemcpy(&input1[0], output, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
		cout << "pos y" << endl;
		for (auto a : input1)
			cout << a.y << " ";
		cout << endl;
		cudaDeviceSynchronize();*/
		
//#endif // TEST

		
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		float max_density_error;
		max_density_error=ReduceMax(fbuf.densityError, fbuf.max_predicted_density,particleNum);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		//reduceMax << <1, particleNum, particleNum * sizeof(float) >> > (fbuf.densityError, fbuf.max_predicted_density);;
		//cudaMemcpy(&max_density_error, fbuf.max_predicted_density, sizeof(float), cudaMemcpyDeviceToHost);
		max_density_error = MAX(0, max_density_error);
		if (max_density_error / restDensity < 0.03)
			densityErrorLarge = false;
		ComputePressureForce << <gridsize_p, blocksize_p >> >(fbuf);
		cnt++;
	}
}
void Advance(float3* output) {
	advanceParticles << <gridsize_p, blocksize_p >> > (fbuf,output);
	//auto input2 = vector<float3>(particleNum);
	//cudaMemcpy(&input2[0], output, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
	/*for (auto pos : input2) 
		cout << pos.y << " ";
	cout << endl; 
	cudaMemcpy(&input2[0], fbuf.vel_update, particleNum * sizeof(float3), cudaMemcpyDeviceToHost);
	for (auto ve : input2)
		cout << ve.y << " ";
	cout << endl;*/
	////Safe
}
void stepTime() {
	float3 * input,* output;
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
	PredictonCorrection(output);
	Advance(output);
	
	
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[0], 0);
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource[1], 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	swapBuff();
}
__global__ void scanSumInt(int * input, int * output, int *aux,int numPerthread) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int offset = id * numPerthread;
	int prefix=0;
	for (int i = offset; i < offset + numPerthread; i++) {
		int tmp = input[i];
		output[i] = prefix;
		prefix += tmp;
	}
	//if(aux)
	//for (int i = offset; i < offset + numPerthread; i++) {
	if(aux)
		aux[id] = prefix;
	
}
__global__ void addUpPrefix(int* prefix, int* aux ,int stride) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	prefix[id] += aux[id/stride];
}

void prescanInt(int* input, int* output, int len, int numPerThread,int numBlock,int numThread) {
	int * aux;
	//auto input1 = vector<int>(GridNum);

	int totalthread =  (numBlock*numThread);
	CUDA_SAFE_CALL(cudaMalloc(&aux, totalthread *sizeof(int)));
	cudaDeviceSynchronize();

	//cudaMemcpy(&input1[0], input, GridNum * sizeof(int), cudaMemcpyDeviceToHost);

	int sum = 0;
	/*for (int a : input1)
		cout << (sum=sum+a) << " ";
	cudaDeviceSynchronize();*/
	scanSumInt << <numBlock, numThread >> > (input, output,aux,numPerThread);
	

	scanSumInt << <1, 1 >> > (aux, aux, NULL, totalthread);  //in place 
	cudaDeviceSynchronize();
	
	cudaDeviceSynchronize();
	addUpPrefix<<<numBlock*numThread,  numPerThread >>>(output, aux, numPerThread);
	CUDA_SAFE_CALL(cudaFree(aux));

	//cudaMemcpy(&input1[0], output, GridNum * sizeof(int), cudaMemcpyDeviceToHost);
	//
	//
	//for (int a : input1)
	//	cout << a << " ";
	//cudaDeviceSynchronize();

	
}

//__global__ void reduceMax(float* input,float*output, int num) {
//	__shared__ float table[num];
//	int total = num;
//	int tid = threadIdx.x;
//	while (total>5)
//	{
//		__syncthreads();
//	}
//
//}

void testf() {
	int data[6000];
	for (int i = 0; i < 6000; i++)
		data[i] = i;
	int result = thrust::reduce(data, data + 6000,
		-1,
		thrust::maximum<int>());
	cout << result << endl;
	//float input[125];
	//for (int i = 0; i < 125; i++)
	//	input[i] = rand()%125;
	//for (int a : input)
	//	cout << a << " ";
	//cout << endl;
	//float* dev_a,* dev_b;
	//CUDA_SAFE_CALL(cudaMalloc(&dev_a,125*sizeof(float)));
	//CUDA_SAFE_CALL(cudaMalloc(&dev_b, sizeof(float)));
	//////cudaDeviceSynchronize();

	//CUDA_SAFE_CALL(cudaMemcpy(dev_a, input, 125 * sizeof(float), cudaMemcpyHostToDevice));
	//ReduceMax(dev_a, dev_b, 125);
	//////prescanInt(dev_a, dev_b, 125, 5, 5, 5);
	////cudaDeviceSynchronize();
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//float *res=new float[1];
	//////cudaMemset(dev_b, 0, sizeof(float));
	//cudaMemcpy(res, dev_b,  sizeof(float), cudaMemcpyDeviceToHost);
	//cout << res[0] << endl;
}
int getghostNum() {
	return ghostnum;
}

