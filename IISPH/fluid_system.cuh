#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "timer.h"
#include <stdio.h>
#include <algorithm>
#include <GL/glew.h>
#define _USE_MATH_DEFINES
//#include <cmath>
#include <math.h>
#include <cuda_gl_interop.h>
#include "math_define.cuh"
using namespace std;

#  define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
		char c;\
      while(cin>>c){if(c=='\t') break;}		\
        exit(EXIT_FAILURE);                                                  \
    } }
#  define SAFE_KERNEL cudaError_t error = cudaGetLastError(); \
if (error != cudaSuccess) { \
	fprintf(stderr, "CUDA ERROR: : %s\n", cudaGetErrorString(error)); \
}

#define MAX(a,b) a>b?a:b

const int neighborGridNum = 27;



struct bufflist
{
	//float3*			pos;
	float3*			pos_update;  //position after sorted
	float3*			vel_old;     
	float3*         vel_update;
	float3*			force;
	float*			correction_pressure;
	float*			boudary_particle_pos; // ghost particle;
	
	float*			densityError;
	float*			max_predicted_density;
	uint*			particle_grid_cell_index;
	uint*			particle_grid_cell_index_update;
	uint*			grid_particle_offset;
	float3*         correction_pressure_force;
	uint*			sort_index;
	uint*			particle_index_grid; //

	int*			grid_particles_num;
	int*			grid_off;   //	

	int*			ghost_grid_particles_num;
	int*			ghost_grid_off;
	float3*			ghost_pos;
	float*	        ghost_volum;  
	//for IIsph
	float3*			Dii;
	float*			aii;
	float3*			sumDij_Pj;
	float*			particle_density;
	float*			advect_density;
	float*			correction_pressure_update;

};


struct ParticleParams {

	float3 _minGridCorner;
	float3 _maxGridCorner;
	float3 minOuterBound;
	float _GridSize;
	int3 outerGridDim;
	uint particleNum;   //same as grid num
	float gravity;
	float mass;
	float time_step;
	float smooth_radius;
	float rest_density;
	float poly6kernel;
	float poly6kernelGradient;
	float spikykernelGradient;
	float param_density_error_factor;
	int3   _neighbor_off[neighborGridNum];   //might larger than actual neighbors
											 //static ParticleParams* createParam(float gravity, float mass, float time_step)() {return };
											 //ParticleParams(float gravity, float mass, float time_step):gravity(gravity), mass(mass),time_step(time_step){	};
};
class fluid_system {
public:
	virtual  void step() {};
	virtual  GLuint getRenderVBO() { return 0; };
	virtual uint getParticleNum() { return 0; };
	virtual uint getGhostParticleNum() { return 0; };
	virtual float getRadius() { return 0.0; };
	virtual float getSmoothRadius() { return 0.0; };
	virtual void particleSetUp() {};
	void setTimeStep(float timestep) { this->time_step = timestep; };
	void setRadius(float r) { radius = r; }
	void setSmoothRadius(float sr) { smooth_radius = sr; };

protected:
	float radius= 0.015, smooth_radius= 4*radius, time_step=0.0005;
};
class PCISPH_solver :public fluid_system{
public:
	void step();
	GLuint getRenderVBO();
	uint getParticleNum();
	uint getGhostParticleNum();
	float getRadius();
	float getSmoothRadius();
	void particleSetUp();
private:
	void setUpParameter();
};

class IISPH_solver :public fluid_system {
public:
	void step();
	GLuint getRenderVBO();

	
	
	uint getParticleNum();
	uint getGhostParticleNum();
	float getRadius();
	float getSmoothRadius();
	void particleSetUp();
private:
	void setUpParameter();
	float ReduceMax(float * input, float * output, int num);
	void prescanInt(int * input, int * output, int len, int numPerThread, int numBlock, int numThread);
	void IndexSort(float3 * pos_old);
	void RelaxedJacobiIteration(float3 * output);
	void ComputeBeforIteration();
	void Advance(float3 * output);
	void CountParticles(float3 * input);
	void swapBuff();
};
