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
//#define FLOAT3_ADD(a,b) make_float3(a.x+b.x,a.y+b.y,a.z+b.z)
//#define FLOAT3_SUB(a,b) make_float3(a.x-b.x,a.y-b.y,a.z-b.z)
//#define FlOAT3_NEG(a)  make_float3(-a.x,-b.y,-c.z)   
//#define FLOAT3_DOT(a,b)  (a.x*b.x+a.y*b.y+a.z*b.z)
//#define FLOAT3_MUL_SCALAR(f,v) make_float3(f.x*v,f.y*v,f.z*v)
#define MAX(a,b) a>b?a:b


const float radius = 0.025;
const float smoothRadius = radius * 4;
const float densityRatio = 1;   //control neighborNum
const float GridSize = smoothRadius / densityRatio ;
const float3 minGridCorner = {-0.5,-0.5,-0.5};
const float3 maxGridCorner = { 0.5,0.5,0.5 };
const float3 OuterGridRange = maxGridCorner - minGridCorner + (2 * smoothRadius)*make_float3(1, 1, 1); //FLOAT3_ADD( FLOAT3_SUB(maxGridCorner,minGridCorner),make_float3(smoothRadius*2, smoothRadius*2, smoothRadius*2));
const float3 minOuterBound = { minGridCorner.x - smoothRadius,minGridCorner.y - smoothRadius,minGridCorner.z - smoothRadius };
const float3 minWaterCorner = { -0,-0.5,-0.5 };
const float3 maxWaterCorner = {0.5,0.5,0.5};
const float3 waterRange = maxWaterCorner - minWaterCorner;// FLOAT3_SUB(maxWaterCorner, minWaterCorner);
const float restDensity = 1000;
const uint influcedParticleNum = 50;
const float mass = 4*M_PI/(3*influcedParticleNum)*pow(smoothRadius,3.0f)*restDensity;
const float initialDistance = pow(mass/restDensity,1.0/3.0);
 const int3 outerGridDim = { ceil(OuterGridRange.x / GridSize),ceil(OuterGridRange.y / GridSize) ,ceil(OuterGridRange.z / GridSize) };
const uint particleNum = (int)ceil(waterRange.x/initialDistance)*(int)ceil(waterRange.y/initialDistance)*(int)ceil(waterRange.z/initialDistance);  //a particle per grid
const int neighborGridNum = 27;
const int outerGridNum=outerGridDim.x*outerGridDim.y*outerGridDim.z; //including ghost particles


extern "C" {
	uint getLocation();//VBO
	void advectParticles(float3*,float3*);
	void Setup();
	void stepTime();
	void prescanInt(int* input, int* output, int len, int numPerThread, int numBlock, int numThread);
	void testf();
	int getghostNum();
}

struct bufflist
{
	//float3*			pos;
	float3*			pos_update;  //position after sorted
	float3*			vel_old;     
	float3*         vel_update;
	float3*			force;
	float*			correction_pressure;
	float*			boudary_particle_pos; // ghost particle;
	float*			predicted_density;
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
	float3*			test_buff;
	//int*			grid_active;

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
	float param_density_error_factor;
	int3   _neighbor_off[neighborGridNum];   //might larger than actual neighbors
	//static ParticleParams* createParam(float gravity, float mass, float time_step)() {return };
	//ParticleParams(float gravity, float mass, float time_step):gravity(gravity), mass(mass),time_step(time_step){	};
};
class particleSystem {
public:
	void step() { stepTime(); }
	GLuint getRenderVBO() { return getLocation(); };
	uint getParticleNum() { return particleNum; }
	uint getGhostParticleNum() { return getghostNum(); }

	void particleStepUp() { Setup();};
};
