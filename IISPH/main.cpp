


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <cstdio>
//#include <time.h>
#include "timer.h"
#include "GLSLShader.h"
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include <glm/tra>
#include <glm/gtx/rotate_vector.hpp>
#include <GL/glew.h>
#include <vector>
#include "sphere.h"
#include "PCISPH.h"
//#include <cmath>
using namespace glm;

GLFWwindow* window;
const int SCR_WIDTH = 1024;
const int SCR_HEIGHT = 768;
float angleX = 0;
float angleY = 0;
float oldangleX = angleX;
float oldangleY = angleY;
double oldPosX = -25;
double oldPosY = -200.5;
enum State { click, DragWater, DragRotate, noInput };
bool paused = false;
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
State Mode = noInput;
mat4 wroldMatrix = mat4();
mat4 viewMatrix = lookAt(vec3(0, 0, -3), vec3(0, 0, 0), vec3(0, 1, 0));//glm::translate(mat4(), glm::vec3(0.0f, 0.0f, -3.0f));
float mdistance = 3; vec3 eyepos = vec3(0, 0, -3);
mat4 projectionMatrix = perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
#define testDuration(x) timer.start(); x; timer.stop();std::cout << timer.duration() << "ms" << std::endl;

extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

extern "C" cudaError_t subWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" void advectParticles(GLuint vbo, float *v);
Timer timer;

void GetCursorPos(double&x, double&y) {
	double x1, y1;
	glfwGetCursorPos(window, &x1, &y1);
	x = x1 - SCR_WIDTH / 2;
	y = SCR_HEIGHT / 2 - y1;
}
void handleinput() {
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS) {
		double x, y;
		GetCursorPos(x, y);
		if (Mode == DragRotate) {
			angleX = oldangleX + (x - oldPosX) * .5;
			angleY = oldangleY + (y - oldPosY) * .5;
			if (angleY > 89) {
				angleY = 89;
			}
			if (angleY < -89)
				angleY = -89;
		}
		/*else if (Mode == DragWater) {
			auto raydes1 = inverse(projectionMatrix*viewMatrix)*vec4(x / (SCR_WIDTH / 2), y / (SCR_HEIGHT / 2), 1.0, 1);
			auto raydir = vec3(raydes1) / raydes1.w - eyepos;
			auto minpos = make_pair(-(TEX_WIDTH*details) / 2, -(TEX_HEIGHT*details) / 2);
			auto maxpos = make_pair(TEX_WIDTH*details / 2, TEX_HEIGHT*details / 2);
			auto result = mRaytracer.HitAxisAlinedPlane(eyepos, raydir, minpos, maxpos);
			if (result.first) {
				auto hitpoint = result.second.second;
				dragPos = pair<int, int>{ round(hitpoint.x / details + TEX_WIDTH * 0.5),round(hitpoint.z / details + TEX_HEIGHT * 0.5) };
			}
		}*/

	}
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	mdistance -= yoffset * .2;
	eyepos = glm::rotateX(vec3(0, 0.0, -mdistance), radians(-angleY));
	eyepos = rotateY(eyepos, radians(-angleX));

}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		paused = !paused;
	/*if (key == GLFW_KEY_R && action == GLFW_PRESS)
	Mode = Mode == DragRotate ? DragWater : DragRotate;*/
}
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		//Mode =paused?DragWater:DragRotate;
		oldangleX = angleX; oldangleY = angleY;
		double x, y;
		GetCursorPos(x, y);
		oldPosX = x; oldPosY = y;
		/*auto raydes1 = inverse(projectionMatrix*viewMatrix)*vec4(x / (SCR_WIDTH / 2), y / (SCR_HEIGHT / 2), 1.0, 1);
		auto raydir = vec3(raydes1) / raydes1.w - eyepos;
		auto minpos = make_pair(-(TEX_WIDTH*details) / 2, -(TEX_HEIGHT*details) / 2);
		auto maxpos = make_pair(TEX_WIDTH*details / 2, TEX_HEIGHT*details / 2);
		auto result = mRaytracer.HitAxisAlinedPlane(eyepos, raydir, minpos, maxpos);
		if (result.first)
			Mode = DragWater;
		else*/
			Mode = DragRotate;

	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		Mode = noInput;
		double x, y;
		GetCursorPos(x, y);
		//cout << x << " " << y;
	}
}

float vertices[] = {
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,

	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

	-0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	//bottom
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
};

int main()
{
	const int arraySize = 500;
	 int a[arraySize] ;
	memset(a, sizeof(a), 0);
	 int b[arraySize];
	memset(b, sizeof(b), 0);
	int c[arraySize];
	//auto t=clock();
	// Add vectors in parallel.
	//time_point<Clock> start = Clock::now();
	////sleep_for(500ms);
		cudaError_t cudaStatus;
		int k = 100;
		testDuration(addWithCuda(c, a, b, arraySize);)
	k = 100;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	
	glfwMakeContextCurrent(window);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glewExperimental = GL_TRUE;
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SPH_FLUID", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);


	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	//glewExperimental = true; // Needed for core profile

	//glCreateShader(GL_VERTEX_SHADER);
	GLSLShader render = GLSLShader::createFromShaderFile("shader/vt.glsl", "shader/fg.glsl");
	GLSLShader water = GLSLShader::createFromShaderFile("shader/watervt.glsl", "shader/waterfg.glsl");

	//render.Use();
	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	//setup for sphere drawing
	

	sphere testsphere(8,0.5);
	testsphere.generateBuffer();
	unsigned int VBO1, VAO1, EBO;
	vector<float> pointData = { -0.0,0.0,0,-0.0,-0.0,0,0.0,0,0 };
	glGenVertexArrays(1, &VAO1);
	glGenBuffers(1, &VBO1);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO1);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*pointData.size(), &(pointData[0]), GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*indices.size(), &(indices[0]), GL_STATIC_DRAW);
	//this->VAO = VAO; this->VBO = VBO;
	
	

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	

	do {
		handleinput();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		eyepos = glm::rotateX(vec3(0, 0.0, -mdistance), radians(-angleY));
		eyepos = rotateY(eyepos, radians(-angleX));
		viewMatrix = lookAt(eyepos, vec3(0, 0, 0), vec3(0, 1, 0));
		glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
		mat4 mvp = projectionMatrix * viewMatrix * wroldMatrix;
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		render.Use();
		glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
		//glEnable(GL_DEPTH_TEST);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 30);
		glDisable(GL_CULL_FACE);
		// Draw nothing, see you in tutorial 2 !
		water.Use();
		glUniformMatrix4fv(water("MVP"),1,GL_FALSE, value_ptr(mvp));
		glBindVertexArray(testsphere.getVAO());
		//glDrawArrays(GL_TRIANGLES,0, 3 );
		//glBindBuffer(GL_ARRAY_BUFFER, Location);
		//advectParticles(&loc);
		//glEnableVertexAttribArray(1);
		advectParticles(1, &pointData[0]);
		glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
		glVertexAttribDivisor(1, 1);
		glDrawElementsInstanced(GL_TRIANGLES, testsphere.getElementSize(), GL_UNSIGNED_INT, 0,3);
		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;

}

// Helper function for using CUDA to add vectors in parallel.

