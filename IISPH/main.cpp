


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <cstdio>
//#include <time.h>
#include "timer.h"
#include "fluid_system.cuh"
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

//#include <cmath>
using namespace glm;

GLFWwindow* window;
const int SCR_WIDTH = 1024;
const int SCR_HEIGHT = 1024;
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

//extern "C" cudaError_t subWithCuda(int *c, const int *a, const int *b, unsigned int size);
//extern "C" void advectParticles(GLuint vbo);
//extern "C" GLuint Location[2];
Timer timer;
particleSystem particle;

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
float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
	 // positions   // texCoords
	-1.0f,  1.0f,  0.0f, 1.0f,
	-1.0f, -1.0f,  0.0f, 0.0f,
	1.0f, -1.0f,  1.0f, 0.0f,

	-1.0f,  1.0f,  0.0f, 1.0f,
	1.0f, -1.0f,  1.0f, 0.0f,
	1.0f,  1.0f,  1.0f, 1.0f
};
void UserInputSetup() {
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);
}
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
			testf();
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
	GLSLShader CubeRender = GLSLShader::createFromShaderFile("shader/watervt.glsl", "shader/waterfg.glsl");
	GLSLShader screen = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/screen.fs");
	GLSLShader Smooth = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/smooth.fs");
	GLSLShader UpadeNorm = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/updateNorm.fs");
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
	
	sphere testsphere(8,radius);
	testsphere.generateBuffer();
	particle.particleStepUp();
	

	// Ensure we can capture the escape key being pressed below
	UserInputSetup();
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	//{
		unsigned int quadVAO, quadVBO;
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		unsigned int framebuffer,framebuffer1;
		glGenFramebuffers(1, &framebuffer);
		glGenFramebuffers(1, &framebuffer1);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		unsigned int DepthNormTex, DepthNormTex1;
		glGenTextures(1, &DepthNormTex);
		glGenTextures(1, &DepthNormTex1);
		glBindTexture(GL_TEXTURE_2D, DepthNormTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, DepthNormTex, 0);

	
		// create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
		unsigned int rbo;
		glGenRenderbuffers(1, &rbo);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
		
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
		glBindTexture(GL_TEXTURE_2D, DepthNormTex1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, DepthNormTex1, 0);
	//}

	

	do {
		handleinput();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		//eyepos = glm::rotateX(vec3(0, 0.0, -mdistance), radians(-angleY));
		//eyepos = rotateY(eyepos, radians(-angleX));
		//viewMatrix = lookAt(eyepos, vec3(0, 0, 0), vec3(0, 1, 0));
		//glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
		//mat4 mvp = projectionMatrix * viewMatrix * wroldMatrix;
		//glEnable(GL_CULL_FACE);
		//glCullFace(GL_FRONT);
		//render.Use();
		//glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
		////glEnable(GL_DEPTH_TEST);
		//glBindVertexArray(VAO);
		//glDrawArrays(GL_TRIANGLES, 0, 30);
		//glDisable(GL_CULL_FACE);
		//// Draw nothing, see you in tutorial 2 !
		//water.Use();
		//glUniformMatrix4fv(water("MVP"),1,GL_FALSE, value_ptr(mvp));
		//glBindVertexArray(testsphere.getVAO());
		//glDrawArrays(GL_TRIANGLES,0, 3 );
		{
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)
			//glDepthMask(GL_FALSE);
									 // make sure we clear the framebuffer's content
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			eyepos = glm::rotateX(vec3(0, 0.0, -mdistance), radians(-angleY));
			eyepos = rotateY(eyepos, radians(-angleX));
			viewMatrix = lookAt(eyepos, vec3(0, 0, 0), vec3(0, 1, 0));
			glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
			mat4 mvp = projectionMatrix * viewMatrix * wroldMatrix;
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			render.Use();

			glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
			glUniformMatrix4fv(render("ViewMatrix"), 1, GL_FALSE, value_ptr(viewMatrix));

			glUniform3fv(render("eyePos"),1,value_ptr(eyepos));
			//glEnable(GL_DEPTH_TEST);
			glBindVertexArray(VAO);
			glEnable(GL_PROGRAM_POINT_SIZE);
			//glPointSize(50);
			
			glEnable(GL_POINT_SPRITE);
			glDepthRange(0.0, 10.0f);
			glDrawArrays(GL_POINTS, 0, 12);
			glDisable(GL_CULL_FACE);
			glDisable(GL_PROGRAM_POINT_SIZE);	
			glDisable(GL_POINT_SPRITE);
			glDisable(GL_DEPTH_TEST);
			int i = 5;
			while (i-->0)
			{
				//UpadeNorm.Use();
				UpadeNorm.Use();

				glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
				glClear(GL_COLOR_BUFFER_BIT);
				//glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
				glBindVertexArray(quadVAO);
				glBindTexture(GL_TEXTURE_2D, DepthNormTex);
				glDrawArrays(GL_TRIANGLES, 0, 6);
				swap(DepthNormTex, DepthNormTex1);
				swap(framebuffer,framebuffer1);
				Smooth.Use();
				glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
				glClear(GL_COLOR_BUFFER_BIT);
				//glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
				glBindVertexArray(quadVAO);
				glBindTexture(GL_TEXTURE_2D, DepthNormTex);
				glDrawArrays(GL_TRIANGLES, 0, 6);
				swap(DepthNormTex, DepthNormTex1);
				swap(framebuffer, framebuffer1);
			}
			
			 // disable depth test so screen-space quad isn't discarded due to depth test.
									  // clear all relevant buffers
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(.5f, .1f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
			glClear(GL_COLOR_BUFFER_BIT);

			screen.Use();
			glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
			glBindVertexArray(quadVAO);
			glBindTexture(GL_TEXTURE_2D, DepthNormTex);	// use the color attachment texture as the texture of the quad plane
			glDrawArrays(GL_TRIANGLES, 0, 6);
			glDisable(GL_DEPTH_TEST);
			//swap(DepthNormTex, DepthNormTex1);
			//swap(framebuffer,framebuffer1);
		}

		particle.step();
		{
			//glBindBuffer(GL_ARRAY_BUFFER,particle.getRenderVBO());
			//glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
			//glVertexAttribDivisor(1, 1);
			//glDrawElementsInstanced(GL_TRIANGLES, testsphere.getElementSize(), GL_UNSIGNED_INT, 0,particle.getParticleNum());
		}
		
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

