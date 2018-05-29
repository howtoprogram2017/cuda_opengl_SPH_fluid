


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
//#include <time.h>
#include "timer.h"
#include "fluid_system.cuh"
#include "GLSLShader.h"
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <GL/glew.h>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "sphere.h"
#include "BoundaryData.h"
vector<float3> boundaryWall;
vector<double3> boundaryObject;


//#include <cmath>
using namespace glm;

GLFWwindow* window;
const int SCR_WIDTH = 1024;
const int SCR_HEIGHT = 1024;
float angleX = 45;
float angleY = -45;
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
mat4 viewMatrix = lookAt(vec3(0, 3, 0), vec3(0, 0, 0), vec3(0, 1, 0));
string ProgramName;
float mdistance = 3; vec3 eyepos = vec3(0, 3, 0);
mat4 projectionMatrix = perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.2f, 100.0f);
vec4 LightPos = vec4(0.0,-300.0,0.0,1.0);
#define testDuration(x) timer.start(); x; timer.stop();std::cout << timer.duration() << "ms" << std::endl;
extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


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
	}
}

const int bytesPerPixel = 3; /// red, green, blue
const int fileHeaderSize = 14;
const int infoHeaderSize = 40;
unsigned char* createBitmapInfoHeader(int height, int width) {
	static unsigned char infoHeader[] = {
		0,0,0,0, /// header size
		0,0,0,0, /// image width
		0,0,0,0, /// image height
		0,0, /// number of color planes
		0,0, /// bits per pixel
		0,0,0,0, /// compression
		0,0,0,0, /// image size
		0,0,0,0, /// horizontal resolution
		0,0,0,0, /// vertical resolution
		0,0,0,0, /// colors in color table
		0,0,0,0, /// important color count
	};
	infoHeader[0] = (unsigned char)(infoHeaderSize);
	infoHeader[4] = (unsigned char)(width);
	infoHeader[5] = (unsigned char)(width >> 8);
	infoHeader[6] = (unsigned char)(width >> 16);
	infoHeader[7] = (unsigned char)(width >> 24);
	infoHeader[8] = (unsigned char)(height);
	infoHeader[9] = (unsigned char)(height >> 8);
	infoHeader[10] = (unsigned char)(height >> 16);
	infoHeader[11] = (unsigned char)(height >> 24);
	infoHeader[12] = (unsigned char)(1);
	infoHeader[14] = (unsigned char)(bytesPerPixel * 8);

	return infoHeader;
}
unsigned char* createBitmapFileHeader(int height, int width) {
	int fileSize = fileHeaderSize + infoHeaderSize + bytesPerPixel * height*width;

	static unsigned char fileHeader[] = {
		0,0, /// signature
		0,0,0,0, /// image file size in bytes
		0,0,0,0, /// reserved
		0,0,0,0, /// start of pixel array
	};
	fileHeader[0] = (unsigned char)('B');
	fileHeader[1] = (unsigned char)('M');
	fileHeader[2] = (unsigned char)(fileSize);
	fileHeader[3] = (unsigned char)(fileSize >> 8);
	fileHeader[4] = (unsigned char)(fileSize >> 16);
	fileHeader[5] = (unsigned char)(fileSize >> 24);
	fileHeader[10] = (unsigned char)(fileHeaderSize + infoHeaderSize);

	return fileHeader;
}
void generateBitmapImage(unsigned char *image, int height, int width,const char* imageFileName) {

	unsigned char* fileHeader = createBitmapFileHeader(height, width);
	unsigned char* infoHeader = createBitmapInfoHeader(height, width);
	unsigned char padding[3] = { 0, 0, 0 };
	int paddingSize = (4 - (width*bytesPerPixel) % 4) % 4;
	
	FILE* imageFile = fopen(imageFileName, "wb");
	fwrite(fileHeader, 1, fileHeaderSize, imageFile);
	fwrite(infoHeader, 1, infoHeaderSize, imageFile);

	int i;
	for (i = 0; i<height; i++) {
		fwrite(image + (i*width*bytesPerPixel), bytesPerPixel, width, imageFile);
		fwrite(padding, 1, paddingSize, imageFile);
	}
	fclose(imageFile);
}
  //change R B channel for .bmp file gennerage
void SwapRBchannel(unsigned char* image,int count){
	for (int i = 0; i < count; i++) {
		swap(image[3*i],image[3*i+2]);
	}
}



void SaveImgTofile() {
	int Size = (int)(SCR_WIDTH*SCR_HEIGHT * 3);
	unsigned char* imageData = (unsigned char *)malloc(Size);
	glReadnPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, sizeof(unsigned char)*(SCR_WIDTH*SCR_HEIGHT * 3), imageData);
	static int count = 0;
	count++;
	SwapRBchannel(imageData, SCR_WIDTH*SCR_HEIGHT);
	if (count == 1)
		CreateDirectory(("ScreenShoot/" + ProgramName).c_str(), NULL);
	string filename = "ScreenShoot/"+ProgramName+  "/screen_shot" +to_string( count)+".bmp";
	generateBitmapImage(imageData, SCR_HEIGHT, SCR_WIDTH, filename.c_str());
	free(imageData);
	return;

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
	else if (paused&& key == GLFW_KEY_S && action == GLFW_PRESS) {
		SaveImgTofile();
	}
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
	-0.55f, -0.55f, -0.55f,  0.0f, 0.0f,
	0.55f,  0.55f, -0.55f,  1.0f, 1.0f,
	0.55f, -0.55f, -0.55f,  1.0f, 0.0f,
	-0.55f, -0.55f, -0.55f,  0.0f, 0.0f,
	-0.55f,  0.55f, -0.55f,  0.0f, 1.0f,
	0.55f,  0.55f, -0.55f,  1.0f, 1.0f,

	-0.55f, -0.55f,  0.55f,  0.0f, 0.0f,
	0.55f, -0.55f,  0.55f,  1.0f, 0.0f,
	0.55f,  0.55f,  0.55f,  1.0f, 1.0f,
	0.55f,  0.55f,  0.55f,  1.0f, 1.0f,
	-0.55f,  0.55f,  0.55f,  0.0f, 1.0f,
	-0.55f, -0.55f,  0.55f,  0.0f, 0.0f,

	-0.55f,  0.55f, -0.55f,  1.0f, 1.0f,
	-0.55f, -0.55f, -0.55f,  0.0f, 1.0f,
	-0.55f,  0.55f,  0.55f,  1.0f, 0.0f,
	-0.55f, -0.55f,  0.55f,  0.0f, 0.0f,
	-0.55f,  0.55f,  0.55f,  1.0f, 0.0f,
	-0.55f, -0.55f, -0.55f,  0.0f, 1.0f,

	0.55f,  0.55f, -0.55f,  1.0f, 1.0f,
	0.55f,  0.55f,  0.55f,  1.0f, 0.0f,
	0.55f, -0.55f, -0.55f,  0.0f, 1.0f,
	0.55f, -0.55f,  0.55f,  0.0f, 0.0f,
	0.55f, -0.55f, -0.55f,  0.0f, 1.0f,
	0.55f,  0.55f,  0.55f,  1.0f, 0.0f,
	//bottom
	-0.55f, -0.55f, -0.55f,  0.0f, 1.0f,
	0.55f, -0.55f, -0.55f,  1.0f, 1.0f,
	0.55f, -0.55f,  0.55f,  1.0f, 0.0f,
	-0.55f, -0.55f,  0.55f,  0.0f, 0.0f,
	-0.55f, -0.55f, -0.55f,  0.0f, 1.0f,
	0.55f, -0.55f,  0.55f,  1.0f, 0.0f,
};
float testVertices[] = {
	0.1,0.2,-1.2,
	0.4,0.2,1.3,
	0.3,0.2,1.2,
	-0.2,0.5,-1.2,
	0.1,0.2,0.2
};
float quadVertices[] =
{ // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
  // positions   // texCoords
	-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,

		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f
};

void UserInputSetup() {
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);
}

int main()
{
	ProgramName = " ss";
	//cin >> ProgramName;
	OBJLoader loader;
	vector<float3> x;
	vector<float3> normal;
	vector<float2> tex;
	double3 scale = { 1.1f,1.1f,1.1f };
	vector< OBJLoader::MeshFaceIndices> mesh;
	string meshFileName = "BoundaryData/UnitBox.obj";
	//loader.loadObj("BoundaryData/UnitBox.obj", &x, &mesh, &normal,&tex, scale);
	PoissonDiskSampling sampler;
	TriangleMesh geo;
	TriangleMesh boxgeo; 

	loadObj(meshFileName, boxgeo, scale);
	loadObj("BoundaryData/Dragon_50k.obj", geo, { 0.6,0.6,0.5 });
	vec3 offset(-0.3,-0.4,0.0);
	for (auto& el : geo.getVertices()) {
		//rotateX(a, radians(-angleY));
		vec3 newPos = (rotateY(vec3((float)el.x, (float)el.y, (float)el.z), radians(90.0f)) + offset);
		el = {newPos.x,newPos.y,newPos.z};
		assert(el.x < -0.1);
		assert(el.x > -0.55);

	}
	geo.updateVertexNormals();
	//sampler.sampleMesh(x.size(), &x[0], );
	vector<double3> boundaryDataD;
	int preindNum = boxgeo.getVertices().size();
	/*for (auto& vert : geo.getVertices()) {
		boxgeo.addVertex(vert);
	}
	
	for (int i = 0; i < geo.getFaces().size(); i += 3) {
		unsigned indices[] = { geo.getFaces()[i]+ preindNum,geo.getFaces()[i+1]+ preindNum,geo.getFaces()[i+2]+ preindNum };
		boxgeo.addFace(indices);
	}*/
	
	sampler.sampleMesh(boxgeo.numVertices(), boxgeo.getVertices().data(), boxgeo.numFaces(), boxgeo.getFaces().data(), 0.01, 10, 1, boundaryDataD);
	//sampler.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), 0.010, 10, 1, boundaryDataD);

	boxgeo.release();
	
	sampler.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), 0.015, 10, 1, boundaryDataD);
	//vector<float3> boudaryf;//= { {0.5f,0.5f,0.5f},{ 0.5f,0.5f,-0.5f },{ 0.5f,-0.5f,0.5f },{ -0.5f,0.5f,0.5f }
	//,{-0.5f,0.5f,-0.5f}, {-0.5f,-0.5f,0.5f}, {0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f} };
	for (auto & b : boundaryDataD)
		boundaryWall.push_back({ (float)b.x,(float)b.y,(float)b.z });
	boundaryDataD.clear();
	//for (auto & b : boundaryDataD)
	//	boundaryWall.push_back({ (float)b.x,(float)b.y,(float)b.z });
	/*const int arraySize = 500;
	 int a[arraySize] ;
	memset(a, sizeof(a), 0);
	 int b[arraySize];
	memset(b, sizeof(b), 0);
	int c[arraySize];*/
	//auto t=clock();
	// Add vectors in parallel.
	//time_point<Clock> start = Clock::now();
	////sleep_for(500ms);
	/*	cudaError_t cudaStatus;
		int k = 100;
		testDuration(addWithCuda(c, a, b, arraySize);)
			testf();*/
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
	GLSLShader textured = GLSLShader::createFromShaderFile("shader/Simplevt.glsl", "shader/textured.fs");
	GLSLShader screen = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/screenWater.fs");
	GLSLShader Smooth = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/smooth.fs");
	GLSLShader UpadeNorm = GLSLShader::createFromShaderFile("shader/screen.vt", "shader/updateNorm.fs");
	//render.Use();
	unsigned int VBO, cubeVAO, vboTest, vaoTest, EBO;
	glGenVertexArrays(1, &cubeVAO);
	glGenVertexArrays(1, &vaoTest);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	//unsigned indices[] = {0, 1, 2, 3, 4, 5};
	glBindVertexArray(cubeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glBindVertexArray(vaoTest);
	glGenBuffers(1, &vboTest);
	glBindBuffer(GL_ARRAY_BUFFER, vboTest);
	glBufferData(GL_ARRAY_BUFFER, sizeof(testVertices), testVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,0, (void*)0);
	glEnableVertexAttribArray(0);
	sphere testsphere(8, particle.getRadius());
	testsphere.generateBuffer();
	particle.particleSetUp();
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
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		unsigned int framebuffer,framebuffer1;
		glGenFramebuffers(1, &framebuffer);
		glGenFramebuffers(1, &framebuffer1);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		unsigned int DepthNormTex, DepthNormTex1;
		glGenTextures(1, &DepthNormTex);
		glGenTextures(1, &DepthNormTex1);
		unsigned int rbo1;
		glGenRenderbuffers(1, &rbo1);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo1);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo1);
		glBindTexture(GL_TEXTURE_2D, DepthNormTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, DepthNormTex1, 0);
		
		unsigned int cubeTex;
		glGenTextures(1, &cubeTex);
		glBindTexture(GL_TEXTURE_2D, cubeTex);
		int width, height, nrChannels;
		unsigned char *data = stbi_load("textures/images.jpg", &width, &height, &nrChannels, 0);
		if (data) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else
			std::cout << "Failed to load texture" << std::endl;
		stbi_image_free(data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//}
		vector<vec3> vertexes;
		 offset = { -0.0,-0.0,0.0 };
		for (auto& el:geo.getVertices()) {
			//rotateX(a, radians(-angleY));
			vertexes.push_back( vec3( (float)el.x,(float)el.y,(float)el.z ));
		}
		unsigned StaticObjectVA0;
		unsigned StaticObjectVB0;
		unsigned StaticObjectEB0;
		glGenBuffers(1, &StaticObjectVB0);
		glGenBuffers(1, &StaticObjectEB0);
		glGenVertexArrays(1, &StaticObjectVA0);
		glBindVertexArray(StaticObjectVA0);
		glBindBuffer(GL_ARRAY_BUFFER, StaticObjectVB0);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*vertexes.size(), &vertexes[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, StaticObjectEB0);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, geo.getFaces().size() * sizeof(unsigned int), &geo.getFaces()[0], GL_STATIC_DRAW);
		//vertexes.clear();

		//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		/*unsigned boundary;
		glGenBuffers(1, &boundary);
		glBindBuffer(GL_ARRAY_BUFFER, boundary);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*boudaryf.size(), &boudaryf[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);*/

	do {
		handleinput(); 
		float w;
		if(!paused)
		particle.step();
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		eyepos = glm::rotateX(vec3(0, 0.0, -mdistance), radians(-angleY));
		eyepos = rotateY(eyepos, radians(-angleX));
		viewMatrix = lookAt(eyepos, vec3(0, 0, 0), vec3(0, 1, 0));
		mat4 mvp = projectionMatrix * viewMatrix * wroldMatrix;
		w = projectionMatrix[0].x;

		{
			
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)
			//glDepthMask(GL_FALSE);
									 // make sure we clear the framebuffer's content
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			viewMatrix = lookAt(eyepos, vec3(0, 0, 0), vec3(0, 1, 0));
			vec4 LightPosView4 = viewMatrix * LightPos;
			vec3 LightPosView = { LightPosView4.x / LightPosView4.w,LightPosView4.y / LightPosView4.w,LightPosView4.z / LightPosView4.w };
			glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
			
			render.Use();
			glUniformMatrix4fv(render("MVP"), 1, GL_FALSE, value_ptr(mvp));
			glUniformMatrix4fv(render("ViewMatrix"), 1, GL_FALSE, value_ptr(viewMatrix));
			glUniform1f(render("w"), w);
			glUniform1f(render("WordSize"), particle.getSmoothRadius() / 2.30);
			glUniform3fv(render("eyePos"), 1, value_ptr(eyepos));
			//glEnable(GL_DEPTH_TEST);
			//	glBindVertexArray(VAO);
			glEnable(GL_PROGRAM_POINT_SIZE);
			glEnable(GL_POINT_SPRITE);
			glBindVertexArray(particle.getRenderVBO());
			glDrawArrays(GL_POINTS, 0, particle.getParticleNum());
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisable(GL_PROGRAM_POINT_SIZE);
			glDisable(GL_POINT_SPRITE);
			glDisable(GL_DEPTH_TEST);
			int smoothIteration = 20;
			int i = smoothIteration -1;
			while (i-->0)
			{
			
				Smooth.Use();
				glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
				glClear(GL_COLOR_BUFFER_BIT);
				glBindVertexArray(quadVAO);
				glBindTexture(GL_TEXTURE_2D, DepthNormTex);
				glUniform1f(Smooth("w"),w);
				//glViewport(0, 0, 1.0, 1.0);
				glDrawArrays(GL_TRIANGLES, 0, 6);
				swap(DepthNormTex, DepthNormTex1);
				swap(framebuffer, framebuffer1);
			}
			/*if (smoothIteration & 1) {
				swap(DepthNormTex, DepthNormTex1);
				swap(framebuffer, framebuffer1);
			}*/
				
			UpadeNorm.Use();
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
			glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
			glBindVertexArray(quadVAO);
			glBindTexture(GL_TEXTURE_2D, DepthNormTex);
			glUniform1f(Smooth("w"), w);
			glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
			glDrawArrays(GL_TRIANGLES, 0, 6);
			swap(DepthNormTex, DepthNormTex1);
			swap(framebuffer, framebuffer1);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(.5f, .1f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
			//glClear(GL_COLOR_BUFFER_BIT);

			screen.Use();
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			

			glUniform3fv(screen("LightPos"), 1, value_ptr(LightPosView));
			glUniformMatrix4fv(screen("MVP"), 1, GL_FALSE, value_ptr(mvp));
			glUniformMatrix4fv(screen("Projection"), 1, GL_FALSE, value_ptr(projectionMatrix));
			glUniform1i(screen("State"), 0);
			glUniform1f(screen("w"), w);
			glBindVertexArray(quadVAO);
			glBindTexture(GL_TEXTURE_2D, DepthNormTex);	// use the color attachment texture as the texture of the quad plane
			glDrawArrays(GL_TRIANGLES, 0, 6); 
			
		//	
		//	
				textured.Use();
				glUniformMatrix4fv(textured("MVP"), 1, GL_FALSE, value_ptr(mvp));
				glBindVertexArray(StaticObjectVA0);
				glUniform1i(textured("State"), 2);

				//
				glDrawElements(GL_TRIANGLES, geo.getFaces().size(), GL_UNSIGNED_INT, (void*)0);
				//glEnable(GL_CULL_FACE);
				//glCullFace(GL_FRONT);
				glUniform1i(textured("State"), 1);
				glBindTexture(GL_TEXTURE_2D, cubeTex);

				glBindVertexArray(cubeVAO);
				glDrawArrays(GL_TRIANGLES, 0, 30);
			glDisable(GL_DEPTH_TEST);

		}


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

//textured.Use();
//glUniformMatrix4fv(textured("MVP"), 1, GL_FALSE, value_ptr(mvp));
//glBindVertexArray(StaticObjectVA0);
//glUniform1i(textured("State"), 2);
//
////
//glDrawElements(GL_TRIANGLES, geo.getFaces().size(), GL_UNSIGNED_INT, (void*)0);
//		//glEnable(GL_CULL_FACE);
//		//glCullFace(GL_FRONT);
//		glUniform1i(textured("State"), 1);
//		glBindTexture(GL_TEXTURE_2D,cubeTex);
//		
//		glBindVertexArray(cubeVAO);
//		glDrawArrays(GL_TRIANGLES, 0, 30);
//glDisable(GL_CULL_FACE);