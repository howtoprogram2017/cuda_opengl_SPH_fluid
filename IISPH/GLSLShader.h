//A simple class for handling GLSL shader compilation
//Auhtor: Movania Muhammad Mobeen
#pragma once
#ifndef GLSL_SHADER
#define GLSL_SHADER
#include <GL/glew.h>
#include <map>
#include <string>

using namespace std;

class GLSLShader
{
public:
	GLSLShader(void);
	~GLSLShader(void);
	void LoadFromString(unsigned int whichShader, const string source);
	void LoadFromFile(unsigned int whichShader, const string filename);
	void CreateAndLinkProgram();
	void Use();
	void UnUse();
	void AddAttribute(const string attribute);
	void AddUniform(const string uniform);
	unsigned int GetProgram() const;
	//An indexer that returns the location of the attribute/uniform
	unsigned int operator[](const string attribute);
	unsigned int operator()(const string uniform);
	//Program deletion
	void DeleteProgram() { glDeleteProgram(_program); _program = -1; }
	static  GLSLShader createFromShaderFile(const string vertexshader, const string fragmentshader) {
		GLSLShader waterP;
		waterP.LoadFromFile(GL_VERTEX_SHADER, vertexshader);
		waterP.LoadFromFile(GL_FRAGMENT_SHADER, fragmentshader);
		waterP.CreateAndLinkProgram();
		return waterP;
	}
private:
	enum ShaderType { VERTEX_SHADER, FRAGMENT_SHADER, GEOMETRY_SHADER };
	unsigned int	_program;
	int _totalShaders;
	unsigned int _shaders[3];//0->vertexshader, 1->fragmentshader, 2->geometryshader
	map<string, unsigned int> _attributeList;
	map<string, unsigned int> _uniformLocationList;
};
#endif
