#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
uniform vec3 eyePos;
uniform float w;
//uniform float h;
//const float eyePos=
out vec2 TexCoord;
out  float Size;
out float ScreenDepth;   //positive
uniform mat4 MVP;  //projection * view *model
uniform mat4 ViewMatrix;
const float SCR_DIM = 1024.0;
#define M_PI 3.1415926535897932384626433832795
 uniform float WordSize;
void main()
{
	Size = WordSize*(SCR_DIM*w) / sqrt(dot(aPos - eyePos, aPos - eyePos));
	gl_PointSize = Size;
	vec4 screenPos4 = ViewMatrix * vec4(aPos, 1.0);
	ScreenDepth = -screenPos4.z / screenPos4.w;
	gl_Position = MVP * vec4(aPos, 1.0f);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}

