#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 centerPos;
//layout (location = 1) in vec2 aTexCoord;

//out vec2 TexCoord;

uniform mat4 MVP;  //projection * view *model

void main()
{
	gl_Position = MVP * vec4(aPos+centerPos, 1.0f);
	//TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
