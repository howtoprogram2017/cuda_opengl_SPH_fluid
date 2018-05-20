#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
const float threshhold = 1e-17;
const float texOff=1.0/1024.0;
uniform vec3 LightPos;
uniform int State;
uniform float w; 
uniform mat4 Projection;
uniform sampler2D screenTexture;
const vec3 ambientColor=vec3(0.0,0.1,0.5);
const vec3 specularColor=vec3(10.0, 8.0, 6.0);
const vec3 eyePos =vec3(0.0,0.0,0.0);
vec3 PointCoord(vec2 Coord){
	float depth=texture2D(screenTexture,Coord).r;
	return  vec3(depth*(2.0* Coord-1.0)*w,depth);
}
void main()
{
if(State == 0){
float depth=texture(screenTexture,TexCoords).r;
	if(depth<=threshhold){
		discard;
		return;
	}
	vec4 data=texture(screenTexture,TexCoords);

	vec3 normal;
	normal.xy = data.gb;
	normal.z=-(1.0-sqrt(dot(normal.xy,normal.xy)));
	float mixed;
	vec3 position = PointCoord(TexCoords);
	vec3 incomingRay = normalize( position-eyePos);
	vec3 reflectedRay = reflect(incomingRay, normal);
	vec3 toLight = LightPos-position;
	toLight = normalize(toLight);
	
	float Cos=dot(normal,normalize(toLight));
	vec3 diffuse = vec3(0.1,0.1,0.7)*(0.8*Cos+0.2);
	vec3 specular = specularColor*pow(max(0.0,dot(reflectedRay,toLight)),10.0);
   vec4 clipSpacePos = Projection* vec4(position,1.0);
   gl_FragDepth = - clipSpacePos.z/clipSpacePos.w;
   FragColor = vec4((specular+diffuse), 1.0);
}
else if(State == 1){
FragColor=texture2D(screenTexture,TexCoords);
}
	
} 