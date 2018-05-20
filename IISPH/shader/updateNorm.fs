#version 330 core
out vec4 FragColor;
uniform float w;
in vec2 TexCoords;
const float threshhold = 1e-24;
const float texOff=1.0/1024.0;
const vec2 xShift=vec2(texOff,0.0);
const vec2 yShift=vec2(0.0,texOff);
uniform sampler2D screenTexture;


vec3 PointCoord(vec2 Coord){
	float depth=texture2D(screenTexture,Coord).r;
	return  vec3(depth*(2.0* Coord-1.0)*w,depth);
}


void main()
{
	float depth=texture2D(screenTexture,TexCoords).r;
	if(depth<=threshhold){
		discard;
		return;
	}
	vec2 normXY=texture2D(screenTexture,TexCoords).gb;
	float leftdepth=texture2D(screenTexture,TexCoords+vec2(-texOff,0.0)).r;
	//vec3 viewPos=XYCoord(TexCoords,depth);
	vec3 ddx,ddy;
	//float DnormxDx,DnormyDy;
	float rightdepth= texture2D(screenTexture,TexCoords+vec2(+texOff,0.0)).r;
	if(abs(leftdepth-depth)>abs(rightdepth-depth)||true){
		ddx=-PointCoord(TexCoords-vec2(texOff/2.0,0.0))+PointCoord(TexCoords+vec2(texOff/2.0,0.0));
		//DnormxDx=texture2D(screenTexture,TexCoords+vec2(texOff,0.0)).g-normXY.x;
		//DnormxDx/=depth;
	}else{
		//ddx=-XYCoord(TexCoords+vec2(-texOff,0.0),leftdepth)+viewPos;
	}
	//const float CX=2.0/tan(M_PI/8.0);
	
	float topdepth=texture2D(screenTexture,TexCoords+vec2(0.0,texOff)).r;
	float bottomdepth=texture2D(screenTexture,TexCoords+vec2(0.0,-texOff)).r;
	if(abs(bottomdepth-depth)>abs(topdepth-depth)||true){
		ddy=-PointCoord(TexCoords-vec2(0.0,texOff/2))+PointCoord(TexCoords+vec2(0.0,texOff/2.0));
	}else{
		//ddy=viewPos- XYCoord(TexCoords+vec2(0.0,-texOff),bottomdepth);
	}
	vec3 norm=cross(ddx,ddy);
	norm=normalize(norm);
    FragColor = vec4(depth,norm.x,norm.y,norm.x);
} 