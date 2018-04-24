#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
const float threshhold = 1e-24;
const float texOff=1.0/1024.0;
uniform sampler2D screenTexture;
#define M_PI 3.1415926535897932384626433832795
vec3 XYCoord(vec2 Coord,float depth ){
	
	return  vec3((2.0* Coord-1.0)*depth*tan(M_PI/8.0),depth);
}
vec3 PointCoord(vec2 Coord){
	float depth=texture2D(screenTexture,Coord).r;
	return  vec3(2.0*(2.0* Coord-1.0)*tan(M_PI/8.0),depth);
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
	vec3 viewPos=XYCoord(TexCoords,depth);
	vec3 ddx,ddy;
	float DnormxDx,DnormyDy;
	float rightdepth=1.0* texture2D(screenTexture,TexCoords+vec2(+texOff,0.0)).r;
	if(abs(leftdepth-depth)>abs(rightdepth-depth)||true){
		//ddx=-PointCoord(TexCoords-vec2(texOff/2.0,0.0))+PointCoord(TexCoords+vec2(texOff/2.0,0.0));
		DnormxDx=texture2D(screenTexture,TexCoords+vec2(texOff/2,0.0)).g-texture2D(screenTexture,TexCoords+vec2(-texOff/2.0,0.0)).g;
		//DnormxDx/=depth;
	}else{
		ddx=-XYCoord(TexCoords+vec2(-texOff,0.0),leftdepth)+viewPos;
		DnormxDx=-texture2D(screenTexture,TexCoords+vec2(-texOff,0.0)).g+normXY.x;
	}
	//const float CX=2.0/tan(M_PI/8.0);
	
	float topdepth=texture2D(screenTexture,TexCoords+vec2(0.0,texOff)).r;
	float bottomdepth=texture2D(screenTexture,TexCoords+vec2(0.0,-texOff)).r;
	if(abs(bottomdepth-depth)>abs(topdepth-depth)||true){
		ddy=-PointCoord(TexCoords-vec2(0.0,texOff/2))+PointCoord(TexCoords+vec2(0.0,texOff/2.0));
		DnormyDy=texture2D(screenTexture,TexCoords+vec2(0.0,texOff/2)).b-texture2D(screenTexture,TexCoords+vec2(0.0,-texOff/2.0)).b;
		//DnormyDy*=depth;
	}else{
		ddy=viewPos- XYCoord(TexCoords+vec2(0.0,-texOff),bottomdepth);
		DnormyDy=(-texture2D(screenTexture,TexCoords+vec2(0.0,-texOff)).b+normXY.y)*depth;
	}
	vec2 norm=texture2D(screenTexture,TexCoords).gb;
	//norm=normalize(norm);
	//if(DnormyDy+DnormxDx<100.0){
	depth-=0.1*(DnormyDy+DnormxDx)*(0.5/sqrt(dot(norm.xy,norm.xy)));
	//}
	
	//if(leftdepth>threshhold){
	//mixed=50.0*abs(leftdepth-depth);
	//}else{
		//mixed=50.0*abs(depth-texture(screenTexture,TexCoords+vec2(texOff,0.0)).r);
	//}
   // vec3 col = 0.5*texture(screenTexture, TexCoords).rgb;
    FragColor = vec4(depth,norm.x,norm.y,20.0*( DnormyDy+DnormxDx));
} 