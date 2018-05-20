#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
const float threshhold = 1e-24;
const float texOff=1.0/1024.0;
const vec2 xShift=vec2(texOff,0.0);
const vec2 yShift=vec2(0.0,texOff);

uniform sampler2D screenTexture;
uniform float w;  //wx=wy
#define M_PI 3.1415926535897932384626433832795
#define getDepth(Coord) texture2D(screenTexture,vec2(Coord)).r
vec3 XYCoord(vec2 Coord,float depth ){
	
	return  vec3((2.0* Coord-1.0)*depth*tan(M_PI/8.0),depth);
}

vec3 PointCoord(vec2 Coord){
	float depth=texture2D(screenTexture,Coord).r;
	return  vec3(depth*(2.0* Coord-1.0)*w,depth);
}


float divN(vec2 Coord){
	float z=getDepth(Coord);
	float zx1=getDepth(Coord+xShift);
	float zx2=getDepth(Coord-xShift);
	float zdx=(zx1==.0||zx1==.0)?0.0:(zx1-zx2)/2.0;
	float zy1=getDepth(Coord+yShift);
	float zy2=getDepth(Coord-yShift);
	float zdy=(zy1==.0||zy1==.0)?0.0:(zy1-zy2)/2.0;
	if(zx1==0.0||zx2==0.0||zy1==0.0||zy2==0.0
	||abs(zdx)>0.02||abs(zdy)>0.02
	){
		return 0.0;
	}

	

	float zdxx=zx1+zx2-2.0*z;
	float zdyy=zy1+zy2-2.0*z;
	float zx1y1=getDepth(Coord+xShift+yShift);
	float zx1y2=getDepth(Coord+xShift-yShift);
	float zx2y1=getDepth(Coord-xShift+yShift);
	float zx2y2=getDepth(Coord-xShift-yShift);
	if (zx1y1==0.0||zx1y2==0.0||zx2y1==0.0||zx2y2==0.0){return 0.0;}
	float zdxy=(zx1y1 - zx1y2 - zx2y1 + zx2y2)/4.0;

	float C=2*w;
	float C_2=C*C;
	float D = C_2*zdx*zdx + C_2*zdy*zdy+C_2*C_2*z*z;
	float Ddx = 2*C_2*zdx*zdxx + 2*C_2*zdy*zdxy + 2*C_2*C_2*z*zdx;
	float Ddy =2*C_2*zdx*zdxy + 2*C_2*zdy*zdyy + 2*C_2*C_2*z*zdy;
	float Ex = 0.5*zdx*Ddx-zdxx*D;
	float Ey = 0.5*zdy*Ddy-zdyy*D;
	float H = (C*Ex+C*Ey)/(2*D*sqrt(D));
	return H*z*w*(3.0);
}
void main()
{
	float depth=texture2D(screenTexture,TexCoords).r;
	if(depth<=threshhold){
		discard;
		return;
	}


 float H=divN(TexCoords);
 //if(abs(H)<1.5)
	if(H>0)
	{
	depth-=.5*H;
	}
	
	 
    FragColor = vec4(depth,0.0,0.0,0.0);
} 