#version 330 core
//#extension GL_ARB_conservative_depth : enable
out vec4 Color;
//in  float Size; //in world
in float ScreenDepth;
uniform float WordSize;
uniform vec3 eyePos;
//layout(depth_any) out float gl_FragDepth;
 void main()
{

	// vec2 Coord =;
	 vec3 N;
	 N.xy = 1.0 - 2.0*gl_PointCoord.xy; //vec2(0.5 - gl_PointCoord.x, 0.5 - gl_PointCoord.y);
	 float r2 = dot(N.xy, N.xy);
	 if (r2>1.0) discard;
	 N.z = -sqrt(1.0-r2);
//	 N.z = -.5;
  float depth = ScreenDepth + 2.0* WordSize *(N.z);
  gl_FragDepth = depth/ (2.0*sqrt(dot(eyePos,eyePos)));   //clamp to [0,1]
  Color=vec4(depth,0.0,0.0,1.0);
  /* code */
}
