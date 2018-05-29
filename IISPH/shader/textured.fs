#version 420 core
#extension GL_ARB_conservative_depth : enable
//layout (depth_any) out float gl_FragDepth;
out vec4 FragColor;
in vec2 TexCoord;
uniform vec3 LightPos;

uniform sampler2D texture0;
uniform int State;
void main()
{
   //gl_FragDepth = -1.0;//- 0.0*clipSpacePos.z/clipSpacePos.w;
   if(State==1){
   FragColor = texture(texture0, TexCoord);
  // if(gl_FragCoord.z>=.90)
  
   //gl_FragDepth=  (gl_FragCoord.z);
   }
   else{
   //if(gl_FragCoord.z>=.90)
   //discard;
  // gl_FragDepth=  (gl_FragCoord.z);
   FragColor = vec4(0.6,0.4,0.6,1.0);
   
   
   }
   
   
}