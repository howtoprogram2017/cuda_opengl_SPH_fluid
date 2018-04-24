#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
const float threshhold = 1e-17;
const float texOff=1.0/1024.0;
uniform sampler2D screenTexture;

void main()
{
	float depth=texture(screenTexture,TexCoords).r;
	if(depth<=threshhold){
		discard;
		return;
	}
	vec4 data=texture(screenTexture,TexCoords);
	float leftdepth=texture(screenTexture,TexCoords+vec2(-texOff,0.0)).r;
	//const float CX=2.0/tan(M_PI/8.0);
	float rightdepth=texture(screenTexture,TexCoords+vec2(+texOff,0.0)).r;
	float topdepth=texture(screenTexture,TexCoords+vec2(+texOff,0.0)).r;

	float mixed;

	//if(leftdepth>threshhold){
	//}else{
		//mixed=50.0*abs(depth-texture(screenTexture,TexCoords+vec2(texOff,0.0)).r);
	//}
   // vec3 col = 0.5*texture(screenTexture, TexCoords).rgb;
   if(data.a>0){
   FragColor = vec4((data.a)*2.0,0.0*texture(screenTexture,TexCoords).gb, 1.0);
   }else{
    FragColor = vec4(0,-(data.a)*2.0,0, 1.0);
	}
} 