#version 330 core
out vec4 Color;

 void main()
{
  Color=vec4(1.0,1.0,1.0,1.0);
  /* code */
}
 //vec2 normXY = texture2D(screenTexture, TexCoords).gb;
 //float leftdepth = texture2D(screenTexture, TexCoords + vec2(-texOff, 0.0)).r;
 //vec3 viewPos = XYCoord(TexCoords, depth);
 //vec3 ddx, ddy;
 //float DnormxDx, DnormyDy;
 //float rightdepth = 1.0* texture2D(screenTexture, TexCoords + vec2(+texOff, 0.0)).r;
 //if (abs(leftdepth - depth)>abs(rightdepth - depth) || true) {
 //	ddx = -PointCoord(TexCoords - vec2(texOff / 2.0, 0.0)) + PointCoord(TexCoords + vec2(texOff / 2.0, 0.0));
 //	DnormxDx = texture2D(screenTexture, TexCoords + vec2(texOff / 2, 0.0)).g - texture2D(screenTexture, TexCoords + vec2(-texOff / 2.0, 0.0)).g;
 //	DnormxDx/=depth;
 //}
 //else {
 //	ddx = -XYCoord(TexCoords + vec2(-texOff, 0.0), leftdepth) + viewPos;
 //	DnormxDx = -texture2D(screenTexture, TexCoords + vec2(-texOff, 0.0)).g + normXY.x;
 //}
 //const float CX=2.0/tan(M_PI/8.0);
 //
 //float topdepth = texture2D(screenTexture, TexCoords + vec2(0.0, texOff)).r;
 //float bottomdepth = texture2D(screenTexture, TexCoords + vec2(0.0, -texOff)).r;
 //if (abs(bottomdepth - depth)>abs(topdepth - depth) || true) {
 //	ddy = -PointCoord(TexCoords - vec2(0.0, texOff / 2)) + PointCoord(TexCoords + vec2(0.0, texOff / 2.0));
 //	DnormyDy = texture2D(screenTexture, TexCoords + vec2(0.0, texOff / 2)).b - texture2D(screenTexture, TexCoords + vec2(0.0, -texOff / 2.0)).b;
 //	DnormyDy*=depth;
 //}
 //else {
 //	ddy = viewPos - XYCoord(TexCoords + vec2(0.0, -texOff), bottomdepth);
 //	DnormyDy = (-texture2D(screenTexture, TexCoords + vec2(0.0, -texOff)).b + normXY.y)*depth;
 //}
 //vec3 norm = cross(ddx, ddy);
 //norm = normalize(norm);