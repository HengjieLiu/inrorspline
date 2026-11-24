export const vertex = `varying vec2 vUV;void main(){vUV=uv;gl_Position=vec4(position.xy,0.0,1.0);}`;
export const fragment = `precision highp float;precision highp sampler3D;
varying vec2 vUV;
uniform sampler3D imageTex,otherTex,maskTex,fixedMaskTex,mapTex;
uniform vec3 origin,stepMM,shape,center,point,target;
uniform vec2 span,pixels;
uniform int axis,mode;
uniform float level,width,diffRange,mapRange,gridSpacing;
uniform bool hasMap,anatomyOn,targetOn,crossOn,targetMarker,lightTheme;
vec3 tc(vec3 p){return ((p-origin)/stepMM+.5)/shape;}
bool inside(vec3 p){vec3 q=(p-origin)/stepMM;return all(greaterThanEqual(q,vec3(-.001)))&&all(lessThanEqual(q,shape-1.0+vec3(.001)));}
vec2 project(vec3 p){return axis==0?p.xz:axis==1?p.yz:vec2(p.x,-p.y);}
vec3 diverging(float x){x=clamp(x,-1.0,1.0);vec3 mid=vec3(.93,.914,.879);return x<0.0?mix(mid,vec3(.137,.412,.71),-x):mix(mid,vec3(.777,.239,.282),x);}
float outline(sampler3D tex,vec3 t){float c=texture(tex,t).r;vec3 dx=dFdx(t)*1.25,dy=dFdy(t)*1.25;float edge=max(max(abs(c-texture(tex,t+dx).r),abs(c-texture(tex,t-dx).r)),max(abs(c-texture(tex,t+dy).r),abs(c-texture(tex,t-dy).r)));return step(.001,c)*step(.001,edge);}
void main(){
 vec2 local=(vUV-.5)*2.0;float aspect=span.x/span.y;float ratio=pixels.x/pixels.y;
 if(ratio>aspect)local.x*=ratio/aspect;else local.y*=aspect/ratio;
 vec3 bg=lightTheme?vec3(.945,.961,.98):vec3(.018,.027,.038);if(any(greaterThan(abs(local),vec2(1.0)))){gl_FragColor=vec4(bg,1);return;}
 vec3 p=center;vec2 off=local*span*.5;
 if(axis==0){p.x+=off.x;p.z+=off.y;}else if(axis==1){p.y+=off.x;p.z+=off.y;}else{p.x+=off.x;p.y-=off.y;}
 if(!inside(p)){gl_FragColor=vec4(lightTheme?vec3(.88,.90,.93):vec3(.045,.055,.073),1);return;}
 vec3 t=tc(p);vec4 field=hasMap?texture(mapTex,t):vec4(0,0,0,1);
 if(field.a<.999){gl_FragColor=vec4(lightTheme?vec3(.88,.90,.93):vec3(.055,.065,.08),1);return;}
 float image=texture(imageTex,t).r;vec3 color;
 if(mode==0){float c=clamp((image-level)/width+.5,0.0,1.0);color=vec3(c);}
 else if(mode==1)color=diverging((image-texture(otherTex,t).r)/diffRange);
 else if(mode==2){vec2 pull=p.xz+field.xy;vec2 d=abs(fract(pull/gridSpacing-.5)-.5)*gridSpacing;vec2 aa=max(fwidth(pull),vec2(.001));float lines=1.0-min(smoothstep(.35*aa.x,1.2*aa.x,d.x),smoothstep(.35*aa.y,1.2*aa.y,d.y));color=mix(vec3(.01,.015,.02),vec3(.94,.96,.98),lines);}
 else color=diverging((mode==3?field.z:field.y)/mapRange);
 if(anatomyOn)color=mix(color,vec3(.25,.94,.96),outline(maskTex,t));
 if(targetOn)color=mix(color,vec3(1,.9,.32),outline(fixedMaskTex,t));
 if(crossOn){vec2 dp=abs(project(p)-project(point));vec2 pp=max(fwidth(project(p)),vec2(.001));float line=1.0-min(smoothstep(.35*pp.x,.9*pp.x,dp.x),smoothstep(.35*pp.y,.9*pp.y,dp.y));color=mix(color,vec3(.2,.95,1),line*.65);}
 if(targetMarker){vec2 dp=(project(p)-project(target))/max(fwidth(project(p)),vec2(.001));float d=length(dp);float ring=1.0-smoothstep(.6,1.3,abs(d-4.5));color=mix(color,vec3(1,.9,.32),ring);}
 gl_FragColor=vec4(color,1);
}`;
