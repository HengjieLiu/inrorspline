import * as THREE from "three";
import { meshSegments, checkedSpacing } from "./model.js";

export const screenVertex = `precision highp float;
in vec3 position;in vec2 uv;out vec2 vUV;
void main(){vUV=uv;gl_Position=vec4(position.xy,0.,1.);}`;
const bakeFragment = `precision highp float;precision highp sampler3D;
uniform sampler3D mapTex;uniform vec3 shape,stepMM;uniform float layer,spacing;uniform int kind;
out vec4 outColor;
float grid(vec2 v){return (mod(v.x,spacing)<.1||mod(v.y,spacing)<.1)?1.:0.;}
void main(){
 vec2 p=gl_FragCoord.xy-.5;
 vec3 tc=(vec3(p.x,layer,p.y)+.5)/shape;
 vec4 f=texture(mapTex,tc);
 if(f.a<.999){outColor=vec4(0,0,0,1);return;}
 vec2 q=p+f.xy/stepMM.xz;float c;
 if(kind==0){vec2 d=abs(mod(q+spacing*.5,spacing)-spacing*.5);vec2 aa=max(fwidth(q),vec2(.001));c=1.-min(smoothstep(.35*aa.x,1.2*aa.x,d.x),smoothstep(.35*aa.y,1.2*aa.y,d.y));}
 else{q=clamp(q,vec2(0),shape.xz-1.);vec2 a=floor(q),b=min(a+1.,shape.xz-1.),w=fract(q);c=mix(mix(grid(a),grid(vec2(b.x,a.y)),w.x),mix(grid(vec2(a.x,b.y)),grid(b),w.x),w.y);}
 outColor=vec4(c,0,0,1);
}`;
const meshVertex = `precision highp float;precision highp sampler3D;
in vec3 position;in vec4 endpoints;
uniform sampler3D mapTex;uniform vec3 shape,stepMM;uniform float layer;
out float side,valid;
void main(){
 vec2 a=endpoints.xy,b=endpoints.zw;
 vec4 fa=texture(mapTex,(vec3(a.x,layer,a.y)+.5)/shape),fb=texture(mapTex,(vec3(b.x,layer,b.y)+.5)/shape);
 valid=min(fa.a,fb.a);
 vec2 qa=a+fa.xy/stepMM.xz,qb=b+fb.xy/stepMM.xz;
 vec2 d=qb-qa,n=vec2(-d.y,d.x)/max(length(d),.00001);
 vec2 q=mix(qa,qb,position.x)+n*position.y;
 side=position.y;gl_Position=vec4(2.*(q+.5)/shape.xz-1.,0.,1.);
}`;
const meshFragment = `precision highp float;
in float side,valid;out vec4 outColor;
void main(){if(valid<.999)discard;outColor=vec4(1.-smoothstep(.35,.85,abs(side)),0,0,1);}`;
const frame = () => new Promise((r) => requestAnimationFrame(r));

export function disposeCache(cache) {
  if (!cache || cache.disposed) return;
  cache.disposed = true;
  for (const t of Object.values(cache.targets)) t.dispose();
}

export async function buildGridCache(
  renderer,
  maps,
  geometry,
  spacing,
  options = {},
) {
  spacing = checkedSpacing(spacing);
  const gl = renderer.getContext(),
    [nz, ny, nx] = geometry.shape_zyx;
  if (ny > gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS))
    throw new Error("Case exceeds GPU texture-array layer limit");
  const started = performance.now(),
    result = { targets: {}, spacing, bytes: nx * ny * nz * 9, disposed: false };
  const isCurrent = options.isCurrent || (() => true);
  const check = () => {
    if (!isCurrent()) throw new DOMException("Canceled", "AbortError");
  };
  const uniforms = {
    mapTex: { value: null },
    shape: { value: new THREE.Vector3(nx, ny, nz) },
    stepMM: { value: new THREE.Vector3(...geometry.step_lps_mm) },
    layer: { value: 0 },
    spacing: { value: spacing },
    kind: { value: 0 },
  };
  const material = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: screenVertex,
    fragmentShader: bakeFragment,
    uniforms,
    depthTest: false,
    depthWrite: false,
  });
  const plane = new THREE.PlaneGeometry(2, 2),
    scene = new THREE.Scene(),
    camera = new THREE.Camera();
  scene.add(new THREE.Mesh(plane, material));
  const segments = meshSegments(nx, nz, spacing);
  const meshGeometry = new THREE.InstancedBufferGeometry();
  meshGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(
      [0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0],
      3,
    ),
  );
  meshGeometry.setAttribute(
    "endpoints",
    new THREE.InstancedBufferAttribute(segments, 4),
  );
  meshGeometry.instanceCount = segments.length / 4;
  const meshMaterial = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: meshVertex,
    fragmentShader: meshFragment,
    uniforms,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
    transparent: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.MaxEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
  });
  const meshScene = new THREE.Scene(),
    mesh = new THREE.Mesh(meshGeometry, meshMaterial);
  mesh.frustumCulled = false;
  meshScene.add(mesh);
  let done = 0,
    last = performance.now(),
    sync;
  try {
    for (const [method, map] of Object.entries(maps)) {
      uniforms.mapTex.value = map;
      for (let kind = 0; kind < 3; kind++) {
        check();
        uniforms.kind.value = kind;
        const target = new THREE.WebGLArrayRenderTarget(nx, nz, ny, {
          format: THREE.RedFormat,
          type: THREE.UnsignedByteType,
          internalFormat: "R8",
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          generateMipmaps: false,
          depthBuffer: false,
          stencilBuffer: false,
        });
        target.texture.colorSpace = THREE.NoColorSpace;
        result.targets[`${method}_${kind}`] = target;
        for (let layer = 0; layer < ny; layer++) {
          check();
          uniforms.layer.value = layer;
          renderer.setRenderTarget(target, layer);
          renderer.setScissorTest(false);
          renderer.setViewport(0, 0, nx, nz);
          renderer.setClearColor(0x000000, 1);
          renderer.clear();
          renderer.render(kind === 2 ? meshScene : scene, camera);
          done++;
          if (performance.now() - last > 12) {
            renderer.setRenderTarget(null);
            gl.flush();
            options.onProgress?.(done / (ny * 9));
            await frame();
            check();
            last = performance.now();
          }
        }
      }
    }
    renderer.setRenderTarget(null);
    // A queued draw is not Ready until GPU completion, including every cache layer.
    sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    let status;
    do {
      check();
      status = gl.clientWaitSync(sync, 0, 0);
      if (status === gl.WAIT_FAILED)
        throw new Error("GPU cache completion failed");
      if (status === gl.TIMEOUT_EXPIRED) await frame();
    } while (status === gl.TIMEOUT_EXPIRED);
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR)
      throw new Error("GPU error while building grid cache");
    result.seconds = (performance.now() - started) / 1000;
    return result;
  } catch (e) {
    disposeCache(result);
    throw e;
  } finally {
    if (sync) gl.deleteSync(sync);
    renderer.setRenderTarget(null);
    plane.dispose();
    material.dispose();
    meshGeometry.dispose();
    meshMaterial.dispose();
  }
}

export const displayFragment = `precision highp float;precision highp sampler3D;precision highp sampler2DArray;
in vec2 vUV;out vec4 outColor;
uniform sampler3D imageTex,otherTex,maskTex,fixedMaskTex,mapTex;
uniform sampler2DArray gridTex;
uniform vec3 origin,stepMM,shape,center;uniform vec2 span,pixels;
uniform int mode;uniform float layer,level,width,diffRange;
uniform bool hasMap,anatomyOn,targetOn,lightTheme;
vec3 diverging(float x){x=clamp(x,-1.,1.);vec3 mid=vec3(.93,.914,.879);return x<0.?mix(mid,vec3(.137,.412,.71),-x):mix(mid,vec3(.777,.239,.282),x);}
float outline(sampler3D tex,vec3 t){float c=texture(tex,t).r;vec3 dx=dFdx(t)*1.25,dy=dFdy(t)*1.25;float e=max(max(abs(c-texture(tex,t+dx).r),abs(c-texture(tex,t-dx).r)),max(abs(c-texture(tex,t+dy).r),abs(c-texture(tex,t-dy).r)));return step(.001,c)*step(.001,e);}
void main(){
 vec2 local=(vUV-.5)*2.;float aspect=span.x/span.y,ratio=pixels.x/pixels.y;
 if(ratio>aspect)local.x*=ratio/aspect;else local.y*=aspect/ratio;
 vec3 bg=lightTheme?vec3(.945,.961,.98):vec3(.018,.027,.038);
 if(any(greaterThan(abs(local),vec2(1.)))){outColor=vec4(bg,1);return;}
 vec3 p=center;p.xz+=local*span*.5;vec3 q=(p-origin)/stepMM,t=(q+.5)/shape;
 vec3 color;
 if(mode!=4&&hasMap&&texture(mapTex,t).a<.999){outColor=vec4(lightTheme?vec3(.88,.90,.93):vec3(.055,.065,.08),1);return;}
 if(mode==0){float c=clamp((texture(imageTex,t).r-level)/width+.5,0.,1.);color=vec3(c);}
 else if(mode==1)color=diverging((texture(imageTex,t).r-texture(otherTex,t).r)/diffRange);
 else{float c=texture(gridTex,vec3((q.xz+.5)/shape.xz,layer)).r;color=mix(vec3(.01,.015,.02),vec3(.94,.96,.98),c);}
 if(mode!=4){if(anatomyOn)color=mix(color,vec3(.25,.94,.96),outline(maskTex,t));if(targetOn)color=mix(color,vec3(1,.9,.32),outline(fixedMaskTex,t));}
 outColor=vec4(color,1);
}`;
