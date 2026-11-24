// Hardware browser acceptance. All outputs belong in the external caller root.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const out=process.env.VIEWER_VALIDATION_ROOT,url=process.env.VIEWER_URL;
if(!out||!url||!process.env.VIEWER_PLAYWRIGHT_MODULE)throw new Error('Set external VIEWER_VALIDATION_ROOT, VIEWER_URL and VIEWER_PLAYWRIGHT_MODULE');
await fs.mkdir(out,{recursive:true});
const {chromium}=await import(pathToFileURL(process.env.VIEWER_PLAYWRIGHT_MODULE));
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--use-angle=gl-egl','--enable-gpu','--ignore-gpu-blocklist','--disable-software-rasterizer','--js-flags=--max-old-space-size=4096']});
const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('request',r=>requests.push(r.url()));
const report={browser:browser.version(),viewport:{width:1920,height:1080},cases:[],errors};
const ready=async id=>page.waitForFunction(id=>window.gridViewer?.ready&&(!id||gridViewer.diagnostics.case===id),id,{timeout:180000});
const settle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
try{
 // Use a real HTTP navigation so Chromium records the server address space;
 // a fully intercepted document incorrectly triggers local-network CORS checks.
 await page.route('**/app.js',r=>r.fulfill({contentType:'text/javascript',body:''}));
 await page.goto(url+'/index.html');
 await page.evaluate(()=>{document.body.innerHTML='<canvas id="test"></canvas>';});
 report.synthetic=await page.evaluate(async()=>{
  const THREE=await import('three'),{buildGridCache,disposeCache}=await import('/grid_compare/cache.js');
  const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('test')});renderer.autoClear=false;
  renderer.debug.onShaderError=()=>{throw new Error('Synthetic shader failed');};
  const gl=renderer.getContext(),nx=33,nz=33,ny=3,step=[.8,1.2,-1.25];
  const geometry={shape_zyx:[nz,ny,nx],step_lps_mm:step};
  const results=[];
  const check=(ok,message)=>{if(!ok)throw new Error(message);};
  const fixtures=[['zero',()=>[0,0]],['translation',()=>[2,2]],['scaling',(x,z)=>[.25*x,-.15*z]],['nonlinear',(x,z)=>[1.5*Math.sin(z/5),.8*Math.sin(x/6)]],['invalid',()=>[0,0]]];
  for(const [name,offset] of fixtures){
   const data=new Float32Array(nx*ny*nz*4);
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const i=((z*ny+y)*nx+x)*4,[dx,dz]=offset(x,z);data[i]=dx*step[0];data[i+1]=dz*step[2];data[i+3]=name==='invalid'&&x>=15&&x<=17&&z>=15&&z<=17?0:1;}
   const tex=new THREE.Data3DTexture(data,nx,ny,nz);tex.type=THREE.FloatType;tex.format=THREE.RGBAFormat;tex.minFilter=tex.magFilter=THREE.LinearFilter;tex.unpackAlignment=1;tex.needsUpdate=true;
   const cache=await buildGridCache(renderer,{mrdbscp:tex,ptvreg:tex,corrfield:tex},geometry,8);
   const arrays=[];
   for(let k=0;k<3;k++){renderer.setRenderTarget(cache.targets[`mrdbscp_${k}`],1);const b=new Uint8Array(nx*nz*4);gl.readPixels(0,0,nx,nz,gl.RGBA,gl.UNSIGNED_BYTE,b);check(gl.getError()===0,'synthetic readback');arrays.push(b.filter((_,i)=>i%4===0));}
   const at=(k,x,z)=>arrays[k][z*nx+x]/255;
   // Independent source image and bilinear sampler, with edge replication.
   const binary=Array.from({length:nz},(_,z)=>Array.from({length:nx},(_,x)=>+(x%8===0||z%8===0)));
   let maxError=0;
   for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
    const [dx,dz]=offset(x,z),qx=Math.max(0,Math.min(nx-1,x+dx)),qz=Math.max(0,Math.min(nz-1,z+dz));
    const x0=Math.floor(qx),z0=Math.floor(qz),fx=qx-x0,fz=qz-z0;
    let expected=(1-fz)*((1-fx)*binary[z0][x0]+fx*binary[z0][Math.min(nx-1,x0+1)])+fz*((1-fx)*binary[Math.min(nz-1,z0+1)][x0]+fx*binary[Math.min(nz-1,z0+1)][Math.min(nx-1,x0+1)]);
    if(name==='invalid'&&x>=15&&x<=17&&z>=15&&z<=17)expected=0;
    maxError=Math.max(maxError,Math.abs(at(1,x,z)-expected));
   }
   check(maxError<=1/255+.00002,`${name} raster error ${maxError}`);
   if(name==='zero')for(let k=0;k<3;k++)for(const [x,z] of [[8,3],[16,3],[3,8]])check(at(k,x,z)>.95,`zero alignment kind${k}`);
   if(name==='translation')for(let k=0;k<3;k++)check(at(k,k===2?10:6,3)>.8,`translation sign kind${k}`);
   if(name==='scaling'){check(at(0,6,3)>.5,'analytic scaling compression');check(at(2,10,3)>.95,'mesh scaling expansion');}
   if(name==='invalid')for(let k=0;k<3;k++)check(at(k,16,16)===0,`invalid domain kind${k}`);
   results.push({name,raster_max_coverage_error:maxError,cache_seconds:cache.seconds});
   renderer.setRenderTarget(null);disposeCache(cache);tex.dispose();
  }
  // Cancellation must release allocations, rather than return a partial cache.
  const t=new THREE.Data3DTexture(new Float32Array([0,0,0,1]),1,1,1);t.format=THREE.RGBAFormat;t.type=THREE.FloatType;t.needsUpdate=true;
  let canceled=false;try{await buildGridCache(renderer,{mrdbscp:t,ptvreg:t,corrfield:t},geometry,8,{isCurrent:()=>false});}catch(e){canceled=e.name==='AbortError';}check(canceled,'cancel guard');t.dispose();
  renderer.dispose();renderer.forceContextLoss();return {passed:true,fixtures:results,cancel_guard:true};
 });
 await page.unroute('**/app.js');await page.goto(url);await ready();
 const cdp=await page.context().newCDPSession(page);
 for(const id of ['4dct_01','4dct_02','copd_01','copd_10']){
  if(await page.evaluate(()=>gridViewer.diagnostics.case)!==id){await page.evaluate(id=>{gridViewer.loadCase(id);},id);await ready(id);}
  const start=requests.length,d=await page.evaluate(()=>gridViewer.diagnostics),m=await page.evaluate(()=>gridViewer.manifest);
  console.log(`Checking ${id}: ${d.cacheHistory.at(-1).seconds.toFixed(2)}s cache`);
  assert.match(d.renderer,/NVIDIA/);assert.equal(d.cacheVolumes,9);assert.equal(d.cacheBytes,m.geometry.shape_zyx.reduce((a,b)=>a*b)*9);assert.equal(d.spacing,8);
  assert.equal(await page.locator('#comparison .view:not(.blank)').count(),19);assert.equal(await page.locator('#comparison .blank').count(),6);
  assert.equal(await page.locator('#landmarks-panel').locator('*').count(),0);
  if(id==='copd_01')assert.equal(m.geometry.shape_zyx[0],272);
  await page.evaluate(()=>{window.scrollTo(0,0);gridViewer.requestRender();});await settle();await page.screenshot({path:path.join(out,id+'_top.png')});
  const sample=await page.evaluate(()=>[0,1,2].map(k=>{const a=gridViewer.readCache('mrdbscp',k,gridViewer.diagnostics.slice);return {kind:k,nonzero:a.filter(x=>x>0).length,sum:a.reduce((a,b)=>a+b,0)};}));assert.ok(sample.every(x=>x.nonzero>100));
  const oldSlice=d.slice;
  await page.locator('#grid-spacing').fill('12');await page.locator('#apply-grids').click();
  const gating=await page.evaluate(()=>{const before=gridViewer.diagnostics.slice,ready=gridViewer.ready;gridViewer.setSlice(0);return {ready,before,after:gridViewer.diagnostics.slice,disabled:document.getElementById('slice').disabled};});
  assert.equal(gating.ready,false);assert.equal(gating.disabled,true);assert.equal(gating.before,gating.after);await ready(id);assert.equal((await page.evaluate(()=>gridViewer.diagnostics)).spacing,12);
  await page.evaluate(()=>{gridViewer.rebuild(3);gridViewer.rebuild(16);gridViewer.rebuild(8);});await ready(id);
  assert.equal((await page.evaluate(()=>gridViewer.diagnostics)).spacing,8);assert.equal((await page.evaluate(()=>gridViewer.diagnostics)).gpuTextures,d.gpuTextures);
  await page.getByRole('button',{name:'MR-D-BSCP later',exact:true}).click();assert.deepEqual(await page.evaluate(()=>gridViewer.diagnostics.order),['ptvreg','mrdbscp','corrfield']);assert.equal(await page.evaluate(()=>gridViewer.diagnostics.slice),oldSlice);await page.getByRole('button',{name:'MR-D-BSCP earlier',exact:true}).click();
  await page.locator('#theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'light');await page.locator('#theme-toggle').click();
  await page.locator('#tab-landmarks').click();assert.equal(await page.locator('#landmarks-panel').isVisible(),true);await page.locator('#tab-images').click();
  const row5=page.locator('.view[data-method="mrdbscp"][data-mode="4"]');await row5.scrollIntoViewIfNeeded();await settle();const withContour=await row5.screenshot();
  await page.locator('#anatomy').uncheck();await page.locator('#target').uncheck();await row5.scrollIntoViewIfNeeded();await settle();assert.ok(withContour.equals(await row5.screenshot()),'row5 ignores contour toggles');
  await page.locator('#anatomy').check();await page.locator('#target').check();await row5.scrollIntoViewIfNeeded();await settle();await page.screenshot({path:path.join(out,id+'_grids.png')});
  // Actual wheel listener synchronizes every cell through the one slice state.
  const before=await page.evaluate(()=>gridViewer.diagnostics.slice);await row5.hover();await page.mouse.wheel(0,80);await settle();assert.equal(await page.evaluate(()=>gridViewer.diagnostics.slice),Math.min(before+1,m.geometry.shape_zyx[1]-1));
  const run=async region=>{
   if(region==='top')await page.evaluate(()=>window.scrollTo(0,0));else await row5.scrollIntoViewIfNeeded();await settle();
   return page.evaluate(async()=>{const start=performance.now();let n=0;while(performance.now()-start<2500){gridViewer.setSlice((gridViewer.diagnostics.slice+1)%gridViewer.manifest.geometry.shape_zyx[1]);await new Promise(r=>requestAnimationFrame(r));n++;}const d=gridViewer.diagnostics,a=d.latencies.slice(-n).sort((a,b)=>a-b);return {fps:n*1000/(performance.now()-start),p95_ms:a[Math.floor(a.length*.95)],frames:n};});
  };
  const top=await run('top'),grids=await run('grids');for(const v of [top,grids]){assert.ok(v.fps>=30,`fps ${v.fps}`);assert.ok(v.p95_ms<100,`response ${v.p95_ms}`);}
  assert.equal(requests.length,start,'zero requests after Ready, including recaching');
  await cdp.send('HeapProfiler.collectGarbage');const heap=await cdp.send('Runtime.getHeapUsage');
  report.cases.push({id,geometry:m.geometry,diagnostics:await page.evaluate(()=>gridViewer.diagnostics),sample,top,grids,heap,post_ready_requests:requests.length-start,gating});
  await fs.writeFile(path.join(out,'browser_report.json'),JSON.stringify({...report,status:'running'},null,2));
 }
 // Cancel both an incomplete download and an active grid build, then return to
 // the largest case twice. One case and exactly nine cache volumes must remain.
 await page.evaluate(()=>{gridViewer.loadCase('4dct_02');gridViewer.loadCase('copd_01');gridViewer.loadCase('copd_10');});await ready('copd_10');
 await page.evaluate(()=>{gridViewer.rebuild(1);gridViewer.loadCase('4dct_01');});await ready('4dct_01');
 for(let i=0;i<2;i++){await page.evaluate(()=>{gridViewer.loadCase('copd_10');});await ready('copd_10');await cdp.send('HeapProfiler.collectGarbage');report.switching??=[];report.switching.push({diagnostics:await page.evaluate(()=>gridViewer.diagnostics),heap:await cdp.send('Runtime.getHeapUsage')});}
 assert.ok(report.switching.every(x=>x.diagnostics.gpuTextures===24&&x.diagnostics.geometries===1));
 const a=report.switching[0].heap,b=report.switching[1].heap;assert.ok(b.usedSize<a.usedSize+16*2**20,'JS heap bounded');if(a.backingStorageSize)assert.ok(b.backingStorageSize<a.backingStorageSize+16*2**20,'ArrayBuffers bounded');
 assert.deepEqual(errors,[]);report.status='accepted';
}catch(e){report.status='failed';report.failure=e.stack;console.error(e);try{await page.screenshot({path:path.join(out,'failure.png')});}catch{}process.exitCode=1;}
finally{await fs.writeFile(path.join(out,'browser_report.json'),JSON.stringify(report,null,2));await browser.close();}
console.log(JSON.stringify({status:report.status,cases:report.cases.map(x=>({id:x.id,top:x.top,grids:x.grids})),errors}));
