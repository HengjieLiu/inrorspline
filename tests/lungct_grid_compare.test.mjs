import assert from 'node:assert/strict';
import test from 'node:test';
import {checkedSpacing,rasterValue,meshSegments} from '../scripts/visualization/lungct_viewer/web/grid_compare/model.js';

test('spacing is an explicit positive display-voxel integer',()=>{
 assert.equal(checkedSpacing('8'),8);
 for(const x of ['',0,-1,2.5,NaN,Infinity,65537])assert.throws(()=>checkedSpacing(x));
});
test('raster sampling agrees with independent weighted binary-image sampling and border replication',()=>{
 const nx=21,nz=19,h=8,grid=Array.from({length:nz},(_,z)=>Array.from({length:nx},(_,x)=>+(x%h===0||z%h===0)));
 for(const [qx,qz] of [[-5,3],[3,-5],[2.1,7.25],[8.4,4.75],[20.2,18.5],[80,50],[16,8]]){
  const x=Math.min(nx-1,Math.max(0,qx)),z=Math.min(nz-1,Math.max(0,qz));let expected=0;
  for(let j=0;j<nz;j++)for(let i=0;i<nx;i++)expected+=grid[j][i]*Math.max(0,1-Math.abs(x-i))*Math.max(0,1-Math.abs(z-j));
  assert.ok(Math.abs(rasterValue(qx,qz,nx,nz,h)-expected)<1e-12);
 }
 assert.equal(rasterValue(-10,3,nx,nz,h),1,'clamped index0 is a bright border grid line');
});
test('dense mesh uses one shared phase, at most one voxel per segment, and no invented boundary lines',()=>{
 const segments=meshSegments(22,19,8);
 assert.equal(segments.length/4,3*18+3*21);
 for(let i=0;i<segments.length;i+=4){const [x,z,a,b]=segments.subarray(i,i+4);assert.equal(Math.hypot(a-x,b-z),1);assert.ok(x===a?x%8===0:z%8===0);assert.ok(a<=21&&b<=18);}
});
