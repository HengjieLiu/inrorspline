"""P82 exact-operator, transfer, solver, timing and serialization contracts."""
import sys
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments/oasis_fieldfit_scaling'))
from common import Budget,Deadline,jobs
from operators import Grid,GridOperator,lattice_shape,prolong_spline
from trees import Tree,TreeOperator
from solver import solve
from worker import pyramid,save_state,load_state
from basis import Basis,_lattice_shape
from scheduler import plan_batch,retry_decision

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.lo=[-3.,-4.,-2.];self.hi=[5.,7.,6.];self.shape=(9,11,10)
        self.points=torch.stack(torch.meshgrid(*(torch.linspace(a,b,n,dtype=torch.float64)
            for a,b,n in zip(self.lo,self.hi,self.shape)),indexing='ij'),-1).reshape(-1,3)
    def old_tree(self,t):
        nodes=[]
        for i,r in enumerate(t.data[:t.count]):
            d=dict(id=i,lo=r['lo'].tolist(),hi=r['hi'].tolist(),depth=int(r['depth']),
                   parent=None if i==0 else int(r['parent']),branch=None if i==0 else int(r['branch']))
            if r['left']>=0:d.update(children=[int(r['left']),int(r['right'])],axis=int(r['axis']),mid=float(r['mid']),
                                     tau=float(r['tau']),split_index=int(r['split']))
            nodes.append(d)
        return Basis(t.family,t.params,t.lo,t.hi,nodes=nodes)
    def assert_adjoint(self,op):
        c=torch.randn(op.m,3,dtype=torch.float64);y=torch.randn(*self.shape,3,dtype=torch.float64)
        a=(op.forward(c)*y).sum();b=(c*op.adjoint(y)).sum()
        self.assertLess(float(abs(a-b)/max(1.,float(abs(a)),float(abs(b)))),1e-8)
    def test_grid_parity_norms_and_adjoint(self):
        for family,shape in [('bspline',[5,6,5]),('gaussian_translation',[4,5,4]),('gaussian_affine',[3,4,3])]:
            with self.subTest(family=family):
                m=Grid(family,shape,self.lo,self.hi);op=GridOperator(m,self.shape)
                old=Basis(family,m.params,self.lo,self.hi,lattice_shape=shape)
                design=old.features(self.points);c=torch.randn(op.m,3,dtype=torch.float64)
                torch.testing.assert_close(op.forward(c).reshape(-1,3),design@c,rtol=0,atol=1e-8)
                torch.testing.assert_close(op.norms(),design.square().sum(0).sqrt(),rtol=1e-10,atol=1e-10)
                self.assert_adjoint(op)
    def test_tree_parity_core_block_and_leaf_scores(self):
        for family in ('tree_translation_uniform','tree_affine_adaptive'):
            t=Tree(family,self.lo,self.hi)
            while t.k<18:t.split(18,np.arange(t.k,dtype=float) if 'adaptive' in family else None)
            old=self.old_tree(t);design=old.features(self.points);c=torch.randn(t.m,3,dtype=torch.float64)
            for core in (0,1<<28):
                op=TreeOperator(t,self.shape,core_bytes=core,tile=7)
                torch.testing.assert_close(op.forward(c).reshape(-1,3),design@c,rtol=0,atol=1e-8)
                torch.testing.assert_close(op.norms(),design.square().sum(0).sqrt(),rtol=1e-10,atol=1e-10)
                self.assert_adjoint(op)
            leafop=TreeOperator(t,self.shape,leaves=True,tile=3,core_bytes=0)
            weights=old.leaf_weights(self.points)
            torch.testing.assert_close(leafop.forward(torch.eye(t.k,dtype=torch.float64)).reshape(-1,t.k),weights,rtol=0,atol=1e-8)
            power=torch.rand(*self.shape,1,dtype=torch.float64)
            torch.testing.assert_close(leafop.adjoint(power).reshape(-1),weights.T@power.reshape(-1),rtol=1e-10,atol=1e-8)
    def test_split_inheritance(self):
        for family in ('tree_translation_uniform','tree_affine_uniform'):
            t=Tree(family,self.lo,self.hi);t.split(2);c=torch.randn(t.m,3,dtype=torch.float64)
            before=TreeOperator(t,self.shape).forward(c)
            t.split(3);new=torch.cat((c,c.new_zeros((t.m-len(c),3))))
            torch.testing.assert_close(TreeOperator(t,self.shape).forward(new),before,rtol=0,atol=1e-10)
    def test_knot_insertion_unequal_end_spans(self):
        final=Grid('bspline',[8,14,9],self.lo,self.hi).at_factor(1)
        old=final.at_factor(4);c=torch.randn(old.m,3,dtype=torch.float64)
        for f in (2,1):
            new=final.at_factor(f);nc=prolong_spline(old,new,c)
            torch.testing.assert_close(GridOperator(old,self.shape).forward(c),GridOperator(new,self.shape).forward(nc),rtol=0,atol=1e-8)
            old,c=new,nc
    def test_lattice_selection_preserved(self):
        for minimum in (2,4):
            for cap in (64,128,252,512,1008,1024):
                for lengths in ([8,11,8],[159,223,191]):
                    self.assertEqual(lattice_shape(cap,lengths,minimum),_lattice_shape(cap,lengths,minimum))
    def test_zero_affine_and_least_squares(self):
        for family in ('bspline','gaussian_translation','gaussian_affine'):
            m=Grid(family,[4,4,4],self.lo,self.hi);op=GridOperator(m,self.shape)
            old=Basis(family,m.params,self.lo,self.hi,lattice_shape=m.shape)
            matrix=torch.randn(3,3,dtype=torch.float64)*.1;translation=torch.tensor([1.,2.,3.],dtype=torch.float64)
            c=old.coefficients_for_affine(matrix,translation)
            y=(self.points@matrix.T+translation).reshape(*self.shape,3)
            torch.testing.assert_close(op.forward(c),y,rtol=0,atol=1e-8)
            fitted,meta=solve(op,y,iterations=1000)
            self.assertTrue(meta['converged'],meta)
            self.assertLess(float((op.forward(fitted)-y).square().mean().sqrt()),1e-4)
            zero,zmeta=solve(op,torch.zeros_like(y));self.assertTrue(zmeta['converged']);self.assertEqual(float(zero.abs().max()),0)
    def test_rank_deficient_projection_matches_dense(self):
        shape=(4,4,4);m=Grid('gaussian_affine',[3,3,3],self.lo,self.hi);op=GridOperator(m,shape)
        y=torch.randn(*shape,3,dtype=torch.float64)
        c,meta=solve(op,y,iterations=2000)
        eye=torch.eye(op.m,dtype=torch.float64);design=op.forward(eye).reshape(-1,op.m)
        ref=torch.linalg.lstsq(design,y.reshape(-1,3),driver='gelsd',rcond=1e-12).solution
        self.assertLess(float((op.forward(c)-op.forward(ref)).square().mean().sqrt()),1e-3)
    def test_gaussian_transfer_is_measured_projection(self):
        for family in ('gaussian_translation','gaussian_affine'):
            fine=Grid(family,[5,5,5],self.lo,self.hi);coarse=fine.at_factor(2)
            old=GridOperator(coarse,self.shape);new=GridOperator(fine,self.shape)
            c=torch.randn(coarse.m,3,dtype=torch.float64);inherited=old.forward(c)
            projected,meta=solve(new,inherited,iterations=2000)
            actual=float((new.forward(projected)-inherited).square().sum(-1).mean())
            self.assertAlmostEqual(actual,meta['mse_mm2'],places=9)
            self.assertLess(actual,float(inherited.square().sum(-1).mean())*.1)
    def test_pyramid_translation_units(self):
        v=torch.tensor([3.,-5.,2.],dtype=torch.float64);field=v.expand(16,24,20,3).clone()
        for f in (4,2,1):
            p=pyramid(field,f);self.assertEqual(tuple(p.shape),(16//f,24//f,20//f,3))
            torch.testing.assert_close(p,v.expand_as(p),rtol=0,atol=1e-10)
    def test_deadline_shared_cumulative(self):
        b=Budget(1200,spent=121);b.stage(120)
        with self.assertRaises(Deadline):b.check()
        b.stage(360);b.check();b.spent=1201
        with self.assertRaises(Deadline):b.check()
        m=Grid('bspline',[4,4,4],self.lo,self.hi);op=GridOperator(m,self.shape)
        initial=torch.ones(m.m,3,dtype=torch.float64)
        c,meta=solve(op,torch.zeros(*self.shape,3,dtype=torch.float64),initial,b.check)
        torch.testing.assert_close(c,initial);self.assertFalse(meta['converged'])
    def test_serialization(self):
        for model in (Grid('bspline',[5,6,5],self.lo,self.hi).at_factor(1),Tree('tree_affine_uniform',self.lo,self.hi)):
            if isinstance(model,Tree):model.split(2)
            c=torch.randn(model.m,3,dtype=torch.float64)
            with tempfile.TemporaryDirectory() as d:
                save_state(d,model,c,{'stage_index':0,'active_seconds':101})
                m2,c2,meta=load_state(d,'cpu')
                torch.testing.assert_close(c,c2);self.assertEqual(m2.metadata(),model.metadata());self.assertEqual(meta['active_seconds'],101)
    def test_matrix_size(self):
        js=jobs();self.assertEqual(len(js),168);self.assertEqual(len({j['id'] for j in js}),168)
    def test_memory_tiers_and_redistribution(self):
        candidates=[(str(i),5000) for i in range(20)]
        self.assertEqual(len(plan_batch(48000,[],candidates,16)),8)
        self.assertEqual(len(plan_batch(24000,[],candidates,16)),4)
        self.assertEqual(len(plan_batch(3000,[],candidates,4)),0)
        self.assertEqual(len(plan_batch(48000,[],candidates,4)),4)
        # A large first case must not block smaller cases on a busy GPU.
        self.assertEqual(plan_batch(10000,[],[('large',16000),('small',5000)],16),[('small',5000)])
        self.assertEqual(plan_batch(10000,[dict(required=8000,current=1000)],candidates,16),[])
    def test_oom_keeps_time_and_reduces_tiles(self):
        j=dict(tile=128,required_mib=5000,spent=600)
        r=retry_decision(j,dict(active_seconds=700),49000)
        self.assertEqual(r['spent'],700);self.assertEqual(r['tile'],64);self.assertEqual(r['status'],'pending')
        r=retry_decision(dict(tile=1,required_mib=46440,spent=900,exclusive=True),dict(active_seconds=950),49000)
        self.assertEqual(r['status'],'resource_limited');self.assertEqual(r['spent'],950)

if __name__=='__main__':unittest.main()
