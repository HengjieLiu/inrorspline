"""Verify the proposed guard changes neither graph edges nor connected results."""
from pathlib import Path
import os
import sys
import numpy as np
import pytest
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/experiments/lungct_corrected_1mm'))
from forest_guard import tree_or_forest_marginals
source=Path(os.environ.get('CORRFIELD_SOURCE_ROOT','/opt/corrfield-source'))
sys.path.insert(0,str(source/'corrfield'))
from graphs import minimum_spanning_tree
from belief_propagation import tbp


def original(cost,dist):
    edges,level=minimum_spanning_tree(dist)
    return tbp(cost,edges,level,dist)


def test_connected_graph_result_is_bitwise_upstream():
    torch.manual_seed(7)
    cost=torch.rand(1,4,3,3,3)
    dist=torch.tensor([[[0.,1.,2.,0.],[1.,0.,3.,0.],[2.,3.,0.,1.],[0.,0.,1.,0.]]])
    expected=original(cost.clone(),dist)
    got,receipt=tree_or_forest_marginals(cost.clone(),dist,minimum_spanning_tree,tbp)
    assert torch.equal(expected,got)
    assert receipt['components']==1 and receipt['added_edges']==0


def test_disconnected_graph_exactly_factors_into_original_components():
    torch.manual_seed(8)
    cost=torch.rand(1,5,3,3,3)
    dist=torch.zeros(1,5,5)
    dist[0,0,1]=dist[0,1,0]=1.25
    dist[0,2,3]=dist[0,3,2]=2.5
    before=dist.clone()
    expected=cost.clone()
    for pair in [[0,1],[2,3]]:
        local=original(cost[:,pair].clone(),dist[:,pair][:,:,pair])
        expected[:,pair]=local
    got,receipt=tree_or_forest_marginals(cost,dist,minimum_spanning_tree,tbp)
    assert torch.equal(got,expected) and torch.equal(dist,before)
    assert receipt['component_sizes']==[2,2,1] and receipt['added_edges']==0
    assert torch.equal(got[:,4],cost[:,4])


def test_nonfinite_cost_is_not_silently_repaired():
    with pytest.raises(ValueError,match='non-finite'):
        tree_or_forest_marginals(torch.full((1,2,3,3,3),float('nan')),torch.zeros(1,2,2),minimum_spanning_tree,tbp)
