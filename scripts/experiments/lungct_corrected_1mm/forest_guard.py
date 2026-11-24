"""Optional, explicitly tagged support for disconnected CorrField graphs.

Connected graphs take the identical upstream MST/TBP path. Disconnected
graphs factor into independent components: retain all existing edge costs,
run the same upstream MST/TBP in each component, and retain isolated unary
costs. No new edges, keypoints, parameters, or numerical cost rules are added.
This module is not enabled by the original campaign runner.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix,csgraph


def tree_or_forest_marginals(cost,dist,mst,tbp):
    if dist.shape[0]!=1 or dist.shape[1]!=dist.shape[2] or cost.shape[1]!=dist.shape[1]:
        raise ValueError('Expected one square graph matching the keypoint cost tensor')
    if not torch.isfinite(dist).all() or not torch.isfinite(cost).all():
        raise ValueError('Forest support cannot repair non-finite numerical costs')
    graph=csr_matrix(dist[0].detach().cpu().numpy())
    count,labels=csgraph.connected_components(graph,directed=False)
    record=dict(components=int(count),component_sizes=np.bincount(labels).tolist(),
                added_edges=0,parameter_changes=False)
    if count==1:
        edges,level=mst(dist)
        return tbp(cost,edges,level,dist),record
    output=cost.clone()
    for component in range(count):
        nodes=torch.as_tensor(np.flatnonzero(labels==component),device=cost.device)
        local_cost=cost.index_select(1,nodes)
        if len(nodes)>1:
            local_dist=dist.index_select(1,nodes).index_select(2,nodes)
            edges,level=mst(local_dist)
            local_cost=tbp(local_cost,edges,level,local_dist)
        output.index_copy_(1,nodes,local_cost)
    return output,record


def install(corrfield_module,trace):
    def compute_marginals(kpts_fix,img_fix,mind_fix,mind_mov,alpha,beta,disp_radius,disp_step,patch_radius):
        cost=alpha*corrfield_module.ssd(kpts_fix,mind_fix,mind_mov,disp_radius,disp_step,patch_radius)
        dist=corrfield_module.kpts_dist(kpts_fix,img_fix,beta)
        result,record=tree_or_forest_marginals(cost,dist,corrfield_module.minimum_spanning_tree,corrfield_module.tbp)
        trace.append(record)
        print('CORRFIELD_GRAPH',record,flush=True)
        return result
    corrfield_module.compute_marginals=compute_marginals
