"""Immutable P81/P82 rank-0 inputs and explicit grid/physical conversions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .model import REPO


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def external(path):
    path = Path(path).resolve()
    root = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    if not path.is_relative_to(root) or path.is_relative_to(REPO):
        raise ValueError("Artifact path must be external and beneath ARTIFACT_ROOT")
    return path


def atomic_json(path, data):
    path = external(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs(config, *, verify=True):
    root = Path(os.environ["ARTIFACT_ROOT"])
    manifest_path = root / config["reference_manifest"]
    manifest = json.loads(manifest_path.read_text())
    target = manifest["targets"][config["target_id"]]
    if (target["fixed_id"], target["moving_id"], target["shape"]) != ("0333", "0115", [160,224,192]):
        raise ValueError("Wrong canonical pair or geometry")
    receipts = {"reference_manifest_sha256": sha256(manifest_path)}
    for stem in ("canonical", "source"):
        if target[stem+"_sha256"] != config[stem+"_sha256"]:
            raise ValueError("Reference manifest hash changed")
        if verify and sha256(target[stem+"_path"]) != config[stem+"_sha256"]:
            raise ValueError("Immutable target field changed")
        receipts[stem+"_sha256"] = config[stem+"_sha256"]
    split = REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    if sha256(split) != "40ff80b23407c37ed8a20092f018fd9588e5df3eb95628069ec42da698aac3a0":
        raise ValueError("Canonical split changed")
    pair_directory = Path(target["pair_directory"])
    pair = json.loads((pair_directory/"pair.json").read_text())
    arrays = {name:np.load(pair_directory/f"{name}.npy") for name in
              ("fixed_image", "moving_image", "fixed_seg", "moving_seg", "mask")}
    arrays["field_mm"] = np.load(target["canonical_path"])
    affine, moving_affine = np.array(target["fixed_affine"]), np.array(target["moving_affine"])
    if not np.allclose(affine, moving_affine, atol=1e-7, rtol=0):
        raise ValueError("This experiment requires matched atlas grids")
    if not np.allclose(affine[:3,:3].T @ affine[:3,:3], np.eye(3), atol=1e-7, rtol=0):
        raise ValueError("Canonical historical surface metrics require orthogonal 1mm geometry")
    if not np.array_equal(arrays["mask"], arrays["fixed_image"] > 0):
        raise ValueError("Foreground is not raw fixed image > 0")
    if verify:
        from scripts.experiments.oasis_polyaffine_fieldfit.data import reorient
        for name in ("fixed_image", "moving_image", "fixed_seg", "moving_seg"):
            header = pair["raw_headers"][name]
            path = Path(header["path"])
            digest = sha256(path)
            if digest != pair["input_hashes"][path.name]:
                raise ValueError(f"Raw input hash mismatch: {name}")
            volume = nib.load(path)
            value, raw_affine = reorient(np.asarray(volume.dataobj, np.float32), volume.affine)
            if not np.array_equal(value, arrays[name]) or not np.allclose(raw_affine, affine, atol=1e-7, rtol=0):
                raise ValueError(f"Canonical data or affine mismatch: {name}")
            receipts[name] = dict(raw_sha256=digest, canonical_sha256=sha256(pair_directory/f"{name}.npy"))
        raw = np.asarray(nib.load(target["source_path"]).dataobj, np.float32)
        direct_mm = np.moveaxis(raw, 0, -1).astype(np.float64) @ affine[:3,:3].T
        difference = float(np.max(np.abs(direct_mm-arrays["field_mm"])))
        if difference > 1e-5:
            raise ValueError(f"Original SITReg voxel to world conversion discrepancy {difference}")
        receipts["original_to_canonical_max_mm"] = difference
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite input {name}")
    return dict(**arrays, affine=affine, target=target, receipts=receipts)


def flat_coordinates(indices, shape, dtype=torch.float32):
    d, h, w = shape
    z = indices // (h*w)
    y = (indices // w) % h
    x = indices % w
    xyz = torch.stack((x,y,z), -1).to(dtype)
    sizes = xyz.new_tensor([w,h,d])
    return 2*(xyz+.5)/sizes-1


def false_to_mm(displacement, shape, affine):
    sizes = displacement.new_tensor(list(shape)[::-1])
    linear = torch.as_tensor(affine[:3,:3], dtype=displacement.dtype, device=displacement.device)
    return (displacement*sizes/2).flip(-1) @ linear.T


def mm_to_false(displacement, shape, affine):
    linear = torch.as_tensor(affine[:3,:3], dtype=displacement.dtype, device=displacement.device)
    dhw = displacement @ torch.linalg.inv(linear).T
    return dhw.flip(-1)*2/displacement.new_tensor(list(shape)[::-1])


def false_to_true(displacement, shape):
    sizes = displacement.new_tensor(list(shape)[::-1])
    return displacement*sizes/(sizes-1)


def sample_volume(volume, coordinates, *, mode="bilinear"):
    return F.grid_sample(volume[None,None], coordinates[None,None,None],
                         mode=mode, padding_mode="zeros", align_corners=False).reshape(-1)


class Sampler:
    """Uniform shuffled epochs, including short final batches, like the release."""
    def __init__(self, mask, count, *, state=None):
        self.shape = tuple(mask.shape)
        self.ids = mask.reshape(-1).nonzero().flatten()
        self.count = count
        self.order = None
        self.cursor = 0
        if state is not None:
            self.order = None if state["order"] is None else state["order"].to(mask.device)
            self.cursor = state["cursor"]

    def next(self):
        if self.order is None or self.cursor >= len(self.ids):
            self.order = torch.randperm(len(self.ids), device=self.ids.device)
            self.cursor = 0
        indices = self.ids[self.order[self.cursor:self.cursor+self.count]]
        self.cursor += len(indices)
        return indices, flat_coordinates(indices, self.shape)

    def state(self):
        return dict(order=None if self.order is None else self.order.cpu(), cursor=self.cursor)
