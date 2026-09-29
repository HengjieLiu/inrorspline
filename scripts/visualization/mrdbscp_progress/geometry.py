"""CPU-safe 3-D pull-warp sampling on a requested display plane.

Spatial array axes are those of the existing reoriented OASIS loader. XYZ
displacement components follow grid_sample (reverse array-axis order), NOT
anatomical axis names. No 2-D approximation to the displacement is made.
"""
import numpy as np
import torch
import torch.nn.functional as F


def slice_grid(shape, axis, index):
    if axis not in (0, 1, 2) or not 0 <= index < shape[axis]:
        raise ValueError("Slice outside volume")
    vectors = [torch.linspace(-1., 1., n, dtype=torch.float32) for n in shape]
    vectors[axis] = vectors[axis][index:index + 1]
    z, y, x = torch.meshgrid(*vectors, indexing="ij")
    return torch.stack((x, y, z), dim=-1)


def sample_plane(volume, coords, axis, mode="bilinear"):
    sampled = F.grid_sample(volume[None, None], coords[None], mode=mode,
                            padding_mode="zeros", align_corners=True)
    return sampled[0, 0].squeeze(axis).numpy()


def display_plane(plane):
    """Radiological axial display: screen left R, right L, top A, bottom P."""
    return np.fliplr(np.asarray(plane).T)


def reference_grid(shape, axis=2, spacing=8):
    if spacing <= 0 or axis not in (0, 1, 2):
        raise ValueError("Invalid grid parameters")
    grid = torch.zeros(shape, dtype=torch.float32)
    for direction in range(3):
        if direction != axis:
            select = [slice(None)] * 3
            select[direction] = slice(0, None, spacing)
            grid[tuple(select)] = 1.
    return grid


def preview_planes(batch, field, index=96, spacing=8):
    """Sample full 3-D source volumes at one axial plane, including through-plane motion."""
    axis = 2
    coords = slice_grid(field.shape[:3], axis, index)
    coords += torch.from_numpy(np.array(field[:, :, index:index + 1, :], copy=True))
    planes = {name: batch[name][:, :, index].numpy()
              for name in ("moving", "fixed", "moving_seg", "fixed_seg")}
    planes["warped"] = sample_plane(batch["moving"], coords, axis)
    planes["warped_seg"] = sample_plane(batch["moving_seg"], coords, axis, "nearest")
    planes["grid"] = sample_plane(reference_grid(field.shape[:3], axis, spacing), coords, axis)
    allowed = set(torch.unique(batch["moving_seg"]).tolist()) | {0.}
    if not set(np.unique(planes["warped_seg"])).issubset(allowed):
        raise ValueError("Nearest-neighbor label warp created new labels")
    return {key: display_plane(value) for key, value in planes.items()}
