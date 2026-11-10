"""P81 physical-coordinate linear dictionaries and field-preserving affine trees.

Scalar features have shape [N,M]; coefficients have shape [M,3] in world mm.
Grid XYZ is deliberately absent from this module. Callers chunk native grids.
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import torch


FAMILIES = (
    "bspline", "gaussian_translation", "gaussian_affine",
    "tree_translation_uniform", "tree_translation_adaptive",
    "tree_affine_uniform", "tree_affine_adaptive", "global_affine",
)


def _lattice_shape(cap: int, lengths: list[float], minimum: int) -> list[int]:
    """Maximize occupied budget among reasonably isotropic tensor lattices."""
    if cap < minimum ** 3:
        raise ValueError(f"Need at least {minimum ** 3} lattice coefficients")
    scale = (math.prod(lengths) / cap) ** (1.0 / 3.0)
    upper = [max(minimum + 2, int(math.ceil(2 * length / scale)) + 4)
             for length in lengths]
    candidates = []
    for shape in itertools.product(*(range(minimum, n + 1) for n in upper)):
        count = math.prod(shape)
        if count > cap:
            continue
        # A cubic clamped spline with n controls has n-3 nonempty knot spans.
        intervals = [n - 3 if minimum == 4 else n - 1 for n in shape]
        spacing = [length / n for length, n in zip(lengths, intervals)]
        anisotropy = max(spacing) / min(spacing)
        candidates.append((anisotropy, count, shape))
    good = [row for row in candidates if row[0] <= 1.6]
    if good:
        chosen = min(good, key=lambda row: (-row[1], row[0], row[2]))
    else:
        chosen = min(candidates, key=lambda row: (row[0], -row[1], row[2]))
    return list(chosen[2])


def _affine_features(points: torch.Tensor, lo, hi) -> torch.Tensor:
    lower, upper = points.new_tensor(lo), points.new_tensor(hi)
    normalized = (points - (lower + upper) / 2) / ((upper - lower) / 2)
    return torch.cat((torch.ones_like(points[:, :1]), normalized), dim=1)


def _knots(n: int, lo: float, hi: float, like: torch.Tensor) -> torch.Tensor:
    middle = torch.linspace(lo, hi, n - 2, device=like.device, dtype=like.dtype)
    return torch.cat((middle[:1].repeat(3), middle, middle[-1:].repeat(3)))


def _bspline_1d(x: torch.Tensor, knots: torch.Tensor, n: int) -> torch.Tensor:
    # Cox-de Boor with a closed last interval makes the boundary interpolate
    # the last control while retaining the left derivative of the polynomial.
    result = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).to(x.dtype)
    last = (x == knots[-1]).to(x.dtype)
    marker = torch.zeros_like(knots[:-1]); marker[n - 1] = 1
    result = result + last[:, None] * marker[None]
    for degree in range(1, 4):
        count = len(knots) - degree - 1
        left_denom = knots[degree:degree + count] - knots[:count]
        right_denom = knots[degree + 1:degree + count + 1] - knots[1:count + 1]
        safe_left = torch.where(left_denom != 0, left_denom, torch.ones_like(left_denom))
        safe_right = torch.where(right_denom != 0, right_denom, torch.ones_like(right_denom))
        left = ((x[:, None] - knots[:count]) / safe_left) * (left_denom != 0)
        right = ((knots[degree + 1:degree + count + 1] - x[:, None]) / safe_right) * (right_denom != 0)
        result = left * result[:, :count] + right * result[:, 1:count + 1]
    return result


@dataclass
class Basis:
    family: str
    budget_cap: int
    domain_min: list[float]
    domain_max: list[float]
    lattice_shape: list[int] | None = None
    nodes: list[dict[str, Any]] = field(default_factory=list)
    gaussian_sigma_over_h: float = .85
    tree_gate_width_fraction: float = .12
    max_depth: int = 24
    min_box_size_mm: float = 1.0
    version: int = 1
    _tensor_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown family {self.family}")
        if len(self.domain_min) != 3 or len(self.domain_max) != 3:
            raise ValueError("Physical box requires three axes")
        if any(not math.isfinite(float(x)) for x in self.domain_min + self.domain_max):
            raise ValueError("Nonfinite physical box")
        if any(b <= a for a, b in zip(self.domain_min, self.domain_max)):
            raise ValueError("Physical box must have positive extent")
        if self.gaussian_sigma_over_h <= 0 or self.tree_gate_width_fraction <= 0:
            raise ValueError("Positive Gaussian and gate widths required")

    @property
    def is_tree(self):
        return self.family.startswith("tree_")

    @property
    def is_affine_tree(self):
        return self.family.startswith("tree_affine_")

    @property
    def is_adaptive(self):
        return self.family.endswith("_adaptive")

    @property
    def leaf_count(self) -> int:
        if self.is_tree:
            return sum("children" not in node for node in self.nodes)
        return math.prod(self.lattice_shape) if self.lattice_shape else 1

    @property
    def feature_count(self) -> int:
        if self.family == "global_affine":
            return 4
        if self.is_tree:
            return 4 + (self.leaf_count - 1) * (4 if self.is_affine_tree else 1)
        if self.family == "gaussian_translation":
            return self.leaf_count + 3
        if self.family == "gaussian_affine":
            return 4 * self.leaf_count
        return self.leaf_count

    @property
    def actual_motion_params(self) -> int:
        return 3 * self.feature_count

    @property
    def target_leaf_count(self) -> int:
        if not self.is_tree:
            return self.leaf_count
        return self.budget_cap // 12 if self.is_affine_tree else (self.budget_cap - 9) // 3

    def to_dict(self) -> dict:
        return {key: copy.deepcopy(getattr(self, key)) for key in (
            "family", "budget_cap", "domain_min", "domain_max", "lattice_shape",
            "nodes", "gaussian_sigma_over_h", "tree_gate_width_fraction",
            "max_depth", "min_box_size_mm", "version")}

    @classmethod
    def from_dict(cls, data: dict) -> "Basis":
        if data.get("version") != 1:
            raise ValueError("Unsupported basis serialization version")
        result = cls(**copy.deepcopy(data))
        if result.actual_motion_params > result.budget_cap:
            raise ValueError("Serialized basis exceeds declared motion cap")
        return result

    def _constants(self, points: torch.Tensor) -> dict:
        key = (str(points.device), points.dtype)
        if key in self._tensor_cache:
            return self._tensor_cache[key]
        result = {}
        if self.lattice_shape:
            axes = [torch.linspace(lo, hi, count, device=points.device, dtype=points.dtype)
                    for lo, hi, count in zip(self.domain_min, self.domain_max, self.lattice_shape)]
            result["centers"] = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            result["spacing"] = points.new_tensor([(hi - lo) / (n - 1) for lo, hi, n
                                                    in zip(self.domain_min, self.domain_max, self.lattice_shape)])
            if self.family == "bspline":
                result["knots"] = [_knots(n, lo, hi, points) for n, lo, hi
                                    in zip(self.lattice_shape, self.domain_min, self.domain_max)]
        if self.is_tree:
            internal = sorted((n for n in self.nodes if "children" in n), key=lambda n: n["split_index"])
            result["internal"] = internal
            depth_groups = []
            previous_ids = {}
            concatenated_indices = []
            for depth in range(max((n["depth"] for n in internal), default=-1) + 1):
                group = [n for n in internal if n["depth"] == depth]
                if not group:
                    continue
                meta = {"axis": torch.tensor([n["axis"] for n in group], device=points.device),
                        "mid": points.new_tensor([n["mid"] for n in group]),
                        "tau": points.new_tensor([n["tau"] for n in group])}
                if depth:
                    meta["parent"] = torch.tensor([previous_ids[n["parent"]] for n in group], device=points.device)
                    meta["right"] = torch.tensor([n["branch"] == 1 for n in group], device=points.device)
                previous_ids = {n["id"]: i for i, n in enumerate(group)}
                concatenated_indices += [n["split_index"] for n in group]
                depth_groups.append(meta)
            result["depth_groups"] = depth_groups
            result["reorder"] = torch.tensor(sorted(range(len(internal)), key=lambda i: concatenated_indices[i]), device=points.device, dtype=torch.long)
            if internal:
                result["centers"] = points.new_tensor([[(a + b) / 2 for a, b in zip(n["lo"], n["hi"])] for n in internal])
                result["scales"] = points.new_tensor([[(b - a) / 2 for a, b in zip(n["lo"], n["hi"])] for n in internal])
                internal_by_id = {n["id"]: n["split_index"] for n in internal}
                leaves = [n for n in self.nodes if "children" not in n]
                result["leaf_parents"] = torch.tensor([internal_by_id[n["parent"]] for n in leaves], device=points.device)
                result["leaf_right"] = torch.tensor([n["branch"] == 1 for n in leaves], device=points.device)
        self._tensor_cache[key] = result
        return result

    def _tree_components(self, points: torch.Tensor):
        """Return path weights and right-child contrasts in stable split order."""
        constants = self._constants(points)
        all_path, all_q = [], []
        previous_path = previous_gate = None
        for meta in constants["depth_groups"]:
            gate = torch.sigmoid((points[:, meta["axis"]] - meta["mid"]) / meta["tau"])
            if previous_path is None:
                path = torch.ones_like(gate)
            else:
                parent_gate = previous_gate[:, meta["parent"]]
                path = previous_path[:, meta["parent"]] * torch.where(meta["right"], parent_gate, 1 - parent_gate)
            all_path.append(path); all_q.append(path * gate)
            previous_path, previous_gate = path, gate
        if not all_q:
            empty = points.new_empty((len(points), 0))
            return empty, empty
        order = constants["reorder"]
        return torch.cat(all_path, 1)[:, order], torch.cat(all_q, 1)[:, order]

    def features(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point():
            raise ValueError("Expected floating physical points [N,3]")
        if self.family == "global_affine":
            return _affine_features(points, self.domain_min, self.domain_max)
        if self.is_tree:
            root = _affine_features(points, self.domain_min, self.domain_max)
            if self.leaf_count == 1:
                return root
            _, contrasts = self._tree_components(points)
            if self.is_affine_tree:
                const = self._constants(points)
                local = (points[:, None, :] - const["centers"]) / const["scales"]
                local = torch.cat((torch.ones_like(local[:, :, :1]), local), -1)
                contrasts = (contrasts[:, :, None] * local).reshape(len(points), -1)
            return torch.cat((root, contrasts), 1)
        const = self._constants(points)
        if self.family == "bspline":
            univariate = [_bspline_1d(points[:, axis], knots, n) for axis, (knots, n)
                          in enumerate(zip(const["knots"], self.lattice_shape))]
            return torch.einsum("ni,nj,nk->nijk", *univariate).reshape(len(points), -1)
        local = (points[:, None, :] - const["centers"]) / const["spacing"]
        weights = torch.softmax(-.5 * (local / self.gaussian_sigma_over_h).square().sum(-1), dim=1)
        if self.family == "gaussian_translation":
            # Drop one normalized weight: its constant redundancy is absorbed
            # by the shared affine root. All K translation functions remain in span.
            return torch.cat((_affine_features(points, self.domain_min, self.domain_max), weights[:, :-1]), 1)
        return (weights[:, :, None] * torch.cat((torch.ones_like(local[:, :, :1]), local), -1)).reshape(len(points), -1)

    def evaluate(self, points: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        if tuple(coefficients.shape) != (self.feature_count, 3):
            raise ValueError(f"Expected coefficients [{self.feature_count},3]")
        return self.features(points) @ coefficients

    def _native_descriptors(self):
        """Separate each scalar feature into three exact one-dimensional factors.

        Axis-gated tree paths repeat across branches. Deduplicating those paths
        exposes a small tensor-product core without truncating logistic tails.
        """
        one = ("one",)
        descriptors = []

        def add_affine(base, lo, hi):
            descriptors.append(tuple(base))
            for axis in range(3):
                factors = list(base)
                factors[axis] = ("multiply_linear", base[axis], (lo[axis] + hi[axis]) / 2,
                                 (hi[axis] - lo[axis]) / 2)
                descriptors.append(tuple(factors))

        if self.family == "global_affine":
            add_affine([one] * 3, self.domain_min, self.domain_max)
        elif self.is_tree:
            add_affine([one] * 3, self.domain_min, self.domain_max)
            lookup = {n["id"]: n for n in self.nodes}
            internals = sorted((n for n in self.nodes if "children" in n), key=lambda n: n["split_index"])
            for node in internals:
                path = []
                current = node
                while current["parent"] is not None:
                    parent = lookup[current["parent"]]
                    path.append((parent["axis"], parent["mid"], parent["tau"], current["branch"]))
                    current = parent
                path.reverse()
                path.append((node["axis"], node["mid"], node["tau"], 1))
                base = [("gate_product", tuple((mid, tau, branch) for a, mid, tau, branch in path if a == axis))
                        if any(a == axis for a, *_ in path) else one for axis in range(3)]
                if self.is_affine_tree:
                    add_affine(base, node["lo"], node["hi"])
                else:
                    descriptors.append(tuple(base))
        else:
            lattice = list(itertools.product(*(range(n) for n in self.lattice_shape)))
            if self.family == "gaussian_translation":
                add_affine([one] * 3, self.domain_min, self.domain_max)
                lattice = lattice[:-1]
            for index in lattice:
                base = [("spline" if self.family == "bspline" else "gaussian", i) for i in index]
                if self.family == "gaussian_affine":
                    descriptors.append(tuple(base))
                    for axis in range(3):
                        factor = list(base)
                        spacing = (self.domain_max[axis] - self.domain_min[axis]) / (self.lattice_shape[axis] - 1)
                        center = self.domain_min[axis] + index[axis] * spacing
                        factor[axis] = ("multiply_linear", base[axis], center, spacing)
                        descriptors.append(tuple(factor))
                else:
                    descriptors.append(tuple(base))
        if len(descriptors) != self.feature_count:
            raise RuntimeError("Native descriptor count differs from scalar feature count")
        return descriptors

    def _native_factors(self, shape, affine, like):
        """Return cached factor matrices/core indices, or None for generic grids."""
        shape = tuple(int(n) for n in shape)
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("Require a native grid with three dimensions>=2")
        h = torch.as_tensor(affine).detach().to(device="cpu", dtype=torch.float64)
        if h.shape != (4, 4) or not torch.isfinite(h).all():
            raise ValueError("Invalid native-grid affine")
        linear = h[:3, :3]
        if (linear - torch.diag(torch.diag(linear))).abs().max() > 1e-12:
            return None
        if torch.diag(linear).abs().min() < 1e-12:
            raise ValueError("Singular native-grid affine")
        affine_key = tuple(h.reshape(-1).tolist())
        key = ("native", shape, affine_key, str(like.device), like.dtype)
        if key in self._tensor_cache:
            return self._tensor_cache[key]
        descriptions = self._native_descriptors()
        unique = [list(dict.fromkeys(row[axis] for row in descriptions)) for axis in range(3)]
        counts = tuple(map(len, unique))
        # Pathologically unbalanced trees can have too many distinct axis paths.
        # The generic chunked evaluator remains correct in that case.
        if math.prod(counts) > 2_000_000:
            self._tensor_cache[key] = None
            return None
        factor_matrices = []
        axis_indices = []
        for axis in range(3):
            coords = (torch.arange(shape[axis], device=like.device, dtype=like.dtype)
                      * float(h[axis, axis]) + float(h[axis, 3]))
            cache = {}
            univariate = {}

            def evaluate_descriptor(descriptor):
                if descriptor in cache:
                    return cache[descriptor]
                kind = descriptor[0]
                if kind == "one":
                    value = torch.ones_like(coords)
                elif kind == "multiply_linear":
                    _, base, center, scale = descriptor
                    value = evaluate_descriptor(base) * ((coords - center) / scale)
                elif kind == "gate_product":
                    value = torch.ones_like(coords)
                    for middle, tau, branch in descriptor[1]:
                        gate = torch.sigmoid((coords - middle) / tau)
                        value = value * (gate if branch else 1 - gate)
                elif kind in ("gaussian", "spline"):
                    if kind not in univariate:
                        n = self.lattice_shape[axis]
                        lo, hi = self.domain_min[axis], self.domain_max[axis]
                        if kind == "spline":
                            univariate[kind] = _bspline_1d(coords, _knots(n, lo, hi, like), n)
                        else:
                            centers = torch.linspace(lo, hi, n, device=like.device, dtype=like.dtype)
                            width = (hi - lo) / (n - 1) * self.gaussian_sigma_over_h
                            univariate[kind] = torch.softmax(-.5 * ((coords[:, None] - centers) / width).square(), 1)
                    value = univariate[kind][:, descriptor[1]]
                else:
                    raise ValueError(f"Unsupported native descriptor {kind}")
                cache[descriptor] = value
                return value

            factor_matrices.append(torch.stack([evaluate_descriptor(d) for d in unique[axis]], 1))
            lookup = {descriptor: i for i, descriptor in enumerate(unique[axis])}
            axis_indices.append(torch.tensor([lookup[row[axis]] for row in descriptions],
                                             device=like.device, dtype=torch.long))
        flat_indices = (axis_indices[0] * counts[1] + axis_indices[1]) * counts[2] + axis_indices[2]
        result = dict(factors=factor_matrices, counts=counts, flat_indices=flat_indices)
        self._tensor_cache[key] = result
        return result

    def dense_forward(self, coefficients: torch.Tensor, shape, affine):
        """Fast exact-basis native evaluation for diagonal physical affines.

        Returns None on unsupported geometry/core size, requesting the caller's
        generic chunked fallback. No field-sized design matrix is constructed.
        """
        if tuple(coefficients.shape) != (self.feature_count, 3):
            raise ValueError("Native evaluation coefficient shape mismatch")
        native = self._native_factors(shape, affine, coefficients)
        if native is None:
            return None
        a, b, c = native["factors"]
        core = coefficients.new_zeros((math.prod(native["counts"]), 3))
        core = core.index_add(0, native["flat_indices"], coefficients).reshape(*native["counts"], 3)
        first = torch.einsum("da,abcj->dbcj", a, core)
        second = torch.einsum("hb,dbcj->dhcj", b, first)
        return torch.einsum("wc,dhcj->dhwj", c, second).contiguous()

    def dense_adjoint(self, gradient: torch.Tensor, shape, affine):
        """Adjoint of dense_forward for world-vector field gradients."""
        if tuple(gradient.shape) != (*tuple(shape), 3):
            raise ValueError("Native adjoint field gradient shape mismatch")
        native = self._native_factors(shape, affine, gradient)
        if native is None:
            return None
        a, b, c = native["factors"]
        last = torch.einsum("wc,dhwj->dhcj", c, gradient)
        middle = torch.einsum("hb,dhcj->dbcj", b, last)
        first = torch.einsum("da,dbcj->abcj", a, middle)
        return first.reshape(-1, 3)[native["flat_indices"]]

    def leaf_weights(self, points: torch.Tensor) -> torch.Tensor:
        if not self.is_tree:
            raise ValueError("Leaf weights require a tree")
        if self.leaf_count == 1:
            return points.new_ones((len(points), 1))
        path, q = self._tree_components(points)
        const = self._constants(points); idx = const["leaf_parents"]
        return torch.where(const["leaf_right"], q[:, idx], path[:, idx] - q[:, idx])

    def coefficients_for_affine(self, matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Coefficients reproducing u(x)=matrix@x+translation throughout box."""
        if tuple(matrix.shape) != (3, 3) or tuple(translation.shape) != (3,):
            raise ValueError("Affine matrix/translation shapes must be 3x3 and3")
        coeff = matrix.new_zeros((self.feature_count, 3))
        if self.family == "bspline":
            knots = [_knots(n, lo, hi, matrix) for n, lo, hi
                     in zip(self.lattice_shape, self.domain_min, self.domain_max)]
            greville = [torch.stack([t[i + 1:i + 4].mean() for i in range(n)])
                        for t, n in zip(knots, self.lattice_shape)]
            control_points = torch.stack(torch.meshgrid(*greville, indexing="ij"), -1).reshape(-1, 3)
            return control_points @ matrix.T + translation
        if self.family == "gaussian_affine":
            const = self._constants(matrix)
            block = coeff.view(-1, 4, 3)
            block[:, 0] = const["centers"] @ matrix.T + translation
            block[:, 1:] = const["spacing"][:, None] * matrix.T
            return coeff
        center = matrix.new_tensor([(lo + hi) / 2 for lo, hi in zip(self.domain_min, self.domain_max)])
        scale = matrix.new_tensor([(hi - lo) / 2 for lo, hi in zip(self.domain_min, self.domain_max)])
        coeff[0] = matrix @ center + translation
        coeff[1:4] = scale[:, None] * matrix.T
        return coeff

    def _split_ids(self, chosen: list[int]) -> "Basis":
        result = Basis.from_dict(self.to_dict())
        split_index = self.leaf_count - 1
        lookup = {node["id"]: node for node in result.nodes}
        for identifier in chosen:
            parent = lookup[identifier]
            lengths = [b - a for a, b in zip(parent["lo"], parent["hi"])]
            axis = max(range(3), key=lambda a: (lengths[a], -a))
            middle = (parent["lo"][axis] + parent["hi"][axis]) / 2
            child_ids = [len(result.nodes), len(result.nodes) + 1]
            parent.update(axis=axis, mid=middle, tau=self.tree_gate_width_fraction * lengths[axis],
                          children=child_ids, split_index=split_index)
            split_index += 1
            for branch, child_id in enumerate(child_ids):
                lo, hi = parent["lo"].copy(), parent["hi"].copy()
                if branch: lo[axis] = middle
                else: hi[axis] = middle
                child = dict(id=child_id, lo=lo, hi=hi, depth=parent["depth"] + 1,
                             parent=identifier, branch=branch)
                result.nodes.append(child); lookup[child_id] = child
        return result

    def split(self, coefficients: torch.Tensor, sample_points: torch.Tensor | None = None,
              residual: torch.Tensor | None = None, max_leaves: int | None = None,
              growth: float = .25, *, score_chunk_size: int = 4096):
        if not self.is_tree:
            raise ValueError("Only trees support split")
        if tuple(coefficients.shape) != (self.feature_count, 3):
            raise ValueError("Coefficient shape mismatch")
        if not 0 < growth <= 1:
            raise ValueError("Growth must be in(0,1]")
        limit = min(self.target_leaf_count, max_leaves if max_leaves is not None else self.target_leaf_count)
        leaves = [n for n in self.nodes if "children" not in n]
        count = min(max(1, int(math.floor(len(leaves) * growth))), limit - len(leaves))
        if count <= 0:
            return self, coefficients, {"split_count": 0, "reason": "budget_reached", "leaf_count": len(leaves)}
        if self.is_adaptive:
            if sample_points is None or residual is None or residual.shape != sample_points.shape:
                raise ValueError("Adaptive split needs paired [N,3] points/residual")
            with torch.no_grad():
                scores = torch.zeros(len(leaves), device=sample_points.device, dtype=torch.float64)
                for start in range(0, len(sample_points), score_chunk_size):
                    weights = self.leaf_weights(sample_points[start:start + score_chunk_size])
                    power = residual[start:start + score_chunk_size].square().sum(-1)
                    scores += (weights * power[:, None]).sum(0).double()
                values = scores.cpu().tolist()
        else:
            values = [math.prod(b - a for a, b in zip(n["lo"], n["hi"])) for n in leaves]
        eligible = []
        for node, score in zip(leaves, values):
            lengths = [b - a for a, b in zip(node["lo"], node["hi"])]
            if node["depth"] < self.max_depth and max(lengths) / 2 >= self.min_box_size_mm:
                eligible.append((node["id"], score))
        chosen = [identifier for identifier, _ in sorted(eligible, key=lambda item: (-item[1], item[0]))[:count]]
        if not chosen:
            return self, coefficients, {"split_count": 0, "reason": "geometry_limit", "leaf_count": len(leaves)}
        updated = self._split_ids(chosen)
        added = updated.feature_count - self.feature_count
        new_coefficients = torch.cat((coefficients, coefficients.new_zeros((added, 3))), 0)
        return updated, new_coefficients, {"split_count": len(chosen), "split_leaf_ids": chosen,
            "old_feature_count": self.feature_count, "new_feature_count": updated.feature_count,
            "old_leaf_count": self.leaf_count, "leaf_count": updated.leaf_count,
            "score": "endpoint_residual_weighted" if self.is_adaptive else "box_volume",
            "field_preserving_by_zero_contrast": True}


def create_basis(family: str, budget_cap: int, domain_min, domain_max, *,
                 start_root: bool = False, gaussian_sigma_over_h: float = .85,
                 tree_gate_width_fraction: float = .12, max_depth: int = 24,
                 min_box_size_mm: float = 1.0) -> Basis:
    if type(budget_cap) is not int or budget_cap < 12:
        raise ValueError("Integer motion budget >=12 required")
    lo, hi = [float(x) for x in domain_min], [float(x) for x in domain_max]
    result = Basis(family, budget_cap, lo, hi, gaussian_sigma_over_h=gaussian_sigma_over_h,
                   tree_gate_width_fraction=tree_gate_width_fraction, max_depth=max_depth,
                   min_box_size_mm=min_box_size_mm)
    lengths = [b - a for a, b in zip(lo, hi)]
    if family == "bspline":
        result.lattice_shape = _lattice_shape(budget_cap // 3, lengths, 4)
    elif family.startswith("gaussian_"):
        cap = budget_cap // 12 if family == "gaussian_affine" else (budget_cap - 9) // 3
        result.lattice_shape = _lattice_shape(cap, lengths, 2)
    elif result.is_tree:
        result.nodes = [dict(id=0, lo=lo, hi=hi, depth=0, parent=None, branch=None)]
        if not start_root:
            if result.is_adaptive:
                raise ValueError("Adaptive trees require start_root=True and residual-driven splits")
            coeff = torch.zeros((4, 3))
            while result.leaf_count < result.target_leaf_count:
                result, coeff, event = result.split(coeff)
                if not event["split_count"]:
                    break
    if result.actual_motion_params > budget_cap:
        raise RuntimeError("Basis construction exceeded budget")
    return result
