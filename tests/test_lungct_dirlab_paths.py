from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lungct = load_module("test_lungct_common", "scripts/data/common/lungct.py")
experiment_common = load_module(
    "test_lungct_experiment_common",
    "scripts/experiments/lungct_no_sweep_repro/common.py",
)
def touch_4dct_case(root: Path, marker: str) -> None:
    image_dir = root / marker / "Case1Pack" / "Images"
    landmark_dir = root / marker / "Case1Pack" / "ExtremePhases"
    image_dir.mkdir(parents=True)
    landmark_dir.mkdir(parents=True)
    for phase in ("00", "50"):
        (image_dir / f"case1_T{phase}_s.img").touch()
        (landmark_dir / f"Case1_300_T{phase}_xyz.txt").touch()


def touch_copd_case(root: Path, marker: str) -> None:
    case_dir = root / marker / "copd1" / "copd1"
    case_dir.mkdir(parents=True)
    for phase in ("iBH", "eBH"):
        (case_dir / f"copd1_{phase}CT.img").touch()
        (case_dir / f"copd1_300_{phase}_xyz_r1.txt").touch()


def test_raw_resolvers_use_only_dirlab_raw_layout(tmp_path: Path) -> None:
    touch_4dct_case(tmp_path, "raw/4DCT")
    touch_4dct_case(tmp_path, "4DCT")
    touch_copd_case(tmp_path, "raw/COPD")
    touch_copd_case(tmp_path, "COPD")

    fourdct = lungct.resolve_4dct_case(1, tmp_path)
    copd = lungct.resolve_copd_case(1, tmp_path)

    assert fourdct.status == "complete"
    assert Path(fourdct.root).is_relative_to(tmp_path / "raw" / "4DCT")
    assert copd.status == "complete"
    assert Path(copd.root).is_relative_to(tmp_path / "raw" / "COPD")


def test_processed_dataset_names_use_canonical_capitalization(tmp_path: Path) -> None:
    assert lungct.mask_root_case_dir(tmp_path, 1, "4dct") == tmp_path / "4DCT" / "case01"
    assert lungct.mask_root_case_dir(tmp_path, 2, "copd") == tmp_path / "COPD" / "case02"
    assert experiment_common.case_dir(tmp_path, 1, "4dct") == tmp_path / "4DCT" / "case01"
    assert experiment_common.case_dir(tmp_path, 2, "copd") == tmp_path / "COPD" / "case02"


def test_existing_legacy_mask_tree_remains_readable_for_audits(tmp_path: Path) -> None:
    (tmp_path / "4dct").mkdir()

    assert lungct.mask_root_case_dir(tmp_path, 1, "4dct") == tmp_path / "4dct" / "case01"


def test_same_root_promotion_is_a_safe_noop(tmp_path: Path) -> None:
    saved_common = sys.modules.pop("common", None)
    saved_common_volume = sys.modules.pop("common.volume", None)
    try:
        promote_masks = load_module(
            "test_promote_4dct_lung_masks",
            "scripts/data/preprocessing/lungct/promote_4dct_lung_masks.py",
        )
    finally:
        sys.modules.pop("common.volume", None)
        sys.modules.pop("common", None)
        if saved_common is not None:
            sys.modules["common"] = saved_common
        if saved_common_volume is not None:
            sys.modules["common.volume"] = saved_common_volume

    source = tmp_path / "4DCT"
    source.mkdir()
    marker = source / "marker.txt"
    marker.write_text("keep\n")

    result = promote_masks.promote_tree(tmp_path, tmp_path, "20260802_000000")

    assert result["status"] == "already_current"
    assert result["backup"] == ""
    assert marker.read_text() == "keep\n"
    assert not (tmp_path / "4DCT_backup_20260802_000000").exists()


def test_canonical_environment_override_precedes_deprecated_override() -> None:
    env = {
        "INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT": "/canonical",
        "INRORSPLINE_LUNGCT_PTVPREP_ROOT": "/deprecated",
    }
    assert experiment_common.resolve_default_ptvprep_root(env, docker_available=False) == Path("/canonical")


def test_deprecated_environment_override_remains_supported() -> None:
    env = {"INRORSPLINE_LUNGCT_PTVPREP_ROOT": "/deprecated"}
    assert experiment_common.resolve_default_ptvprep_root(env, docker_available=False) == Path("/deprecated")


def test_default_root_selects_host_or_docker_layout() -> None:
    assert experiment_common.resolve_default_ptvprep_root({}, docker_available=False) == (
        experiment_common.HOST_PTVPREP_ROOT
    )
    assert experiment_common.resolve_default_ptvprep_root({}, docker_available=True) == Path(
        "/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"
    )
