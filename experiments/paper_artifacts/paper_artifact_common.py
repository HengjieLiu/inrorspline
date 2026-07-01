"""Shared helpers for final-paper artifact standards."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PAPER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PAPER_ROOT.parents[1]
DEFAULT_STANDARD_PATH = PAPER_ROOT / "artifact_standard.json"
DEFAULT_NICKNAME_PATH = PAPER_ROOT / "method_nicknames.csv"

LR_TAG_RE = re.compile(r"lr(?P<tag>0p0+1|0p01|0p001|0p0001|1e-2|1e-3|1e-4)")
PUBLIC_PATH_REPLACEMENTS = [
    (re.compile(r"/database/inrorspline"), "${DATA_ROOT}"),
    (re.compile(r"/data/[^/]+/inrorspline"), "${DATA_ROOT}"),
    (re.compile(r"/homebase/code_sync/inrorspline"), "${REPO_ROOT}"),
    (re.compile(r"/home/[^/]+/code_sync/inrorspline"), "${REPO_ROOT}"),
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pattern, new in PUBLIC_PATH_REPLACEMENTS:
            out = pattern.sub(new, out)
        return out
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_public_value(item) for item in value)
    if isinstance(value, dict):
        return {key: sanitize_public_value(item) for key, item in value.items()}
    return value


def sanitize_public_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_public_value(dict(record)) for record in records]


def load_standard(path: str | Path | None = None) -> dict[str, Any]:
    standard_path = repo_path(path) if path is not None else DEFAULT_STANDARD_PATH
    return json.loads(standard_path.read_text())


def load_nicknames(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    nickname_path = repo_path(path) if path is not None else DEFAULT_NICKNAME_PATH
    with nickname_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(row["method_id"]): {key: str(value) for key, value in row.items()} for row in rows}


def display_name(
    method_id: str,
    original_name: str | None = None,
    nicknames: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    if nicknames is None:
        nicknames = load_nicknames()
    method_id = str(method_id)
    if method_id in nicknames:
        return str(nicknames[method_id].get("nickname") or nicknames[method_id].get("original_name") or method_id)
    if original_name is not None:
        matches = [
            row
            for row in nicknames.values()
            if str(row.get("original_name", "")) == str(original_name)
        ]
        if len(matches) == 1:
            return str(matches[0].get("nickname") or original_name)
    return str(original_name if original_name is not None else method_id)


def method_family(method_id: str, standard: Mapping[str, Any] | None = None) -> str:
    if standard is None:
        standard = load_standard()
    if str(method_id) in set(str(item) for item in standard.get("reference_method_ids", [])):
        return "reference"
    return str(standard.get("method_family_by_id", {}).get(str(method_id), "unknown"))


def canonical_lr(method_id: str, standard: Mapping[str, Any] | None = None) -> float | None:
    if standard is None:
        standard = load_standard()
    family = method_family(method_id, standard)
    value = standard.get("canonical_lr_by_family", {}).get(family)
    if value is None:
        return None
    return float(value)


def artifact_output_dir(artifact: str, standard: Mapping[str, Any] | None = None) -> Path:
    if standard is None:
        standard = load_standard()
    spec = standard["artifact_versions"][artifact]
    return repo_path(str(spec["output_dir"]))


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def finite_float(value: Any) -> float | None:
    number = safe_float(value)
    return number if math.isfinite(number) else None


def format_lr(value: Any) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    known = {
        1e-2: "1e-2",
        1e-3: "1e-3",
        1e-4: "1e-4",
    }
    for lr, label in known.items():
        if lr_close(number, lr):
            return label
    return f"{number:g}"


def lr_close(a: Any, b: Any, *, rtol: float = 1.0e-6, atol: float = 1.0e-12) -> bool:
    af = safe_float(a)
    bf = safe_float(b)
    return math.isfinite(af) and math.isfinite(bf) and math.isclose(af, bf, rel_tol=rtol, abs_tol=atol)


def parse_lr_tag(text: str) -> float | None:
    match = LR_TAG_RE.search(str(text))
    if not match:
        return None
    tag = match.group("tag")
    mapping = {
        "0p01": 1e-2,
        "0p001": 1e-3,
        "0p0001": 1e-4,
        "1e-2": 1e-2,
        "1e-3": 1e-3,
        "1e-4": 1e-4,
    }
    return mapping.get(tag)


def parse_effective_lr(record: Mapping[str, Any]) -> float | None:
    lr = finite_float(record.get("lr"))
    if lr is not None:
        return lr
    stage_lr = str(record.get("stage_lr", "")).strip()
    if stage_lr:
        values = [finite_float(part) for part in stage_lr.split(",")]
        values = [value for value in values if value is not None]
        if values and all(lr_close(value, values[0]) for value in values):
            return values[0]
    for key in ["summary_path", "source", "source_path", "run_dir"]:
        parsed = parse_lr_tag(str(record.get(key, "")))
        if parsed is not None:
            return parsed
    return None


def _exception_for(
    artifact: str,
    dataset: str,
    method_id: str,
    observed_lr: float | None,
    standard: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for item in standard.get("temporary_exceptions", []):
        if str(item.get("artifact")) != str(artifact):
            continue
        if str(item.get("dataset")) != str(dataset):
            continue
        if str(item.get("method_id")) != str(method_id):
            continue
        if observed_lr is None or lr_close(observed_lr, item.get("current_value")):
            return item
    return None


def selected_hypervariant_records(artifact: str, standard: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if standard is None:
        standard = load_standard()
    return [dict(row) for row in standard.get("artifact_selected_hypervariants", {}).get(artifact, [])]


def validate_lr_records(
    records: Sequence[Mapping[str, Any]],
    *,
    artifact: str,
    standard: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if standard is None:
        standard = load_standard()
    rows: list[dict[str, Any]] = []
    for record in records:
        dataset = str(record.get("dataset", ""))
        method_id = str(record.get("method_id", record.get("method", "")))
        family = method_family(method_id, standard)
        expected = canonical_lr(method_id, standard)
        observed = finite_float(record.get("lr"))
        if observed is None:
            observed = parse_effective_lr(record)
        exception = _exception_for(artifact, dataset, method_id, observed, standard)
        if family == "reference":
            status = "reference"
        elif expected is None:
            status = "unknown_family"
        elif observed is None:
            status = "missing_lr"
        elif lr_close(observed, expected):
            status = "matches_standard"
        elif exception is not None:
            status = "temporary_exception"
        else:
            status = "mismatch"
        rows.append(
            {
                "artifact": artifact,
                "dataset": dataset,
                "method_id": method_id,
                "method_family": family,
                "observed_lr": observed,
                "observed_lr_label": format_lr(observed),
                "canonical_lr": expected,
                "canonical_lr_label": format_lr(expected),
                "status": status,
                "selected_setting": record.get("selected_setting", record.get("variant", "")),
                "source": record.get("source", record.get("source_path", "")),
                "note": str(record.get("note", "")) or str(exception.get("note", "") if exception else ""),
            }
        )
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def validation_report_markdown(title: str, validation_rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "dataset",
        "method_id",
        "method_family",
        "observed_lr_label",
        "canonical_lr_label",
        "status",
        "selected_setting",
        "note",
    ]
    blocking = [row for row in validation_rows if row.get("status") in {"missing_lr", "mismatch", "unknown_family"}]
    lines = [
        f"# {title}",
        "",
        f"- Rows checked: `{len(validation_rows)}`",
        f"- Blocking issues: `{len(blocking)}`",
        "",
        markdown_table(headers, validation_rows),
        "",
    ]
    return "\n".join(lines)
