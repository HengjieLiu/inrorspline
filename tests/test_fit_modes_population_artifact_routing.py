from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
POPULATION_DIR = EXPERIMENTS_DIR / "fit_modes_population"
for directory in (str(POPULATION_DIR), str(EXPERIMENTS_DIR), str(REPO_ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)


def load_module(name: str, filename: str):
    path = POPULATION_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLOT = load_module("inrorspline_population_plot_routing", "plot_case_diagnostics.py")
OASIS = load_module("inrorspline_population_oasis_routing", "launch_oasis_fixedmask_panelA_curves.py")
DIRLAB = load_module("inrorspline_population_dirlab_routing", "launch_dirlab_panelA_schedule_refresh.py")
DIRLAB_AGG = load_module(
    "inrorspline_population_dirlab_aggregate_routing",
    "aggregate_dirlab_panelA_schedule_refresh.py",
)
MASKED = load_module(
    "inrorspline_population_masked_routing",
    "render_masked_panelA_both_datasets_diagnostics.py",
)
REGULARITY = load_module(
    "inrorspline_population_regularity_routing",
    "render_dirlab_regularity_matched_panelA_diagnostics.py",
)
SCHEDULE = load_module(
    "inrorspline_population_schedule_routing",
    "render_schedule_matched_diagnostics.py",
)
POPULATION = load_module("inrorspline_population_launch_routing", "launch_population.py")
POPULATION_AGG = load_module(
    "inrorspline_population_aggregate_routing",
    "aggregate_population.py",
)


def configured_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    store = tmp_path / "artifact-store"
    review = store / "repo_experiments"
    review.mkdir(parents=True)
    monkeypatch.setenv("ARTIFACT_ROOT", str(store))
    return store, review


def external_input_file(review: Path, name: str) -> Path:
    path = review / "migration_inputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("header\n")
    return path


def external_input_dir(review: Path, name: str) -> Path:
    path = review / "migration_inputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_population_producers_default_to_external_run_and_review_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, review = configured_store(tmp_path, monkeypatch)
    historical_index = external_input_file(review, "historical_index.csv")
    v002_root = external_input_dir(review, "v002")
    v003_root = external_input_dir(review, "v003")
    alpha_rows = external_input_file(review, "alpha_case_rows.csv")
    base_index = external_input_file(review, "base_index.csv")
    replacement_index = external_input_file(review, "replacement_index.csv")
    figure_root = review / POPULATION.FIGURE_EXPERIMENT_KEY
    figure_root.mkdir(parents=True)
    selection_csv = figure_root / "fit_modes_standardized_panelAB_selected_runs.csv"
    selection_csv.write_text("panel,plot_case,method_id,lr,available,source,note\n")

    oasis = OASIS.resolve_artifact_paths(OASIS.parse_args([]))
    assert oasis.run_root == store / "runs" / OASIS.RUN_KEY
    assert oasis.output_dir == review / OASIS.EXPERIMENT_KEY

    dirlab = DIRLAB.resolve_artifact_paths(DIRLAB.parse_args([]))
    assert dirlab.review_root == review / DIRLAB.EXPERIMENT_KEY
    assert dirlab.run_root == store / "runs" / DIRLAB.EXPERIMENT_ID
    assert dirlab.smoke_run_root == store / "runs" / f"{DIRLAB.EXPERIMENT_ID}_smoke"

    dirlab_agg = DIRLAB_AGG.resolve_artifact_paths(
        DIRLAB_AGG.parse_args(["--historical-index", str(historical_index)])
    )
    assert dirlab_agg.review_root == review / DIRLAB_AGG.EXPERIMENT_KEY
    assert dirlab_agg.run_root == store / "runs" / DIRLAB_AGG.RUN_KEY
    assert dirlab_agg.historical_index == historical_index

    masked = MASKED.resolve_artifact_paths(MASKED.parse_args(["--v002-root", str(v002_root)]))
    assert masked.output_root == review / MASKED.EXPERIMENT_KEY
    assert masked.v002_root == v002_root

    regularity = REGULARITY.resolve_artifact_paths(
        REGULARITY.parse_args(
            ["--v003-root", str(v003_root), "--alpha-case-rows", str(alpha_rows)]
        )
    )
    assert regularity.output_root == review / REGULARITY.EXPERIMENT_KEY
    assert regularity.alpha_case_rows == alpha_rows

    schedule = SCHEDULE.resolve_artifact_paths(
        SCHEDULE.parse_args(
            ["--base-index", str(base_index), "--replacement-index", str(replacement_index)]
        )
    )
    assert schedule.output_root == review / SCHEDULE.EXPERIMENT_KEY
    assert schedule.base_index == base_index
    assert schedule.replacement_index == replacement_index

    population = POPULATION.resolve_artifact_paths(POPULATION.parse_args([]))
    assert population.out_root == store / "runs" / POPULATION.RUN_KEY
    assert population.review_root == review / POPULATION.EXPERIMENT_KEY
    assert population.figure_root == figure_root
    assert population.reuse_csv == selection_csv

    population_agg = POPULATION_AGG.resolve_artifact_paths(POPULATION_AGG.parse_args([]))
    assert population_agg.run_root == store / "runs" / POPULATION_AGG.RUN_KEY
    assert population_agg.review_root == review / POPULATION_AGG.EXPERIMENT_KEY
    assert population_agg.figure_root == figure_root

    plot = PLOT.resolve_artifact_paths(PLOT.parse_args([]))
    assert plot.run_root == store / "runs" / PLOT.RUN_KEY
    assert plot.review_root == review / PLOT.EXPERIMENT_KEY
    assert plot.output_dir == plot.review_root / "diagnostic_figures"
    assert plot.comparison_md == plot.review_root / "diagnostic_comparison.md"


def test_population_producers_reject_repo_local_generated_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, review = configured_store(tmp_path, monkeypatch)
    v002_root = external_input_dir(review, "v002")

    with pytest.raises(ValueError, match="external artifact root"):
        OASIS.resolve_artifact_paths(
            OASIS.parse_args(["--run-root", str(REPO_ROOT / "experiments" / "local-runs")])
        )
    with pytest.raises(ValueError, match="external artifact root"):
        MASKED.resolve_artifact_paths(
            MASKED.parse_args(
                [
                    "--v002-root",
                    str(v002_root),
                    "--output-root",
                    str(REPO_ROOT / "experiments" / "local-output"),
                ]
            )
        )
    with pytest.raises(ValueError, match="external artifact root"):
        PLOT.resolve_artifact_paths(
            PLOT.parse_args(["--output-dir", str(REPO_ROOT / "experiments" / "local-output")])
        )


@pytest.mark.parametrize(
    "module,arguments",
    [
        (DIRLAB_AGG, []),
        (MASKED, []),
        (REGULARITY, []),
        (SCHEDULE, []),
    ],
)
def test_migrated_generated_inputs_are_explicit(module, arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        module.parse_args(arguments)


def test_all_current_population_producers_invoke_the_shared_resolver() -> None:
    filenames = [
        "launch_oasis_fixedmask_panelA_curves.py",
        "launch_dirlab_panelA_schedule_refresh.py",
        "aggregate_dirlab_panelA_schedule_refresh.py",
        "render_masked_panelA_both_datasets_diagnostics.py",
        "render_dirlab_regularity_matched_panelA_diagnostics.py",
        "render_schedule_matched_diagnostics.py",
        "launch_population.py",
        "aggregate_population.py",
        "plot_case_diagnostics.py",
    ]
    for filename in filenames:
        source = (POPULATION_DIR / filename).read_text()
        assert "import artifact_paths" in source
        assert "def resolve_artifact_paths(" in source
        assert "resolve_artifact_paths(parse_args())" in source
        assert 'REPO_ROOT / "experiments' not in source
