#!/usr/bin/env python3
"""Run the released DUAL A0/A1 protocol on prepared COPD author-format data."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.lungct_dual_inr_author import run_author as core


def _tag_copd_output(output_root: Path, variant: str, case: int) -> None:
    case_dir = output_root / variant.lower() / f"case_{case:02d}"
    config_path = case_dir / "resolved_config.json"
    summary_path = case_dir / "summary.json"
    provenance_path = case_dir / "provenance.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("export", {}).update(
        {
            "dataset": "dirlab_copd",
            "fixed_phase_name": "iBH",
            "moving_phase_name": "eBH",
            "registration_direction": "eBH_to_iBH",
            "protocol_handle": "dual_author_native_lungbbox_totalseg",
        }
    )
    config["copd_binding"] = {
        "dataset": "copd",
        "fixed_phase": "iBH / author phase_00",
        "moving_phase": "eBH / author phase_05",
        "registration_direction": "eBH_to_iBH",
        "metric_mask": "fixed iBH R231 is posthoc-only; five-lobe masks drive author crop/sampling",
    }
    core.write_json(config_path, config)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["dataset"] = "dirlab_copd"
    provenance["fixed_phase_name"] = "iBH"
    provenance["moving_phase_name"] = "eBH"
    provenance["registration_direction"] = "eBH_to_iBH"
    provenance["wrapper"] = Path(__file__).relative_to(core.repository_root()).as_posix()
    core.write_json(provenance_path, provenance)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["dataset"] = "copd"
    summary["fixed_phase_name"] = "iBH"
    summary["moving_phase_name"] = "eBH"
    summary["registration_direction"] = "eBH_to_iBH"
    summary["protocol_tags"] = {
        "dataset": "copd",
        "image_domain": "lung_union_bbox",
        "spacing": "native",
        "crop_rule": "fixed_moving_lobe_union_padding2",
        "intensity_rule": "per_image_minmax",
        "mask_rule": (
            "fixed_lobe_union_sampling_plus_paired_lobe_loss"
            if summary["variant"] == "A0"
            else "fixed_lobe_union_sampling"
        ),
        "loss_domain": "mask_sampled",
        "metric_domain": "native_landmarks_after_author_crop",
        "metric_resolution": "native_full_or_crop",
        "tre_mode": "both",
        "method_family": "dual_inr",
    }
    core.write_json(summary_path, summary)


def run_case(**kwargs):
    summary = core.run_case(**kwargs)
    _tag_copd_output(kwargs["output_root"], kwargs["variant"], kwargs["case"])
    return summary


def main(argv: list[str] | None = None) -> int:
    args = core.build_parser().parse_args(argv)
    cases = [args.case] if args.case is not None else args.cohort
    if args.variant == "A1" and args.replay_a0_transitions is None:
        raise SystemExit("A1 requires --replay-a0-transitions")
    if args.variant == "A0" and args.replay_a0_transitions is not None:
        raise SystemExit("A0 is adaptive and cannot use --replay-a0-transitions")
    data_root, output_root = core.validate_roots(args.data_root, args.output_root)
    execution_root = output_root / f"smoke_{args.smoke_steps}steps" if args.smoke_steps is not None else output_root
    summaries = []
    for case in cases:
        try:
            summaries.append(
                run_case(
                    variant=args.variant,
                    case=case,
                    data_root=data_root,
                    output_root=execution_root,
                    device=args.device,
                    seed=args.seed,
                    trace_input=args.replay_a0_transitions,
                    single_case_selection=args.case is not None,
                    smoke_steps=args.smoke_steps,
                )
            )
        except Exception as exc:
            core._mark_partial_case_failed(
                output_root=execution_root,
                variant=args.variant,
                case=case,
                seed=args.seed,
                error=exc,
            )
            raise
    if args.cohort is not None:
        core.write_json(
            execution_root / args.variant.lower() / "cohort_summary.json",
            {
                "schema_version": 1,
                "dataset": "copd",
                "variant": args.variant,
                "seed": args.seed,
                "cases": cases,
                "n_complete": len(summaries),
                "case_summaries": summaries,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
