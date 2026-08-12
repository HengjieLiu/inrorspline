from __future__ import annotations

from scripts.experiments.lungct_dual_inr_controlled import launch_lungmask_revision


def test_revision_job_matrices_are_explicit_and_unique() -> None:
    variants = ("PL1D", "PL1B", "PL1A", "PL1C")
    cases = (1, 8)

    pilot = launch_lungmask_revision.jobs_for(
        "pilot", variants=variants, cases=cases
    )
    smoke = launch_lungmask_revision.jobs_for(
        "smoke", variants=variants, cases=cases
    )

    assert len(pilot) == 8
    assert len({job.key for job in pilot}) == 8
    assert [(job.variant, job.case_id) for job in pilot] == [
        ("PL1D", 1),
        ("PL1B", 1),
        ("PL1A", 1),
        ("PL1C", 1),
        ("PL1D", 8),
        ("PL1B", 8),
        ("PL1A", 8),
        ("PL1C", 8),
    ]
    assert all(job.run_mode == "pilot" for job in pilot)

    assert len(smoke) == 4
    assert len({job.key for job in smoke}) == 4
    assert [(job.variant, job.case_id) for job in smoke] == [
        ("PL1D", 1),
        ("PL1B", 1),
        ("PL1A", 1),
        ("PL1C", 1),
    ]
    assert all(job.run_mode == "smoke" for job in smoke)
