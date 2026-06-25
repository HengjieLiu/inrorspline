# Docker

Docker definitions will live here once the runtime is selected.

Current policy:

- mount repo, data, and scratch as separate volumes
- mount data read-only by default
- write runs/checkpoints/logs to `/workspace/scratch`
- do not commit Docker image archives

See `docs/workflows/docker_gpu_workflow.md`.
