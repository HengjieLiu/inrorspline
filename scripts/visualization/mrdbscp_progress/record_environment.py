"""Host-side stdlib-only Docker inspection and replay-command recording.

No preprocessing, CUDA initialization, container changes, or job launches.
Private deployment details are written only to the external review directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def inspect(*args):
    return json.loads(subprocess.check_output(["docker", *args], text=True))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-container", required=True)
    args = parser.parse_args()
    root, artifact = args.output.resolve(), args.artifact_root.resolve()
    repo = Path(__file__).resolve().parents[3]
    if root == artifact or not root.is_relative_to(artifact) or root.is_relative_to(repo):
        raise ValueError("An existing external artifact run is required")
    manifest = json.loads((root / "manifest.json").read_text())
    container = inspect("inspect", args.container)
    image = inspect("image", "inspect", container["Image"])
    env = dict(item.split("=", 1) for item in container["Config"]["Env"])
    container_artifact = env.get("ARTIFACT_ROOT", "")
    if not container_artifact or not Path(container_artifact).is_absolute():
        raise ValueError("Container must declare an absolute ARTIFACT_ROOT")
    artifact_binds = [m for m in container["Mounts"]
                      if m["Destination"] == container_artifact]
    if len(artifact_binds) != 1 or not artifact_binds[0]["RW"] or Path(artifact_binds[0]["Source"]).resolve() != artifact:
        raise ValueError("Container artifact bind does not match the output store")
    invocation = ["docker", "exec", "--user", "1009:1009", "-e", "CUDA_VISIBLE_DEVICES=",
                  "-e", "MPLCONFIGDIR=/tmp/mrdbscp_progress_mpl_1009", "-w", args.repo_container,
                  args.container, "python", "-B"]
    render_command = invocation + ["-m", "scripts.visualization.mrdbscp_progress.preview", *manifest["argv"][1:]]
    commands = dict(
        render=render_command,
        tests=invocation + ["-m", "unittest", "scripts.visualization.mrdbscp_progress.test_progress", "-v"],
        verify=invocation + ["-m", "scripts.visualization.mrdbscp_progress.verify", "--output",
                            str(Path(container_artifact) / root.relative_to(artifact))])
    payload = dict(container_id=container["Id"], image_tag=container["Config"]["Image"],
        image_id=image["Id"], image_repo_digests=image.get("RepoDigests", []),
        image_created=image.get("Created"), mounts=container["Mounts"],
        environment={key: env.get(key) for key in ("ARTIFACT_ROOT", "NVIDIA_VISIBLE_DEVICES")},
        device_requests=container["HostConfig"].get("DeviceRequests"),
        artifact_uid=1009, artifact_gid=1009, gpu_used=False, commands=commands,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / "docker_provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "environment_recorder_source.py").write_text(Path(__file__).read_text())
    (root / "replay_commands.txt").write_text(
        "# Recorded commands. Rendering refuses an existing output; choose a NEW output before replay.\n\n"
        + "\n\n".join(f"# {label}\n{shlex.join(command)}" for label, command in commands.items()) + "\n")
    print(json.dumps(dict(output=str(root), image_id=image["Id"], gpu_used=False)))


if __name__ == "__main__":
    main()
