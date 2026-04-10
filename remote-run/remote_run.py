#!/usr/bin/env python3
"""
Run a workspace-synced command on the remote side of a Mutagen session.

Typical flow:
1. Resolve the current workspace and remote mapping.
2. Wait for the mirrored remote workspace to observe local changes.
3. SSH to the remote host and run the mirrored Python file.
4. Stream stdout/stderr live while also saving them locally.
5. Wait for the remote completion sentinel to sync back locally.
"""

import argparse
import configparser
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = "remote-run/config.ini"
DEFAULT_ARTIFACT_DIR = "remote-run/artifacts"
DEFAULT_MUTAGEN_TEMPLATE = "{{json .}}"
DEFAULT_SYNC_TIMEOUT_SECONDS = 20.0
DEFAULT_SYNC_POLL_INTERVAL_SECONDS = 0.25


class RemoteRunError(RuntimeError):
    """Raised when remote execution cannot be prepared or completed."""


@dataclass
class RuntimeConfig:
    config_path: Path
    local_workspace_root: Path
    mutagen_session: str
    ssh_target: str
    remote_workspace_root: PurePosixPath
    bootstrap: str
    remote_python: str
    remote_shell: str
    artifact_dir: Path
    ssh_options: List[str]


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def maybe_relative_to(path: Path, parent: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(parent.resolve())
    except ValueError:
        return None


def find_repo_root(start_dir: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(start_dir),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return start_dir.resolve()

    repo_root = result.stdout.strip()
    if not repo_root:
        return start_dir.resolve()
    return Path(repo_root).resolve()


def load_config(config_path: Path) -> Dict[str, str]:
    parser = configparser.ConfigParser()
    if not config_path.exists():
        return {}

    parser.read(config_path)
    if not parser.has_section("remote"):
        return {}
    return {key: value.strip() for key, value in parser["remote"].items()}


def load_mutagen_sessions() -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            ["mutagen", "sync", "list", "--template", DEFAULT_MUTAGEN_TEMPLATE],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RemoteRunError("`mutagen` is not installed or not available in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RemoteRunError(
            f"Failed to list Mutagen sessions: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc

    output = result.stdout.strip() or "[]"
    try:
        sessions = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RemoteRunError(f"Mutagen returned invalid JSON: {output}") from exc

    if not isinstance(sessions, list):
        raise RemoteRunError("Unexpected Mutagen session payload.")
    return sessions


def select_mutagen_session(
    sessions: List[Dict[str, Any]],
    local_workspace_root: Path,
    requested_session: Optional[str],
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for session in sessions:
        name = session.get("name")
        identifier = session.get("identifier")
        if requested_session and requested_session not in (name, identifier):
            continue
        beta = session.get("beta", {})
        if beta.get("protocol") != "ssh":
            continue
        alpha = session.get("alpha", {})
        alpha_path = alpha.get("path")
        if not alpha_path:
            continue
        if Path(alpha_path).resolve() != local_workspace_root.resolve():
            continue
        candidates.append(session)

    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(
            session.get("name") or session.get("identifier", "<unknown>")
            for session in candidates
        )
        raise RemoteRunError(
            f"Found multiple Mutagen sessions for {local_workspace_root}: {names}. "
            "Pass --mutagen-session explicitly."
        )
    return candidates[0]


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    repo_root = find_repo_root(Path.cwd())
    config_path = resolve_path(args.config, repo_root)
    file_config = load_config(config_path)

    env_overrides = {
        "mutagen_session": os.environ.get("REMOTE_RUN_MUTAGEN_SESSION", "").strip(),
        "ssh_target": os.environ.get("REMOTE_RUN_SSH_TARGET", "").strip(),
        "remote_workspace_root": os.environ.get("REMOTE_RUN_REMOTE_WORKSPACE_ROOT", "").strip(),
        "bootstrap": os.environ.get("REMOTE_RUN_BOOTSTRAP", ""),
        "python": os.environ.get("REMOTE_RUN_PYTHON", "").strip(),
        "remote_shell": os.environ.get("REMOTE_RUN_REMOTE_SHELL", "").strip(),
        "artifact_dir": os.environ.get("REMOTE_RUN_ARTIFACT_DIR", "").strip(),
        "ssh_options": os.environ.get("REMOTE_RUN_SSH_OPTIONS", "").strip(),
        "local_workspace_root": os.environ.get("REMOTE_RUN_LOCAL_WORKSPACE_ROOT", "").strip(),
    }

    def pick(name: str, cli_value: Optional[str] = None) -> str:
        if cli_value:
            return cli_value
        if env_overrides.get(name):
            return env_overrides[name]
        return file_config.get(name, "")

    local_workspace_raw = pick("local_workspace_root", args.local_workspace_root)
    local_workspace_root = (
        resolve_path(local_workspace_raw, repo_root) if local_workspace_raw else repo_root
    )

    requested_session = pick("mutagen_session", args.mutagen_session)
    ssh_target = pick("ssh_target", args.ssh_target)
    remote_workspace_root_raw = pick(
        "remote_workspace_root", args.remote_workspace_root
    )
    mutagen_session = requested_session
    discovered_session = None

    if not (mutagen_session and ssh_target and remote_workspace_root_raw):
        sessions = load_mutagen_sessions()
        discovered_session = select_mutagen_session(
            sessions=sessions,
            local_workspace_root=local_workspace_root,
            requested_session=requested_session or None,
        )

    if discovered_session:
        if not mutagen_session:
            mutagen_session = (
                discovered_session.get("name")
                or discovered_session.get("identifier")
                or ""
            )
        beta = discovered_session.get("beta", {})
        if not ssh_target:
            ssh_target = beta.get("host", "")
        if not remote_workspace_root_raw:
            remote_workspace_root_raw = beta.get("path", "")

    if not mutagen_session:
        raise RemoteRunError(
            "Unable to determine the Mutagen session. Set it in "
            f"{config_path} or pass --mutagen-session."
        )
    if not ssh_target:
        raise RemoteRunError(
            "Unable to determine the SSH target. Set it in "
            f"{config_path} or pass --ssh-target."
        )
    if not remote_workspace_root_raw:
        raise RemoteRunError(
            "Unable to determine the remote workspace root. Set it in "
            f"{config_path} or pass --remote-workspace-root."
        )

    bootstrap = pick("bootstrap", args.bootstrap)
    remote_python = pick("python", args.remote_python) or "python3"
    remote_shell = pick("remote_shell", args.remote_shell) or "bash"
    artifact_dir_raw = pick("artifact_dir", args.artifact_dir) or DEFAULT_ARTIFACT_DIR
    artifact_dir = resolve_path(artifact_dir_raw, repo_root)

    ssh_options_value = pick("ssh_options")
    ssh_options = shlex.split(ssh_options_value) if ssh_options_value else []
    if args.ssh_option:
        ssh_options.extend(args.ssh_option)

    return RuntimeConfig(
        config_path=config_path,
        local_workspace_root=local_workspace_root,
        mutagen_session=mutagen_session,
        ssh_target=ssh_target,
        remote_workspace_root=PurePosixPath(remote_workspace_root_raw),
        bootstrap=bootstrap,
        remote_python=remote_python,
        remote_shell=remote_shell,
        artifact_dir=artifact_dir,
        ssh_options=ssh_options,
    )


def map_local_to_remote(local_path: Path, config: RuntimeConfig) -> PurePosixPath:
    relative_path = maybe_relative_to(local_path, config.local_workspace_root)
    if relative_path is None:
        raise RemoteRunError(
            f"{local_path} is not inside the local workspace root "
            f"{config.local_workspace_root}."
        )
    return config.remote_workspace_root.joinpath(*relative_path.parts)


def map_absolute_path_argument(arg: str, config: RuntimeConfig) -> str:
    def map_candidate(candidate: str) -> Optional[str]:
        if not candidate.startswith("/"):
            return None
        local_path = Path(candidate)
        relative_path = maybe_relative_to(local_path, config.local_workspace_root)
        if relative_path is None:
            return None
        return str(config.remote_workspace_root.joinpath(*relative_path.parts))

    mapped = map_candidate(arg)
    if mapped:
        return mapped

    if arg.startswith("--") and "=" in arg:
        key, value = arg.split("=", 1)
        mapped_value = map_candidate(value)
        if mapped_value:
            return f"{key}={mapped_value}"

    return arg


def normalize_command_args(command_args: List[str], config: RuntimeConfig) -> List[str]:
    args = list(command_args)
    if args and args[0] == "--":
        args = args[1:]
    return [map_absolute_path_argument(arg, config) for arg in args]


def create_run_dir(config: RuntimeConfig, label: str) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    run_dir = config.artifact_dir / f"{timestamp}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_to_dict(config: RuntimeConfig) -> Dict[str, Any]:
    return {
        "config_path": str(config.config_path),
        "local_workspace_root": str(config.local_workspace_root),
        "mutagen_session": config.mutagen_session,
        "ssh_target": config.ssh_target,
        "remote_workspace_root": str(config.remote_workspace_root),
        "bootstrap": config.bootstrap,
        "remote_python": config.remote_python,
        "remote_shell": config.remote_shell,
        "artifact_dir": str(config.artifact_dir),
        "ssh_options": list(config.ssh_options),
    }


def build_wrapped_remote_command(
    config: RuntimeConfig,
    shell_payload: str,
) -> str:
    return f"{config.remote_shell} -lc {shlex.quote(shell_payload)}"


def run_ssh_capture(config: RuntimeConfig, remote_command: str) -> subprocess.CompletedProcess:
    ssh_command = ["ssh"] + config.ssh_options + [config.ssh_target, remote_command]
    try:
        return subprocess.run(
            ssh_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RemoteRunError("`ssh` is not installed or not available in PATH.") from exc


def resolve_workspace_path(candidate: str, cwd: Path, workspace_root: Path) -> Optional[Path]:
    if not candidate:
        return None

    expanded = Path(candidate).expanduser()
    local_path = expanded if expanded.is_absolute() else (cwd / expanded)
    try:
        resolved = local_path.resolve()
    except FileNotFoundError:
        resolved = local_path.absolute()

    if not resolved.exists():
        return None
    if maybe_relative_to(resolved, workspace_root) is None:
        return None
    return resolved


def discover_wait_paths(
    command_args: List[str],
    extra_wait_paths: List[str],
    cwd: Path,
    workspace_root: Path,
) -> List[Path]:
    wait_paths: List[Path] = []
    seen: set[str] = set()

    def append_if_workspace_path(candidate: str) -> None:
        path = resolve_workspace_path(candidate, cwd=cwd, workspace_root=workspace_root)
        if not path:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        wait_paths.append(path)

    for arg in command_args:
        if arg == "--":
            continue
        append_if_workspace_path(arg)
        if arg.startswith("--") and "=" in arg:
            _, value = arg.split("=", 1)
            append_if_workspace_path(value)

    for wait_path in extra_wait_paths:
        append_if_workspace_path(wait_path)

    return wait_paths


def wait_for_remote_sync(
    config: RuntimeConfig,
    wait_paths: List[Path],
    timeout_seconds: float = DEFAULT_SYNC_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    sync_root = config.artifact_dir / "sentinels"
    local_sentinel = sync_root / "local" / f"{token}.json"
    local_sentinel.parent.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, str]] = []
    for wait_path in wait_paths:
        if maybe_relative_to(wait_path, config.local_workspace_root) is None:
            raise RemoteRunError(
                f"Wait path is outside the local workspace root: {wait_path}"
            )

        remote_path = map_local_to_remote(wait_path, config)
        if wait_path.is_file():
            checks.append(
                {
                    "kind": "file_sha256",
                    "local_path": str(wait_path),
                    "remote_path": str(remote_path),
                    "sha256": sha256_file(wait_path),
                }
            )
        else:
            checks.append(
                {
                    "kind": "exists",
                    "local_path": str(wait_path),
                    "remote_path": str(remote_path),
                }
            )

    payload = {
        "sentinel": str(map_local_to_remote(local_sentinel, config)),
        "checks": [
            {
                "kind": check["kind"],
                "path": check["remote_path"],
                **({"sha256": check["sha256"]} if "sha256" in check else {}),
            }
            for check in checks
        ],
    }
    write_json(
        local_sentinel,
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "checks": checks,
        },
    )

    probe_code = """
import hashlib
import json
import pathlib
import sys

payload = json.loads(sys.argv[1])
if not pathlib.Path(payload["sentinel"]).exists():
    raise SystemExit(1)

for check in payload["checks"]:
    path = pathlib.Path(check["path"])
    kind = check["kind"]
    if kind == "exists":
        if not path.exists():
            raise SystemExit(1)
        continue

    if kind != "file_sha256" or not path.is_file():
        raise SystemExit(1)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != check["sha256"]:
        raise SystemExit(1)
"""
    probe_command_parts = [
        "set -euo pipefail",
        "cd " + shlex.quote(str(config.remote_workspace_root)),
    ]
    if config.bootstrap:
        probe_command_parts.append(config.bootstrap)
    probe_command_parts.append(
        " ".join(
            shlex.quote(part)
            for part in [
                config.remote_python,
                "-c",
                probe_code,
                json.dumps(payload, sort_keys=True),
            ]
        )
    )
    probe_command = build_wrapped_remote_command(config, " && ".join(probe_command_parts))

    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        result = run_ssh_capture(config, probe_command)
        if result.returncode == 0:
            return {
                "token": token,
                "local_sentinel": str(local_sentinel),
                "remote_sentinel": payload["sentinel"],
                "checks": checks,
            }
        last_error = (result.stderr or result.stdout or "").strip()
        time.sleep(DEFAULT_SYNC_POLL_INTERVAL_SECONDS)

    raise RemoteRunError(
        "Timed out waiting for the remote workspace to observe local changes. "
        f"Last remote error: {last_error or '<none>'}"
    )


def wait_for_local_post_run_sync(
    local_done_sentinel: Path,
    timeout_seconds: float = DEFAULT_SYNC_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if local_done_sentinel.exists():
            return
        time.sleep(DEFAULT_SYNC_POLL_INTERVAL_SECONDS)
    raise RemoteRunError(
        "Timed out waiting for the remote completion sentinel to sync back locally: "
        f"{local_done_sentinel}"
    )


def build_remote_check_command(config: RuntimeConfig) -> str:
    command_parts = [
        "set -euo pipefail",
        "cd " + shlex.quote(str(config.remote_workspace_root)),
    ]
    if config.bootstrap:
        command_parts.append(config.bootstrap)
    command_parts.append(
        " ".join(
            shlex.quote(part)
            for part in [
                config.remote_python,
                "-c",
                "import os,sys; print(os.getcwd()); print(sys.version)",
            ]
        )
    )
    return build_wrapped_remote_command(config, " && ".join(command_parts))


def build_remote_exec_command(
    config: RuntimeConfig,
    remote_cwd: PurePosixPath,
    command_args: List[str],
    remote_done_sentinel: PurePosixPath,
) -> str:
    if not command_args:
        raise RemoteRunError("No remote command was provided.")
    shell_segments: List[str] = [
        "set -euo pipefail",
        "cd " + shlex.quote(str(remote_cwd)),
    ]
    if config.bootstrap:
        shell_segments.append(config.bootstrap)
    shell_segments.append("mkdir -p " + shlex.quote(str(remote_done_sentinel.parent)))
    shell_segments.append("status=0")
    shell_segments.append(
        " ".join(shlex.quote(part) for part in command_args) + " || status=$?"
    )
    shell_segments.append(
        "printf '%s\\n' \"$status\" > " + shlex.quote(str(remote_done_sentinel))
    )
    shell_segments.append('exit "$status"')
    return build_wrapped_remote_command(config, "; ".join(shell_segments))


def infer_run_label(command_args: List[str], cwd: Path) -> str:
    args = [arg for arg in command_args if arg != "--"]
    if not args:
        return "command"

    if args[0] in {"python", "python3", "python3.10", "python3.11", "python3.12"} and len(args) > 1:
        script_candidate = resolve_workspace_path(args[1], cwd=cwd, workspace_root=find_repo_root(cwd))
        if script_candidate:
            return script_candidate.stem

    first = args[0]
    if first.startswith("--") and len(args) > 1:
        first = args[1]
    return Path(first).name or "command"


def stream_output(process: subprocess.Popen, stdout_path: Path, stderr_path: Path) -> None:
    def pump(stream: Any, log_path: Path, terminal: Any) -> None:
        if stream is None:
            return
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in iter(stream.readline, ""):
                terminal.write(line)
                terminal.flush()
                log_file.write(line)
                log_file.flush()
        stream.close()

    threads = [
        threading.Thread(
            target=pump,
            args=(process.stdout, stdout_path, sys.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=pump,
            args=(process.stderr, stderr_path, sys.stderr),
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run_ssh_command(
    config: RuntimeConfig,
    remote_command: str,
    run_dir: Path,
    metadata: Dict[str, Any],
) -> int:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    metadata["stdout_log"] = str(stdout_path)
    metadata["stderr_log"] = str(stderr_path)

    ssh_command = ["ssh"] + config.ssh_options + [config.ssh_target, remote_command]
    metadata["ssh_command"] = ssh_command

    print(f"[remote-run] SSH target: {config.ssh_target}")
    print(f"[remote-run] Logs: {run_dir}")

    try:
        process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RemoteRunError("`ssh` is not installed or not available in PATH.") from exc

    stream_output(process, stdout_path=stdout_path, stderr_path=stderr_path)
    return process.wait()


def print_resolved_config(config: RuntimeConfig) -> None:
    print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))


def handle_check(args: argparse.Namespace) -> int:
    config = resolve_runtime_config(args)
    print_resolved_config(config)

    run_dir = create_run_dir(config, label="check")
    remote_command = build_remote_check_command(config)

    metadata: Dict[str, Any] = {
        "kind": "check",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resolved_config": config_to_dict(config),
        "remote_command": remote_command,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metadata.json", metadata)
    exit_code = run_ssh_command(config, remote_command, run_dir, metadata)
    metadata["exit_code"] = exit_code
    metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(run_dir / "metadata.json", metadata)
    write_json(config.artifact_dir / "last-run.json", metadata)
    return exit_code


def handle_run(args: argparse.Namespace) -> int:
    config = resolve_runtime_config(args)
    local_cwd = Path.cwd().resolve()
    remote_cwd = (
        map_local_to_remote(local_cwd, config)
        if maybe_relative_to(local_cwd, config.local_workspace_root) is not None
        else config.remote_workspace_root
    )
    raw_command_args = list(args.command_args)
    if raw_command_args and raw_command_args[0] == "--":
        raw_command_args = raw_command_args[1:]
    if not raw_command_args:
        raise RemoteRunError("No remote command was provided.")

    command_args = normalize_command_args(raw_command_args, config)
    wait_paths = discover_wait_paths(
        command_args=raw_command_args,
        extra_wait_paths=args.wait_path,
        cwd=local_cwd,
        workspace_root=config.local_workspace_root,
    )
    done_token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    local_done_sentinel = config.artifact_dir / "sentinels" / "remote" / f"{done_token}.done"
    remote_done_sentinel = map_local_to_remote(local_done_sentinel, config)
    remote_command = build_remote_exec_command(
        config=config,
        remote_cwd=remote_cwd,
        command_args=command_args,
        remote_done_sentinel=remote_done_sentinel,
    )

    run_dir = create_run_dir(config, label=infer_run_label(raw_command_args, local_cwd))
    metadata: Dict[str, Any] = {
        "kind": "run",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "local_workspace_root": str(config.local_workspace_root),
        "local_cwd": str(local_cwd),
        "remote_cwd": str(remote_cwd),
        "local_command_args": raw_command_args,
        "remote_command_args": command_args,
        "mutagen_session": config.mutagen_session,
        "ssh_target": config.ssh_target,
        "remote_workspace_root": str(config.remote_workspace_root),
        "bootstrap": config.bootstrap,
        "remote_python": config.remote_python,
        "remote_shell": config.remote_shell,
        "local_done_sentinel": str(local_done_sentinel),
        "remote_done_sentinel": str(remote_done_sentinel),
        "wait_path": [str(path) for path in wait_paths],
        "run_dir": str(run_dir),
        "remote_command": remote_command,
    }
    write_json(run_dir / "metadata.json", metadata)

    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        write_json(config.artifact_dir / "last-run.json", metadata)
        return 0

    if args.skip_pre_sync:
        print("[remote-run] Skipping the pre-run sync barrier.")
    else:
        print("[remote-run] Waiting for the remote workspace to observe local changes.")
        metadata["pre_run_sync"] = wait_for_remote_sync(
            config=config,
            wait_paths=wait_paths,
        )
        write_json(run_dir / "metadata.json", metadata)

    exit_code = run_ssh_command(config, remote_command, run_dir, metadata)
    metadata["exit_code"] = exit_code

    if args.skip_post_sync:
        print("[remote-run] Skipping the post-run sync barrier.")
    else:
        print("[remote-run] Waiting for the remote completion sentinel to sync back locally.")
        wait_for_local_post_run_sync(local_done_sentinel)

    metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(run_dir / "metadata.json", metadata)
    write_json(config.artifact_dir / "last-run.json", metadata)

    print(f"[remote-run] Exit code: {exit_code}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a command on the remote side of the workspace's Mutagen sync."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"INI config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    common.add_argument("--mutagen-session", help="Mutagen session name or identifier.")
    common.add_argument("--ssh-target", help="SSH target, for example user@host.")
    common.add_argument(
        "--remote-workspace-root",
        help="Absolute remote workspace root mirrored by Mutagen.",
    )
    common.add_argument(
        "--local-workspace-root",
        help="Local workspace root. Defaults to the git repo root.",
    )
    common.add_argument(
        "--bootstrap",
        help="Shell snippet to run on the remote host before Python starts.",
    )
    common.add_argument(
        "--python",
        dest="remote_python",
        help="Remote Python executable. Defaults to python3.",
    )
    common.add_argument(
        "--remote-shell",
        help="Remote shell used to execute the command. Defaults to bash.",
    )
    common.add_argument(
        "--artifact-dir",
        help=f"Local directory for logs and metadata. Defaults to {DEFAULT_ARTIFACT_DIR}.",
    )
    common.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra SSH option. May be passed multiple times.",
    )
    common.add_argument(
        "--skip-pre-sync",
        "--skip-pre-flush",
        action="store_true",
        dest="skip_pre_sync",
        help="Skip waiting for the pre-run sync barrier.",
    )
    common.add_argument(
        "--skip-post-sync",
        "--skip-post-flush",
        action="store_true",
        dest="skip_post_sync",
        help="Skip waiting for the post-run sync barrier.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check",
        parents=[common],
        help="Verify the resolved remote configuration and execute a tiny remote Python check.",
    )
    check_parser.set_defaults(handler=handle_check)

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run an arbitrary command on the remote host from the mirrored workspace cwd.",
    )
    run_parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Command and arguments to execute remotely.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved remote command and exit without running it.",
    )
    run_parser.add_argument(
        "--wait-path",
        action="append",
        default=[],
        help="Additional local workspace file or directory to verify before the remote run starts.",
    )
    run_parser.set_defaults(handler=handle_run)

    return parser


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    if argv and argv[0] not in {"check", "run", "-h", "--help"}:
        argv = ["run"] + argv
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except RemoteRunError as exc:
        print(f"[remote-run] Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[remote-run] Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
