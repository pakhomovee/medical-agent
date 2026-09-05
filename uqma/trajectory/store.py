"""Reading and writing run artifacts.

A run directory is immutable once written and named ``<date>-<model>-<confighash>``.
Every run carries a manifest with the git SHA, the resolved config and library versions,
because across ~55 sweeps and 38 weeks a run you cannot attribute to a commit is a run
you have to repeat (plan §6.3).

Trajectories stream to JSONL as they complete rather than accumulating in memory: full
transcripts across a sweep do not fit comfortably in RAM, and a crash halfway through
should leave the completed half readable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .schema import SCHEMA_VERSION, Trajectory

MANIFEST_NAME = "manifest.json"
TRAJECTORIES_NAME = "trajectories.jsonl"


def git_sha(short: bool = False) -> str | None:
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    Recorded because a run produced from a dirty tree is not reproducible from its SHA,
    and finding that out in month eight is worse than a warning now.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def _library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("requests", "vllm", "torch", "transformers", "numpy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        out[name] = getattr(module, "__version__", "unknown")
    return out


@dataclass
class Manifest:
    run_id: str
    created_utc: str
    schema_version: str = SCHEMA_VERSION
    git_sha: str | None = None
    git_dirty: bool | None = None
    config: dict = field(default_factory=dict)
    config_hash: str = ""
    libraries: dict = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    hostname: str = ""
    notes: str = ""

    @classmethod
    def create(cls, config: dict, run_id: str | None = None, notes: str = "") -> Manifest:
        digest = config_hash(config)
        stamp = datetime.now(timezone.utc)
        return cls(
            run_id=run_id or default_run_id(config, stamp, digest),
            created_utc=stamp.isoformat(),
            git_sha=git_sha(),
            git_dirty=git_dirty(),
            config=config,
            config_hash=digest,
            libraries=_library_versions(),
            argv=list(sys.argv),
            hostname=platform.node(),
            notes=notes,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config": self.config,
            "config_hash": self.config_hash,
            "libraries": self.libraries,
            "argv": self.argv,
            "hostname": self.hostname,
            "notes": self.notes,
        }


def default_run_id(config: dict, stamp: datetime, digest: str) -> str:
    model = str(config.get("model") or config.get("backend") or "run")
    slug = "".join(c if c.isalnum() or c in "-." else "-" for c in model).strip("-")[:40]
    return f"{stamp:%Y%m%d-%H%M%S}-{slug}-{digest}"


class RunWriter:
    """Creates a run directory and streams trajectories into it.

    Refuses to write into a directory that already holds trajectories, because runs are
    immutable — appending to a finished run silently mixes two configurations.
    """

    def __init__(self, root: str | Path, manifest: Manifest, overwrite: bool = False) -> None:
        self.dir = Path(root) / manifest.run_id if Path(root).name != manifest.run_id else Path(root)
        self.manifest = manifest
        self._handle = None

        traj_path = self.dir / TRAJECTORIES_NAME
        if traj_path.exists() and traj_path.stat().st_size > 0 and not overwrite:
            raise FileExistsError(
                f"{traj_path} already has data. Runs are immutable; pick a new run_id or "
                "pass overwrite=True deliberately."
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        self._path = traj_path

    def __enter__(self) -> RunWriter:
        self._handle = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, trajectory: Trajectory) -> None:
        if self._handle is None:
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(trajectory.to_dict()) + "\n")
        self._handle.flush()  # a crash should leave completed trajectories readable
        os.fsync(self._handle.fileno())

    def write_artifact(self, name: str, payload: dict) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read_manifest(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def iter_trajectories(run_dir: str | Path) -> Iterator[Trajectory]:
    """Stream trajectories back. Lazy so that stage 4/5 never loads a sweep at once."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / TRAJECTORIES_NAME
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trajectory.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def check_compatible(run_dir: str | Path) -> None:
    """Fail loudly when a run was written by a different schema version."""
    version = read_manifest(run_dir).get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{run_dir} was written with schema {version}, this code expects "
            f"{SCHEMA_VERSION}. Write a migration; do not read it directly."
        )
