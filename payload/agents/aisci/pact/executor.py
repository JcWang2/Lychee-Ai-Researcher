# -*- coding: utf-8 -*-
"""pact/executor.py - Isolated execution of a frozen TrialSpec.

Deterministic, no LLM. Runs the candidate code in a dedicated work directory
with a hard timeout, capturing stdout/stderr and the exit code.

Two backends:
  - host:      isolated subprocess on the host (default, zero dependencies)
  - container: v6-style execution container via `docker run` when
               V2_EXEC_IMAGE is set; falls back to host when unavailable.
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from v2_contracts import TrialSpec


@dataclass
class ExecOutcome:
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    wall_clock_seconds: float = 0.0
    error: str = ""
    trial_work_dir: str = ""


class Executor:
    """Runs candidate code inside a dedicated work directory.

    Container mode (V2_EXEC_IMAGE): the same absolute data/work paths are
    mounted into the container so LLM-written paths keep working. GPU is
    passed through via CUDA_VISIBLE_DEVICES. Falls back to host subprocess
    when the docker daemon is unavailable.
    """

    def __init__(self, work_dir, python_bin: Optional[str] = None,
                 exec_image: str = "", exec_python: str = "",
                 docker_bin: str = "", data_dir: str = "",
                 manifest: Optional[dict] = None,
                 torch_cache: str = "",
                 hf_cache: str = "",
                 soft_kill_grace: float = 30.0):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.python_bin = python_bin or sys.executable
        self.exec_image = exec_image or os.environ.get("V2_EXEC_IMAGE", "")
        self.exec_python = (exec_python or os.environ.get("V2_EXEC_PYTHON", "")
                            or "python3")
        self.docker_bin = docker_bin or os.environ.get("V2_DOCKER_BIN", "") \
            or "docker"
        self.data_dir = str(data_dir or "")
        self.manifest = dict(manifest or {})
        self.torch_cache = torch_cache or os.environ.get("V2_TORCH_CACHE", "")
        # v2.3.9: huggingface-hub weight cache (timm pretrained= checkpoints
        # download to HF hub by default, NOT torch hub). Default: explicit
        # V2_HF_CACHE, else a v2_hf_cache sibling of the torch cache.
        self.hf_cache = hf_cache or os.environ.get("V2_HF_CACHE", "")
        if not self.hf_cache and self.torch_cache:
            # v2.3.9: default to a v2_hf_cache sibling of the torch cache even
            # when the directory does not exist yet (docker -v creates host
            # bind dirs on first mount), so the HF mount is deterministic.
            self.hf_cache = os.path.join(os.path.dirname(
                self.torch_cache.rstrip("/\\") or "/"), "v2_hf_cache")
        self.soft_kill_grace = max(0.0, float(
            os.environ.get("V2_SOFT_KILL_GRACE", soft_kill_grace)))

    def set_manifest(self, manifest: Optional[dict]) -> None:
        self.manifest = dict(manifest or {})

    def _mount_root(self) -> str:
        """Mount public/ (label-free) only; fail closed on a broken contract.

        private/ (gold labels) is intentionally NOT mounted: candidate code
        must be physically unable to read the labelled test set. Falling back
        to the full data root would silently expose gold, so a manifest
        without public_dir aborts loudly instead of leaking.
        """
        public_dir = self.manifest.get("public_dir", "")
        if not public_dir:
            raise ValueError(
                "manifest missing 'public_dir': refusing to mount the data "
                "root (would expose private/gold labels)")
        return public_dir

    def _manifest_env(self, env: Optional[dict]) -> dict:
        """Add the data contract env vars for candidate code."""
        env = dict(env or {})
        for key, mkey in (("DATA_DIR", "dataset_root"),
                          ("TRAIN_CSV", "train_csv"),
                          ("TEST_CSV", "test_csv"),
                          ("SAMPLE_SUBMISSION", "sample_submission"),
                          ("TRAIN_IMAGES", "train_images"),
                          ("TEST_IMAGES", "test_images"),
                          ("TARGET_COLUMN", "target_column"),
                          ("TASK_TYPE", "task_type"),
                          ("MASK_TARGET", "mask_target"),
                          ("BBOX_COLUMNS", "bbox_columns"),
                          ("MULTI_ROW_TARGET", "multi_row_target"),
                          ("AUDIO_FILE_COUNT", "audio_file_count")):
            value = self.manifest.get(mkey)
            if value:
                env[key] = str(value)
        # v2.3.8: JSON-encode the bbox column list (templates read it).
        bbox = self.manifest.get("bbox_columns")
        if bbox:
            try:
                env["BBOX_COLUMNS"] = json.dumps(
                    [str(c) for c in bbox], ensure_ascii=False)
            except Exception:  # noqa: BLE001 - best-effort
                pass
        if self.manifest.get("multi_row_target"):
            env["MULTI_ROW_TARGET"] = "1"
        # v2.2.1-rc4: multi-size zero-decode image caches. The dirs live
        # under work_dir (already mounted), so candidate code only needs
        # the {size: dir} map to LOAD instead of decode.
        cache_dirs = self.manifest.get("cache_dirs")
        if cache_dirs:
            try:
                # v2.3.2: only hand out cache dirs that actually exist; a
                # missing dir must never become a hard trial failure (the
                # harness falls back to raw decode when no cache is offered).
                keep = {k: v for k, v in cache_dirs.items()
                        if v and os.path.isdir(str(v))}
                if keep:
                    env["V2_CACHE_DIRS"] = json.dumps(keep)
            except Exception:  # noqa: BLE001 - cache map is best-effort
                pass
        return env

    def exec_mode(self) -> str:
        if self.exec_image and self._docker_available():
            return "container"
        return "host"

    def _docker_available(self) -> bool:
        try:
            probe = subprocess.run(
                [self.docker_bin, "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=15)
            return probe.returncode == 0 and bool(
                (probe.stdout or b"").decode("utf-8", "replace").strip())
        except Exception:  # noqa: BLE001 - daemon down or binary missing
            return False

    def docker_cmd(self, spec: TrialSpec, script_name: str,
                   work_dir: str, env: dict) -> list:
        """Build the `docker run` argv for one trial (unit-testable)."""
        cmd = [self.docker_bin, "run", "--rm", "--name",
               "v2_exec_%s_%s" % (spec.spec_id[:8], int(time.time())),
               "--shm-size=1g",
               "-e", "PYTHONUNBUFFERED=1"]
        # CPU thread caps: torch/OpenMP default to ALL host cores per
        # container; 3 containers x 100+ threads oversubscribe the box,
        # spin-wait, and starve the GPU (GPU-idle rc=-9 symptom).
        threads = str(max(1, int(os.environ.get("V2_CPU_THREADS", "8"))))
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
            cmd += ["-e", "%s=%s" % (_var, threads)]
        devices = env.get("CUDA_VISIBLE_DEVICES", "")
        if devices:
            cmd += ["-e", "CUDA_VISIBLE_DEVICES=" + devices, "--gpus", "all"]
        cmd += ["-v", "%s:%s" % (work_dir, work_dir)]
        seal_file = os.path.join(work_dir, "sealed_spec.json")
        if os.path.isfile(seal_file):
            # frozen-skill-path immutability: the seal record is visible
            # read-only inside the container; candidate code cannot alter it
            cmd += ["-v", "%s:%s:ro" % (seal_file, seal_file)]
        if self.torch_cache:
            # shared pretrained-weight cache (preflight-verified): all trials
            # see the same checkpoints without re-downloading per container
            cmd += ["-v", "%s:/root/.cache/torch" % self.torch_cache]
        if self.hf_cache:
            # v2.3.9: huggingface hub cache (timm downloads pretrained
            # weights here by default); share so trials never re-download
            cmd += ["-v", "%s:/root/.cache/huggingface" % self.hf_cache]
        mount_root = self._mount_root()
        if mount_root:
            cmd += ["-v", "%s:%s:ro" % (mount_root, mount_root)]
        for key, value in self._manifest_env({}).items():
            cmd += ["-e", "%s=%s" % (key, value)]
        tokens = shlex.split(self.exec_python) or ["python3"]
        # Explicit --entrypoint bypasses any image ENTRYPOINT wrapper so the
        # candidate script runs directly under the chosen interpreter.
        cmd += ["-w", work_dir, "--entrypoint", tokens[0], self.exec_image]
        cmd += tokens[1:] + [script_name]
        return cmd

    def run(self, spec: TrialSpec, timeout_seconds: int) -> ExecOutcome:
        trial_dir = self.work_dir / spec.spec_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Frozen-skill-path immutability: a spec may only execute under the
        # code hash it was sealed with. Mismatch -> fail closed before any
        # execution (rc=-4), so a tampered/partial artifact never runs.
        if spec.code_hash:
            digest = "sha256:" + hashlib.sha256(
                spec.code.encode("utf-8")).hexdigest()
            if digest != spec.code_hash:
                return ExecOutcome(
                    returncode=-4, stdout="", stderr="SEAL_MISMATCH code_hash=%s"
                    % spec.code_hash, timed_out=False, wall_clock_seconds=0.0,
                    error="seal mismatch: code does not match sealed hash",
                    trial_work_dir=str(trial_dir))

        script = trial_dir / ("round_%d_%s.py" % (spec.round_num, spec.spec_id[:8]))
        # write_bytes: no newline translation (CRLF would change the hash)
        script.write_bytes(spec.code.encode("utf-8"))

        # Re-verify the bytes on disk (defends against any write-path
        # tampering between seal and execution).
        disk_hash = "sha256:" + hashlib.sha256(
            script.read_bytes()).hexdigest()
        if spec.code_hash and disk_hash != spec.code_hash:
            return ExecOutcome(
                returncode=-4, stdout="", stderr="SEAL_MISMATCH_DISK %s" % disk_hash,
                timed_out=False, wall_clock_seconds=0.0,
                error="seal mismatch: script bytes on disk differ from seal",
                trial_work_dir=str(trial_dir))

        # Materialize the immutable seal record next to the script (mounted
        # read-only into the container); the authoritative copy lives in
        # pact_control_host/state/pact/specs/.
        seal_path = trial_dir / "sealed_spec.json"
        seal_path.write_text(
            json.dumps(spec.seal_record(), ensure_ascii=False, default=str),
            encoding="utf-8")

        if self.exec_mode() == "container":
            return self._run_container(spec, script.name, trial_dir, timeout_seconds)
        return self._run_host(spec, script, trial_dir, timeout_seconds)

    def _run_host(self, spec: TrialSpec, script: Path,
                  trial_dir: Path, timeout_seconds: int) -> ExecOutcome:
        env = self._manifest_env(os.environ.copy())
        env["PYTHONUNBUFFERED"] = "1"
        started = time.time()
        try:
            proc = subprocess.run(
                [self.python_bin, str(script)],
                capture_output=True,
                timeout=max(1, int(timeout_seconds)),
                env=env,
                cwd=str(trial_dir),
            )
            return ExecOutcome(
                returncode=proc.returncode,
                stdout=(proc.stdout or b"").decode("utf-8", "replace"),
                stderr=(proc.stderr or b"").decode("utf-8", "replace"),
                timed_out=False,
                wall_clock_seconds=time.time() - started,
                trial_work_dir=str(trial_dir),
            )
        except subprocess.TimeoutExpired as exc:
            so = (exc.stdout or b"").decode("utf-8", "replace")
            se = (exc.stderr or b"").decode("utf-8", "replace")
            return ExecOutcome(
                returncode=-9, stdout=so,
                stderr=(se + "\nTIMEOUT").strip(),
                timed_out=True,
                wall_clock_seconds=time.time() - started,
                error="timeout after %ss" % timeout_seconds,
                trial_work_dir=str(trial_dir),
            )
        except Exception as e:  # noqa: BLE001 - fail-closed
            return ExecOutcome(
                returncode=-2, stdout="", stderr=str(e),
                wall_clock_seconds=time.time() - started,
                error=str(e),
                trial_work_dir=str(trial_dir),
            )

    def _run_container(self, spec: TrialSpec, script_name: str,
                       trial_dir: Path, timeout_seconds: int) -> ExecOutcome:
        env = self._manifest_env(os.environ.copy())
        env["PYTHONUNBUFFERED"] = "1"
        cmd = self.docker_cmd(spec, script_name, str(trial_dir), env)
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=max(1, int(timeout_seconds)),
                env=env,
                cwd=str(trial_dir),
            )
            return ExecOutcome(
                returncode=proc.returncode,
                stdout=(proc.stdout or b"").decode("utf-8", "replace"),
                stderr=(proc.stderr or b"").decode("utf-8", "replace"),
                timed_out=False,
                wall_clock_seconds=time.time() - started,
                trial_work_dir=str(trial_dir),
            )
        except subprocess.TimeoutExpired as exc:
            name = cmd[4]  # argv: docker run --rm --name NAME ...
            so = (exc.stdout or b"").decode("utf-8", "replace")
            se = (exc.stderr or b"").decode("utf-8", "replace")
            tail = so + se
            try:
                # soft stop: SIGINT gives candidate code a chance
                # to catch BaseException and write artifacts
                subprocess.run(
                    [self.docker_bin, "kill", "--signal", "SIGINT", name],
                    capture_output=True, timeout=20)
            except Exception:  # noqa: BLE001 - container gone
                pass
            time.sleep(self.soft_kill_grace)
            try:
                logs = subprocess.run(
                    [self.docker_bin, "logs", "--tail", "300", name],
                    capture_output=True, timeout=20)
                if logs.stdout or logs.stderr:
                    tail = (tail + "\n"
                            + (logs.stdout or b"").decode("utf-8", "replace")
                            + (logs.stderr or b"").decode("utf-8", "replace")
                            ).strip()
            except Exception:  # noqa: BLE001 - already removed
                pass
            try:
                subprocess.run([self.docker_bin, "kill", name],
                               capture_output=True,
                               timeout=20)
            except Exception:  # noqa: BLE001 - already gone
                pass
            if tail.strip():
                try:
                    (trial_dir / "container.log").write_text(
                        tail, encoding="utf-8", errors="replace")
                except OSError:
                    pass
            return ExecOutcome(
                returncode=-9, stdout=so,
                stderr=(se + "\nTIMEOUT").strip(),
                timed_out=True,
                wall_clock_seconds=time.time() - started,
                error="timeout after %ss" % timeout_seconds,
                trial_work_dir=str(trial_dir),
            )
        except Exception as e:  # noqa: BLE001 - fail-closed
            return ExecOutcome(
                returncode=-2, stdout="", stderr=str(e),
                wall_clock_seconds=time.time() - started,
                error=str(e),
                trial_work_dir=str(trial_dir),
            )

    def preflight(self, required_imports: Optional[list] = None,
                  timeout_seconds: int = 180) -> dict:
        """Preflight inside the execution container (imports + data files).

        Runs before round 1 so dependency/data-path problems fail fast with
        an actionable report instead of N identical trial failures.
        Returns a dict: mode/status/missing_modules/missing_files/detail.
        """
        if self.exec_mode() != "container":
            return {"mode": "host", "status": "skipped",
                    "detail": "container mode not active"}
        imports = list(required_imports or (
            "numpy", "pandas", "sklearn", "scipy", "PIL", "cv2",
            "torch", "torchvision"))
        paths = [self.manifest.get("train_csv", ""),
                 self.manifest.get("test_csv", ""),
                 self.manifest.get("sample_submission", ""),
                 self.manifest.get("train_images", ""),
                 self.manifest.get("test_images", "")]
        paths = [p for p in paths if p]
        script = (
            "import importlib.util, os, glob\n"
            "missing = [m for m in %r if importlib.util.find_spec(m) is None]\n"
            "print('PREFLIGHT_MISSING=' + ','.join(missing))\n"
            "miss_files = [p for p in %r if not os.path.isdir(p) and not os.path.isfile(p)]\n"
            "print('PREFLIGHT_MISSING_FILES=' + ','.join(miss_files))\n"
            "ckpts = sorted(glob.glob(os.path.expanduser('~/.cache/torch/hub/checkpoints/*')))\n"
            "hf_repos = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--*')))\n"
            "print('PREFLIGHT_PRETRAINED=' + ','.join(os.path.basename(p) for p in ckpts))\n"
            "print('PREFLIGHT_HF_PRETRAINED=' + ','.join(os.path.basename(p) for p in hf_repos))\n"
            "print('PREFLIGHT_OK')\n" % (list(imports), paths))
        cmd = [self.docker_bin, "run", "--rm", "--shm-size=1g",
               "-e", "PYTHONUNBUFFERED=1"]
        # CPU thread caps: torch/OpenMP default to ALL host cores per
        # container; 3 containers x 100+ threads oversubscribe the box,
        # spin-wait, and starve the GPU (GPU-idle rc=-9 symptom).
        threads = str(max(1, int(os.environ.get("V2_CPU_THREADS", "8"))))
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
            cmd += ["-e", "%s=%s" % (_var, threads)]
        for key, value in self._manifest_env({}).items():
            cmd += ["-e", "%s=%s" % (key, value)]
        if self.torch_cache:
            cmd += ["-v", "%s:/root/.cache/torch" % self.torch_cache]
        if self.hf_cache:
            cmd += ["-v", "%s:/root/.cache/huggingface" % self.hf_cache]
        mount_root = self._mount_root()
        if mount_root:
            cmd += ["-v", "%s:%s:ro" % (mount_root, mount_root)]
        else:
            for p in paths:
                cmd += ["-v", "%s:%s:ro" % (p, p)]
        gold = self.manifest.get("gold_test_csv", "") or ""
        script += (
            "gold = %r\n"
            "print('PREFLIGHT_GOLD_VISIBLE=' + ('YES' if gold and "
            "os.path.exists(gold) else 'NO'))\n" % gold)
        tokens = shlex.split(self.exec_python) or ["python3"]
        cmd += ["--entrypoint", tokens[0], self.exec_image]
        cmd += tokens[1:] + ["-c", script]
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  timeout=max(10, int(timeout_seconds)))
        except Exception as exc:  # noqa: BLE001 - preflight must not crash the loop
            return {"mode": "container", "status": "error",
                    "missing_modules": [], "missing_files": [],
                    "detail": str(exc)[:500]}
        out = ((proc.stdout or b"").decode("utf-8", "replace")
               + "\n" + (proc.stderr or b"").decode("utf-8", "replace"))
        missing = []
        miss_files = []
        gold_visible = False
        pretrained = []
        for line in out.splitlines():
            if line.startswith("PREFLIGHT_MISSING="):
                missing = [x for x in line.split("=", 1)[1].split(",") if x]
            elif line.startswith("PREFLIGHT_MISSING_FILES="):
                miss_files = [x for x in line.split("=", 1)[1].split(",") if x]
            elif line.startswith("PREFLIGHT_GOLD_VISIBLE=YES"):
                gold_visible = True
            elif line.startswith("PREFLIGHT_PRETRAINED="):
                pretrained = [x for x in line.split("=", 1)[1].split(",") if x]
            elif line.startswith("PREFLIGHT_HF_PRETRAINED="):
                # v2.3.9: HF-hub repos (timm pretrained= path) count as
                # cached weights too; merged into the same whitelist so the
                # resource profiler sees them without any task-name logic.
                hf = [x for x in line.split("=", 1)[1].split(",") if x]
                pretrained = pretrained + [h for h in hf if h not in pretrained]
        ok = (proc.returncode == 0 and not missing and not miss_files
              and not gold_visible)
        return {
            "mode": "container",
            "status": "ok" if ok else "fail",
            "returncode": proc.returncode,
            "missing_modules": missing,
            "missing_files": miss_files,
            "gold_visible": gold_visible,
            "pretrained_available": pretrained,
            "detail": (out or "")[-800:],
        }
