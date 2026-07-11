from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

from traceunit.io import (
    expand_placeholders,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_tree,
    write_json,
)
from traceunit.models import (
    EvidenceRole,
    TestCaseSpec,
    TestExecution,
    TestExecutionMode,
    TestPacket,
    TestStatus,
    TestTier,
)
from traceunit.ontology import ontology_ref, validate_ontology_ref


class InvalidTestPacket(ValueError):
    pass


class TestSandboxUnavailable(RuntimeError):
    """Raised when generated test code cannot be isolated safely."""


_SANDBOX_MODE_ENV = "TRACEUNIT_TEST_SANDBOX_MODE"
_PROTECTED_ENV_KEYS = {
    "HOME",
    "PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "TMPDIR",
    "TRACEUNIT_SOURCE",
    "TRACEUNIT_SUBJECT",
    "TRACEUNIT_TEST_BUNDLE",
}
_PROTECTED_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "CLAUDE_",
    "CODEX_",
    "DEEPSEEK_",
    "GITHUB_",
    "GOOGLE_",
    "HF_",
    "LD_",
    "OPENAI_",
    "TOGETHER_",
)
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_test_packet(bundle: Path) -> TestPacket:
    path = bundle / "test_packet.json"
    if not path.is_file():
        raise InvalidTestPacket(f"missing {path}")
    try:
        packet = TestPacket.from_dict(read_json(path))
    except Exception as exc:
        raise InvalidTestPacket(f"invalid test_packet.json: {exc}") from exc
    validate_test_packet(packet, bundle)
    return packet


def validate_test_packet(packet: TestPacket, bundle: Path) -> None:
    claimed_ontology = packet.metadata.get("ontology")
    if claimed_ontology is not None and not validate_ontology_ref(claimed_ontology):
        raise InvalidTestPacket("packet claims an unknown L0 ontology version or hash")
    if packet.status is TestStatus.ADMITTED and claimed_ontology is None:
        raise InvalidTestPacket(
            "admitted packet is not bound to the frozen L0 ontology"
        )
    if packet.status is not TestStatus.ADMITTED and packet.content_sha256:
        raise InvalidTestPacket(
            "content_sha256 must stay empty until the harness freezes the packet"
        )
    regression_only = packet.metadata.get("packet_kind") == "regression"
    hypothesis_ids = {item.hypothesis_id for item in packet.hypotheses}
    if not hypothesis_ids:
        raise InvalidTestPacket("at least one failure hypothesis is required")
    if packet.target_hypothesis_id not in hypothesis_ids:
        raise InvalidTestPacket("target_hypothesis_id is not present in hypotheses")
    declared_traces = set(packet.source_trace_ids)
    for hypothesis in packet.hypotheses:
        if not hypothesis.evidence_trace_ids:
            raise InvalidTestPacket(
                f"{hypothesis.hypothesis_id}: evidence_trace_ids must not be empty"
            )
        if not set(hypothesis.evidence_trace_ids) <= declared_traces:
            raise InvalidTestPacket(
                f"{hypothesis.hypothesis_id}: evidence trace is not declared by packet"
            )
        if not 0.0 <= hypothesis.confidence <= 1.0:
            raise InvalidTestPacket(
                f"{hypothesis.hypothesis_id}: confidence must be in [0, 1]"
            )
    if not regression_only:
        if len(hypothesis_ids) < 2:
            raise InvalidTestPacket(
                "normal packet must contain at least two competing failure hypotheses"
            )
        target = next(
            item
            for item in packet.hypotheses
            if item.hypothesis_id == packet.target_hypothesis_id
        )
        if packet.primary_family is None:
            raise InvalidTestPacket("normal packet requires primary_family")
        if target.family is not packet.primary_family:
            raise InvalidTestPacket(
                "target hypothesis family must equal packet primary_family"
            )
        valid_alternatives = set(target.alternatives) & hypothesis_ids
        if not valid_alternatives:
            raise InvalidTestPacket(
                "target hypothesis must name at least one alternative in the packet"
            )
    elif packet.primary_family is not None:
        raise InvalidTestPacket("regression-only packet must not claim primary_family")
    if not packet.cases:
        raise InvalidTestPacket("at least one test case is required")
    ids: set[str] = set()
    tiers: set[TestTier] = set()
    for case in packet.cases:
        if not _SAFE_CASE_ID.fullmatch(case.case_id):
            raise InvalidTestPacket(f"unsafe case_id: {case.case_id!r}")
        if case.case_id in ids:
            raise InvalidTestPacket(f"duplicate case_id: {case.case_id}")
        ids.add(case.case_id)
        tiers.add(case.tier)
        expected_roles = {
            TestTier.PUBLIC: {EvidenceRole.TARGET_REPRODUCER},
            TestTier.HIDDEN: {EvidenceRole.STRUCTURAL_SIBLING},
            TestTier.BRIDGE: {EvidenceRole.DOWNSTREAM_BRIDGE},
            TestTier.ADMISSION: {EvidenceRole.POSITIVE_WITNESS},
            TestTier.REGRESSION: {
                EvidenceRole.PRESERVATION_CONTROL,
                EvidenceRole.OFF_TARGET_CONTROL,
            },
        }
        if case.evidence_role not in expected_roles[case.tier]:
            raise InvalidTestPacket(
                f"{case.case_id}: evidence_role {case.evidence_role.value!r} "
                f"is invalid for tier {case.tier.value!r}"
            )
        if case.execution_mode is TestExecutionMode.MODEL_BACKED_PROBE:
            if case.driver != "agent_probe":
                raise InvalidTestPacket(
                    f"{case.case_id}: model-backed probe requires driver='agent_probe'"
                )
            if case.max_model_calls < 1 or case.max_tokens < 1:
                raise InvalidTestPacket(
                    f"{case.case_id}: model-backed probe requires positive call/token budgets"
                )
        elif case.driver not in {"python", "pytest"}:
            raise InvalidTestPacket(f"unsupported driver {case.driver!r}")
        for key in case.environment:
            upper = key.upper()
            if upper in _PROTECTED_ENV_KEYS or upper.startswith(
                _PROTECTED_ENV_PREFIXES
            ):
                raise InvalidTestPacket(
                    f"{case.case_id}: environment may not override protected key {key!r}"
                )
        unresolved_path = bundle / case.path
        if unresolved_path.is_symlink():
            raise InvalidTestPacket(f"test files may not be symlinks: {case.path}")
        path = safe_relative_path(bundle, case.path)
        if not path.is_file():
            raise InvalidTestPacket(f"test file does not exist: {case.path}")
        test_text = path.read_text(encoding="utf-8", errors="replace")
        forbidden = {
            "ground_truth": "ground-truth access",
            "world.evaluate": "benchmark evaluator access",
            "evaluate_task(": "benchmark evaluator access",
            "swebench.harness": "SWE-bench evaluator access",
            "gold_patch": "gold patch access",
            "test_patch": "hidden test patch access",
            "/data/": "absolute host data path",
            "/home/": "absolute host home path",
        }
        for token, reason in forbidden.items():
            if token in test_text:
                raise InvalidTestPacket(
                    f"{case.path} contains forbidden {reason}: {token!r}"
                )
        if case.tier == TestTier.PUBLIC and "public" not in path.parts:
            raise InvalidTestPacket(
                f"public test must live under a public directory: {case.path}"
            )
        if case.tier != TestTier.PUBLIC and "hidden" not in path.parts:
            raise InvalidTestPacket(
                f"non-public test must live under a hidden directory: {case.path}"
            )
    if regression_only:
        invalid = tiers - {TestTier.REGRESSION, TestTier.ADMISSION}
        if invalid:
            raise InvalidTestPacket(
                "regression packet may contain only regression/admission cases"
            )
    elif TestTier.PUBLIC not in tiers or TestTier.HIDDEN not in tiers:
        raise InvalidTestPacket(
            "packet must contain at least one public and one hidden test"
        )
    elif not any(
        case.evidence_role is EvidenceRole.POSITIVE_WITNESS for case in packet.cases
    ):
        raise InvalidTestPacket("packet must contain a positive_witness admission case")


def freeze_test_packet(
    bundle: Path, packet: TestPacket, *, admission_passed: bool
) -> TestPacket:
    provisional = replace(
        packet,
        status=TestStatus.ADMITTED,
        admission_passed=admission_passed,
        content_sha256="",
        metadata={**packet.metadata, "ontology": ontology_ref()},
    )
    digest = packet_content_hash(bundle, provisional)
    frozen = replace(provisional, content_sha256=digest)
    write_json(bundle / "test_packet.json", frozen.to_dict())
    return frozen


def packet_content_hash(bundle: Path, packet: TestPacket) -> str:
    digest = hashlib.sha256()
    normalized = packet.to_dict()
    normalized["content_sha256"] = ""
    digest.update(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        if path.name == "test_packet.json" or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(bundle).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def verify_frozen_packet(bundle: Path, packet: TestPacket) -> bool:
    return (
        bool(packet.content_sha256)
        and packet_content_hash(bundle, packet) == packet.content_sha256
    )


def run_test_cases(
    *,
    packet: TestPacket,
    bundle: Path,
    source: Path,
    subject: str,
    output_dir: Path,
    python: Path | None = None,
    tiers: set[TestTier] | None = None,
    probe_runner: Callable[[TestCaseSpec, Path, Path, str, Path], TestExecution]
    | None = None,
) -> list[TestExecution]:
    source = source.resolve()
    bundle = bundle.resolve()
    output_dir = output_dir.resolve()
    if not source.is_dir():
        raise TestSandboxUnavailable(
            f"test subject source is not a directory: {source}"
        )
    if not bundle.is_dir():
        raise TestSandboxUnavailable(f"test bundle is not a directory: {bundle}")
    for protected_root, label in ((source, "source"), (bundle, "bundle")):
        if output_dir == protected_root or protected_root in output_dir.parents:
            raise TestSandboxUnavailable(
                f"test output directory may not be inside the {label}: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if packet.content_sha256 and not verify_frozen_packet(bundle, packet):
        raise InvalidTestPacket(f"frozen TestPacket hash mismatch: {bundle}")
    source_hash = sha256_tree(source)
    bundle_hash = sha256_tree(bundle)
    selected_cases = tuple(
        case for case in packet.cases if tiers is None or case.tier in tiers
    )
    sandbox = (
        _sandbox_backend()
        if any(
            case.execution_mode is TestExecutionMode.DETERMINISTIC
            for case in selected_cases
        )
        else ""
    )
    executable = str((python or Path(sys.executable)).absolute())
    results: list[TestExecution] = []
    for case in selected_cases:
        if case.execution_mode is TestExecutionMode.MODEL_BACKED_PROBE:
            if probe_runner is None:
                raise TestSandboxUnavailable(
                    f"{case.case_id}: benchmark has no host-controlled agent probe runner"
                )
            result = probe_runner(case, bundle, source, subject, output_dir)
            _validate_probe_result(case, result, subject)
            results.append(result)
            continue
        results.append(
            _run_case(
                case=case,
                backend=sandbox,
                bundle=bundle,
                source=source,
                subject=subject,
                output_dir=output_dir,
                python=executable,
            )
        )
    if sha256_tree(source) != source_hash:
        raise TestSandboxUnavailable(
            f"generated tests mutated the real subject source: {source}"
        )
    if sha256_tree(bundle) != bundle_hash:
        raise TestSandboxUnavailable(
            f"generated tests mutated the real TestPacket bundle: {bundle}"
        )
    if packet.content_sha256 and not verify_frozen_packet(bundle, packet):
        raise InvalidTestPacket(f"generated tests mutated frozen TestPacket: {bundle}")
    write_json(output_dir / "results.json", [item.to_dict() for item in results])
    return results


def _validate_probe_result(
    case: TestCaseSpec,
    result: TestExecution,
    subject: str,
) -> None:
    if (
        result.case_id != case.case_id
        or result.tier is not case.tier
        or result.evidence_role is not case.evidence_role
        or result.execution_mode is not TestExecutionMode.MODEL_BACKED_PROBE
        or result.subject != subject
    ):
        raise TestSandboxUnavailable(
            f"{case.case_id}: host probe returned mismatched execution identity"
        )
    if result.model_calls < 0 or result.tokens < 0:
        raise TestSandboxUnavailable(
            f"{case.case_id}: host probe returned negative resource usage"
        )
    if result.model_calls > case.max_model_calls:
        raise TestSandboxUnavailable(
            f"{case.case_id}: host probe exceeded max_model_calls "
            f"({result.model_calls} > {case.max_model_calls})"
        )
    if result.tokens > case.max_tokens:
        raise TestSandboxUnavailable(
            f"{case.case_id}: host probe exceeded max_tokens "
            f"({result.tokens} > {case.max_tokens})"
        )


def _run_case(
    *,
    case: TestCaseSpec,
    backend: str,
    bundle: Path,
    source: Path,
    subject: str,
    output_dir: Path,
    python: str,
) -> TestExecution:
    with tempfile.TemporaryDirectory(prefix="traceunit-test-") as raw_tmp:
        sandbox_root = Path(raw_tmp)
        source_snapshot = sandbox_root / "source"
        bundle_snapshot = sandbox_root / "bundle"
        work_dir = sandbox_root / "work"
        home_dir = sandbox_root / "home"
        _copy_snapshot(source, source_snapshot)
        _copy_snapshot(bundle, bundle_snapshot)
        _make_tree_read_only(source_snapshot)
        _make_tree_read_only(bundle_snapshot)
        work_dir.mkdir()
        home_dir.mkdir()
        return _execute_isolated_case(
            case=case,
            backend=backend,
            bundle=bundle_snapshot,
            source=source_snapshot,
            subject=subject,
            output_dir=output_dir,
            python=python,
            work_dir=work_dir,
            home_dir=home_dir,
        )


def _sandbox_backend() -> str:
    requested = os.environ.get(_SANDBOX_MODE_ENV, "auto").strip().lower() or "auto"
    if requested == "copy":
        return "copy"
    if requested not in {"auto", "docker", "bwrap"}:
        raise TestSandboxUnavailable(
            f"unsupported {_SANDBOX_MODE_ENV}={requested!r}; "
            "expected auto, docker, bwrap, or copy"
        )
    if requested in {"auto", "docker"} and _docker_available():
        return "docker"
    if requested == "docker":
        raise TestSandboxUnavailable(
            "Docker test sandbox requested, but Docker or the configured image "
            "is unavailable"
        )
    if requested in {"auto", "bwrap"} and _bwrap_available():
        return "bwrap"
    if requested == "bwrap":
        raise TestSandboxUnavailable(
            "bubblewrap test sandbox requested, but unprivileged namespaces are unavailable"
        )
    raise TestSandboxUnavailable(
        "no generated-test sandbox is available: Docker image "
        f"{_docker_image()!r} was not usable and bubblewrap self-test failed. "
        f"Set {_SANDBOX_MODE_ENV}=copy only for explicit unit-test or dry-run use."
    )


def _docker_image() -> str:
    return os.environ.get("TRACEUNIT_TEST_DOCKER_IMAGE", "python:3.12-slim")


@functools.lru_cache(maxsize=1)
def _docker_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        completed = subprocess.run(
            [docker, "image", "inspect", _docker_image()],
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@functools.lru_cache(maxsize=1)
def _bwrap_available() -> bool:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return False
    try:
        completed = subprocess.run(
            [
                bwrap,
                "--die-with-parent",
                "--unshare-all",
                "--new-session",
                "--ro-bind",
                "/",
                "/",
                "--",
                "/bin/true",
            ],
            env={"PATH": os.environ.get("PATH", os.defpath)},
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _copy_snapshot(source: Path, destination: Path) -> None:
    try:
        shutil.copytree(
            source,
            destination,
            # Do not dereference a link that could point into sensitive host data.
            # The read-only pass below rejects every preserved link.
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", ".pytest_cache", "__pycache__", "*.pyc", "*.pyo"
            ),
        )
    except OSError as exc:
        raise TestSandboxUnavailable(
            f"could not create isolated snapshot of {source}: {exc}"
        ) from exc


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise TestSandboxUnavailable(f"snapshot retained a symlink: {path}")
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o555 if executable else 0o444)
    root.chmod(0o555)


def _execute_isolated_case(
    *,
    case: TestCaseSpec,
    backend: str,
    bundle: Path,
    source: Path,
    subject: str,
    output_dir: Path,
    python: str,
    work_dir: Path,
    home_dir: Path,
) -> TestExecution:
    stdout_path = output_dir / f"{case.case_id}.stdout.txt"
    stderr_path = output_dir / f"{case.case_id}.stderr.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend in {"docker", "bwrap"}:
        placeholders = {
            "source": "/traceunit/source",
            "test_bundle": "/traceunit/bundle",
            "subject": subject,
        }
        test_path = f"/traceunit/bundle/{case.path}"
    else:
        placeholders = {
            "source": str(source),
            "test_bundle": str(bundle),
            "subject": subject,
        }
        test_path = str(safe_relative_path(bundle, case.path))
    test_env = _minimal_test_environment(
        case=case,
        placeholders=placeholders,
        home="/traceunit/home" if backend in {"docker", "bwrap"} else str(home_dir),
        tmp="/tmp" if backend in {"docker", "bwrap"} else str(work_dir / "tmp"),
    )
    if backend == "docker":
        runtime_site = Path(sysconfig.get_paths()["purelib"]).resolve()
        if runtime_site.is_dir():
            test_env["PYTHONPATH"] = "/traceunit/runtime-site"
        inside = _test_argv("python", case, test_path)
        cidfile = work_dir.parent / "container.cid"
        argv = _docker_argv(
            source=source,
            bundle=bundle,
            work_dir=work_dir,
            home_dir=home_dir,
            environment=test_env,
            inside_argv=inside,
            cidfile=cidfile,
            runtime_site=runtime_site if runtime_site.is_dir() else None,
        )
        host_env = os.environ.copy()
        cwd = work_dir
    elif backend == "bwrap":
        inside = _test_argv(python, case, test_path)
        cidfile = None
        argv = _bwrap_argv(
            source=source,
            bundle=bundle,
            work_dir=work_dir,
            home_dir=home_dir,
            environment=test_env,
            inside_argv=inside,
            python=python,
        )
        host_env = {"PATH": os.environ.get("PATH", os.defpath)}
        cwd = work_dir
    else:
        tmp = work_dir / "tmp"
        tmp.mkdir()
        inside = _test_argv(python, case, test_path)
        cidfile = None
        argv = inside
        host_env = test_env
        cwd = work_dir

    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    error = ""
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=host_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=case.timeout_s,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        error = f"timed out after {case.timeout_s}s"
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        if cidfile is not None:
            _cleanup_docker_container(cidfile)
    except OSError as exc:
        if cidfile is not None:
            _cleanup_docker_container(cidfile)
        raise TestSandboxUnavailable(
            f"could not launch {backend} test sandbox: {type(exc).__name__}: {exc}"
        ) from exc

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    _raise_for_sandbox_failure(
        backend=backend,
        case=case,
        returncode=returncode,
        stderr=stderr,
    )
    return TestExecution(
        case_id=case.case_id,
        tier=case.tier,
        evidence_role=case.evidence_role,
        execution_mode=case.execution_mode,
        subject=subject,
        passed=returncode == 0 and not timed_out,
        returncode=returncode,
        duration_s=time.monotonic() - started,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        timed_out=timed_out,
        error=error,
    )


def _minimal_test_environment(
    *,
    case: TestCaseSpec,
    placeholders: Mapping[str, str],
    home: str,
    tmp: str,
) -> dict[str, str]:
    environment = {
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": tmp,
        "TRACEUNIT_SOURCE": placeholders["source"],
        "TRACEUNIT_SUBJECT": placeholders["subject"],
        "TRACEUNIT_TEST_BUNDLE": placeholders["test_bundle"],
    }
    for key, value in case.environment.items():
        upper = key.upper()
        if upper in _PROTECTED_ENV_KEYS or upper.startswith(_PROTECTED_ENV_PREFIXES):
            raise InvalidTestPacket(
                f"{case.case_id}: environment may not override protected key {key!r}"
            )
        environment[key] = expand_placeholders(value, placeholders)
    return environment


def _test_argv(python: str, case: TestCaseSpec, test_path: str) -> list[str]:
    if case.driver == "pytest":
        return [
            python,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            test_path,
            *case.arguments,
        ]
    return [python, test_path, *case.arguments]


def _docker_argv(
    *,
    source: Path,
    bundle: Path,
    work_dir: Path,
    home_dir: Path,
    environment: Mapping[str, str],
    inside_argv: list[str],
    cidfile: Path,
    runtime_site: Path | None,
) -> list[str]:
    docker = shutil.which("docker") or "docker"
    argv = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--cpus",
        "1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cidfile",
        str(cidfile),
        "--mount",
        f"type=bind,src={source},dst=/traceunit/source,readonly",
        "--mount",
        f"type=bind,src={bundle},dst=/traceunit/bundle,readonly",
        "--mount",
        f"type=bind,src={work_dir},dst=/traceunit/work",
        "--mount",
        f"type=bind,src={home_dir},dst=/traceunit/home",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=67108864",
        "--workdir",
        "/traceunit/work",
    ]
    if runtime_site is not None:
        argv.extend(
            [
                "--mount",
                f"type=bind,src={runtime_site},dst=/traceunit/runtime-site,readonly",
            ]
        )
    for key, value in sorted(environment.items()):
        argv.extend(["--env", f"{key}={value}"])
    argv.extend([_docker_image(), *inside_argv])
    return argv


def _bwrap_argv(
    *,
    source: Path,
    bundle: Path,
    work_dir: Path,
    home_dir: Path,
    environment: Mapping[str, str],
    inside_argv: list[str],
    python: str,
) -> list[str]:
    bwrap = shutil.which("bwrap") or "bwrap"
    argv = [
        bwrap,
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for path in _bwrap_runtime_paths(python):
        argv.extend(["--ro-bind", str(path), str(path)])
    argv.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/traceunit",
            "--dir",
            "/traceunit/source",
            "--dir",
            "/traceunit/bundle",
            "--dir",
            "/traceunit/work",
            "--dir",
            "/traceunit/home",
            "--ro-bind",
            str(source),
            "/traceunit/source",
            "--ro-bind",
            str(bundle),
            "/traceunit/bundle",
            "--bind",
            str(work_dir),
            "/traceunit/work",
            "--bind",
            str(home_dir),
            "/traceunit/home",
            "--chdir",
            "/traceunit/work",
            "--clearenv",
        ]
    )
    for key, value in sorted(environment.items()):
        argv.extend(["--setenv", key, value])
    argv.extend(["--", *inside_argv])
    return argv


def _bwrap_runtime_paths(python: str) -> list[Path]:
    paths = [
        Path(path) for path in ("/usr", "/bin", "/lib", "/lib64") if Path(path).exists()
    ]
    executable = Path(python)
    if executable.exists():
        venv_root = executable.parent.parent
        if not any(root == venv_root or root in venv_root.parents for root in paths):
            paths.append(venv_root)
    return paths


def _raise_for_sandbox_failure(
    *,
    backend: str,
    case: TestCaseSpec,
    returncode: int | None,
    stderr: str,
) -> None:
    lowered = stderr.lower()
    if backend == "docker" and returncode in {125, 126, 127}:
        raise TestSandboxUnavailable(
            f"Docker failed before executing test {case.case_id}: {stderr[-2000:]}"
        )
    if backend == "bwrap" and lowered.lstrip().startswith("bwrap:"):
        raise TestSandboxUnavailable(
            f"bubblewrap failed before executing test {case.case_id}: {stderr[-2000:]}"
        )
    if case.driver == "pytest" and "no module named pytest" in lowered:
        raise TestSandboxUnavailable(
            f"sandbox image {_docker_image()!r} lacks pytest for case {case.case_id}; "
            "use a dedicated TRACEUNIT_TEST_DOCKER_IMAGE with pytest installed"
        )


def _cleanup_docker_container(cidfile: Path) -> None:
    try:
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return
    docker = shutil.which("docker")
    if not docker or not container_id:
        return
    try:
        subprocess.run(
            [docker, "rm", "-f", container_id],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def admission_contract(
    packet: TestPacket,
    incumbent_results: Iterable[TestExecution],
) -> tuple[bool, list[str]]:
    by_id = {item.case_id: item for item in incumbent_results}
    reasons: list[str] = []
    for case in packet.cases:
        result = by_id.get(case.case_id)
        if result is None:
            reasons.append(f"{case.case_id}: missing result")
            continue
        if result.passed == case.expected_incumbent_pass:
            continue
        else:
            reasons.append(
                f"{case.case_id}: incumbent passed={result.passed}, "
                f"expected={case.expected_incumbent_pass}"
            )
    return not reasons, reasons


def paired_test_metrics(
    packet: TestPacket,
    incumbent: Iterable[TestExecution],
    candidate: Iterable[TestExecution],
) -> dict[str, float]:
    base = {item.case_id: item for item in incumbent}
    cand = {item.case_id: item for item in candidate}

    def gain_for(tier: TestTier) -> float:
        cases = [case for case in packet.cases if case.tier == tier]
        if not cases:
            return 0.0
        gains = 0
        for case in cases:
            b = base.get(case.case_id)
            c = cand.get(case.case_id)
            if b is not None and c is not None and not b.passed and c.passed:
                gains += 1
        return gains / len(cases)

    regression_cases = [
        case
        for case in packet.cases
        if case.tier in {TestTier.REGRESSION, TestTier.ADMISSION}
    ]
    regressions = 0
    for case in regression_cases:
        b = base.get(case.case_id)
        c = cand.get(case.case_id)
        if b is not None and c is not None and b.passed and not c.passed:
            regressions += 1
    return {
        "public_gain": gain_for(TestTier.PUBLIC),
        "hidden_gain": gain_for(TestTier.HIDDEN),
        "bridge_gain": gain_for(TestTier.BRIDGE),
        "regression_loss": (
            regressions / len(regression_cases) if regression_cases else 0.0
        ),
    }


def candidate_contract(
    packet: TestPacket,
    candidate_results: Iterable[TestExecution],
    *,
    tiers: frozenset[TestTier] | None = None,
) -> tuple[bool, list[str]]:
    """Fail closed unless selected frozen cases have their declared outcome."""

    by_id = {item.case_id: item for item in candidate_results}
    reasons: list[str] = []
    for case in (
        packet.cases
        if tiers is None
        else tuple(case for case in packet.cases if case.tier in tiers)
    ):
        result = by_id.get(case.case_id)
        if result is None:
            reasons.append(f"{case.case_id}: missing candidate result")
            continue
        if result.timed_out:
            reasons.append(f"{case.case_id}: candidate execution timed out")
            continue
        if result.error:
            reasons.append(f"{case.case_id}: candidate execution error")
            continue
        if result.passed != case.expected_candidate_pass:
            reasons.append(
                f"{case.case_id}: candidate passed={result.passed}, "
                f"expected={case.expected_candidate_pass}"
            )
    return not reasons, reasons
