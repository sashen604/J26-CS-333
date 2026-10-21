"""Pre-flight check for the PP1 demo: run this before presenting.

Run from the repo root:  .venv\\Scripts\\python.exe p4_robustness/preflight.py   (Windows)
                         .venv/bin/python p4_robustness/preflight.py          (macOS/Linux)

Prints PASS or FAIL for each check, with a one-line fix for every FAIL:
1. The repo's .venv Python is running, and every package in requirements.txt imports
   at its pinned version.
2. The three mirror Parquet files exist in data/ciciot2023_hf/ with the expected sizes.
3. The saved proxy model exists, loads, and uses exactly the current feature list
   (data_prep.feature_columns) with no excluded (window-size leakage) feature.
4. Every results file demo.py reads exists in results/ and can be parsed.
5. results/drift_scenario_c.png exists (Part 2 opens it).
"READY FOR DEMO" is printed only if every check passed (exit status 0, else 1).

Only the standard library is imported at module level, so this runs (and reports
missing packages) even on the wrong Python.
"""
import importlib
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path

P4_DIR = Path(__file__).resolve().parent
REPO_ROOT = P4_DIR.parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = P4_DIR / "requirements.txt"
DATA_DIR = REPO_ROOT / "data" / "ciciot2023_hf"
RESULTS_DIR = P4_DIR / "results"

# Byte sizes of the Hugging Face files (X-Linked-Size of lacg030175/CIC-IoT-2023, random_3way).
EXPECTED_SIZES = {"train": 53_328_937, "test": 6_909_515, "validation": 6_903_808}

# Results files demo.py reads, and the plot Part 2 opens.
DEMO_RESULTS = ("drift_results.md", "xai_results.md")
DEMO_PLOT = "drift_scenario_c.png"

# Distribution name -> import name, where they differ.
IMPORT_NAMES = {
    "fonttools": "fontTools",
    "imageio": "imageio",
    "lazy-loader": "lazy_loader",
    "pillow": "PIL",
    "pygments": "pygments",
    "python-dateutil": "dateutil",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
}

if os.name == "nt":
    VENV_PYTHON = r".venv\Scripts\python.exe"
else:
    VENV_PYTHON = ".venv/bin/python"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""


def run_cmd(script):
    """The exact command, run from the repo root, for a p4_robustness script."""
    return f"{VENV_PYTHON} p4_robustness/{script}"


# ---------------------------------------------------------------------------
# 1. Python and packages
# ---------------------------------------------------------------------------
def check_venv(prefix=sys.prefix, venv_dir=VENV_DIR):
    ok = Path(prefix).resolve() == Path(venv_dir).resolve()
    return Check("Python is the repo's .venv", ok, f"running {sys.executable}",
                 "" if ok else f"run this script with {VENV_PYTHON} (create the venv first: python -m venv .venv)")


def read_requirements(path=REQUIREMENTS):
    """[(distribution name, pinned version)] from 'name==version' lines."""
    reqs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            reqs.append((name.strip(), version.strip()))
    return reqs


def import_name(dist):
    return IMPORT_NAMES.get(dist.lower(), dist.lower().replace("-", "_"))


def check_packages(requirements=REQUIREMENTS):
    """One Check covering every requirement: it must import and match its pinned version."""
    problems = []
    reqs = read_requirements(requirements)
    for dist, pinned in reqs:
        try:
            importlib.import_module(import_name(dist))
        except Exception as exc:  # noqa: BLE001 — any import failure is a FAIL
            problems.append(f"{dist} does not import ({type(exc).__name__})")
            continue
        try:
            installed = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{dist} has no installed metadata")
            continue
        if pinned and installed != pinned:
            problems.append(f"{dist} {installed} installed, {pinned} pinned")
    ok = not problems
    shown = "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
    detail = f"{len(reqs)} packages import at their pinned versions" if ok else shown
    return Check("requirements.txt packages", ok, detail,
                 "" if ok else f"{VENV_PYTHON} -m pip install -r p4_robustness/requirements.txt")


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
def check_data_files(data_dir=DATA_DIR, expected=EXPECTED_SIZES):
    checks = []
    for split, size in expected.items():
        path = Path(data_dir) / f"{split}.parquet"
        name = f"data: {split}.parquet"
        if not path.exists():
            checks.append(Check(name, False, "missing", run_cmd("data_prep.py")))
        elif path.stat().st_size != size:
            checks.append(Check(name, False, f"{path.stat().st_size:,} bytes, expected {size:,}",
                                f"delete {repo_relative(path)}, then run {run_cmd('data_prep.py')}"))
        else:
            checks.append(Check(name, True, f"{size:,} bytes"))
    return checks


# ---------------------------------------------------------------------------
# 3. Proxy model
# ---------------------------------------------------------------------------
def check_model(model_path=None, schema_path=None, data_dir=DATA_DIR):
    """The saved model loads, and its schema and booster both use exactly the current feature list."""
    name = "proxy model"
    fix = run_cmd("proxy_model.py")
    try:
        sys.path.insert(0, str(P4_DIR))
        import data_prep
        import proxy_model
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"cannot import the model code ({type(exc).__name__}: {exc})",
                     f"{VENV_PYTHON} -m pip install -r p4_robustness/requirements.txt")
    model_path = Path(model_path or proxy_model.MODEL_PATH)
    schema_path = Path(schema_path or proxy_model.SCHEMA_PATH)
    missing = [p.name for p in (model_path, schema_path) if not p.exists()]
    if missing:
        return Check(name, False, f"missing {', '.join(missing)}", fix)
    try:
        model = proxy_model.ProxyFusionModel.load(model_path, schema_path)
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"does not load ({type(exc).__name__}: {exc})", fix)
    test_path = data_prep.split_path("test", data_dir)
    if not test_path.exists():
        return Check(name, False, "cannot compare features: test.parquet is missing", run_cmd("data_prep.py"))
    import pandas as pd
    columns = pq.read_schema(test_path).names
    expected = data_prep.feature_columns(pd.DataFrame(columns=columns))
    excluded = [f for f in model.feature_names if f in data_prep.EXCLUDED_FEATURES]
    booster_names = list(model.model.get_booster().feature_names or [])
    if excluded:
        return Check(name, False, f"uses excluded feature(s): {', '.join(excluded)}", fix)
    if model.feature_names != expected:
        extra = sorted(set(model.feature_names) - set(expected))
        absent = sorted(set(expected) - set(model.feature_names))
        return Check(name, False, f"feature list differs from current (extra {extra}, missing {absent}, "
                     "or a different order)", fix)
    if booster_names != model.feature_names:
        return Check(name, False, "the saved booster's features differ from its schema file", fix)
    return Check(name, True, f"loads; {len(expected)} features, matching the current list, none excluded")


# ---------------------------------------------------------------------------
# 4-5. Results the demo reads
# ---------------------------------------------------------------------------
def repo_relative(path):
    """Path relative to the repo root (as typed in a command), or the absolute path if outside it."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def restore_cmd(path):
    return f"git checkout -- {repo_relative(path)}  (restores the committed file)"


def check_results(results_dir=RESULTS_DIR):
    """Each file exists and demo.py's parsers find what they need in it."""
    checks = []
    parsers = {}
    try:
        import demo
        parsers = {
            "drift_results.md": lambda p: (demo.committed_scenario_c(p), demo.committed_calibration(p)),
            "xai_results.md": lambda p: (demo.committed_shap_top(p), demo.committed_repeatability(p)),
        }
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("demo.py imports", False, f"{type(exc).__name__}: {exc}",
                            f"{VENV_PYTHON} -m pip install -r p4_robustness/requirements.txt"))
    for fname in DEMO_RESULTS:
        path = Path(results_dir) / fname
        name = f"results: {fname}"
        if not path.exists():
            checks.append(Check(name, False, "missing", restore_cmd(path)))
            continue
        if fname in parsers:
            try:
                parsed = parsers[fname](path)
                if not all(parsed):
                    raise ValueError("a table the demo reads is empty")
            except Exception as exc:  # noqa: BLE001
                checks.append(Check(name, False, f"exists but demo.py cannot read it ({exc})", restore_cmd(path)))
                continue
        checks.append(Check(name, True, "exists and parses" if parsers
                            else "exists (not parsed, because demo.py did not import)"))
    return checks


def check_plot(results_dir=RESULTS_DIR):
    path = Path(results_dir) / DEMO_PLOT
    ok = path.exists() and path.stat().st_size > 0
    return Check(f"results: {DEMO_PLOT}", ok, f"{path.stat().st_size:,} bytes" if ok else "missing or empty",
                 "" if ok else restore_cmd(path))


def run_all():
    checks = [check_venv(), check_packages()]
    checks += check_data_files()
    checks.append(check_model())
    checks += check_results()
    checks.append(check_plot())
    return checks


def print_checks(checks):
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name:{width}}  {c.detail}")
        if not c.ok:
            print(f"       {'':{width}}  fix: {c.fix}")


def main():
    print("Pillar 4 demo pre-flight check\n")
    checks = run_all()
    print_checks(checks)
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\nNOT READY: {len(failed)} of {len(checks)} checks failed. Apply the fixes above and rerun.")
        return 1
    print(f"\nAll {len(checks)} checks passed.\nREADY FOR DEMO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
