from __future__ import annotations

import argparse
import json
import shutil
import webbrowser
from pathlib import Path

import pandas as pd

from .dashboard import build_dashboard_html
from .analysis_cli import main as analysis_main
from .protonate import main as protonate_main
from .reporting import build_preparation_report, write_preparation_report


def _add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-propka", action="store_true")
    parser.add_argument("--pypka-csv")
    parser.add_argument("--deepka-csv")
    parser.add_argument("--opm-id", default="")
    parser.add_argument("--opm-pdb")
    parser.add_argument("--opm-residues")
    parser.add_argument("--opm-embedded", action="append")


def _metadata_path(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def run_workflow(args: argparse.Namespace) -> Path:
    pdb = Path(args.pdb).resolve()
    if not pdb.exists():
        raise SystemExit(f"Input PDB not found: {pdb}")
    project = Path(args.project or f"{pdb.stem}_pkaraptor").resolve()
    paths = {
        name: project / name
        for name in ("input", "analysis", "dashboard", "decisions", "structures", "report")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    copied_pdb = paths["input"] / pdb.name
    shutil.copy2(pdb, copied_pdb)

    dashboard_pdb = copied_pdb
    if args.opm_pdb:
        oriented_pdb = Path(args.opm_pdb).resolve()
        if not oriented_pdb.exists():
            raise SystemExit(f"Oriented PDB not found: {oriented_pdb}")
        oriented_name = oriented_pdb.name
        if oriented_name == pdb.name:
            oriented_name = f"oriented_{oriented_name}"
        dashboard_pdb = paths["input"] / oriented_name
        shutil.copy2(oriented_pdb, dashboard_pdb)

    analysis_csv = paths["analysis"] / "residues.csv"
    command = ["--pdb", str(pdb), "--ph", str(args.ph), "--out", str(analysis_csv)]
    for flag in ("no_propka",):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    for name in ("pypka_csv", "deepka_csv", "opm_id", "opm_pdb", "opm_residues"):
        value = getattr(args, name)
        if value is not None and value != "":
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for value in args.opm_embedded or []:
        command.extend(["--opm-embedded", value])
    analysis_main(command)

    frame = pd.read_csv(analysis_csv)
    dashboard_path = paths["dashboard"] / "pkaraptor.html"
    dashboard_path.write_text(
        build_dashboard_html(frame, str(dashboard_pdb), args.ph),
        encoding="utf-8",
    )

    metadata = {
        "schema": "pkaraptor.run_metadata.v1",
        "input_pdb": str(copied_pdb.relative_to(project)),
        "dashboard_pdb": str(dashboard_pdb.relative_to(project)),
        "analysis_ph": args.ph,
        "analysis_csv": str(analysis_csv.relative_to(project)),
        "dashboard": str(dashboard_path.relative_to(project)),
    }
    (paths["analysis"] / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[pkaraptor] Project written: {project}")
    print(f"[pkaraptor] Dashboard: {dashboard_path}")
    print(f"[pkaraptor] Export decisions into: {paths['decisions']}")
    print(f"[pkaraptor] Then run: pkaraptor finalize DECISIONS.json --project {project}")
    if not args.no_open:
        try:
            if not webbrowser.open(dashboard_path.as_uri(), new=2):
                print(f"[pkaraptor] Browser did not open; open manually: {dashboard_path}")
        except Exception as exc:
            print(f"[pkaraptor] Browser did not open ({exc}); open manually: {dashboard_path}")
    return project


def finalize_workflow(args: argparse.Namespace) -> Path:
    decisions = Path(args.decisions).resolve()
    if not decisions.exists():
        raise SystemExit(f"Decisions JSON not found: {decisions}")
    project = Path(args.project).resolve()
    metadata_path = project / "analysis" / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Project metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pdb = Path(args.pdb).resolve() if args.pdb else _metadata_path(project, metadata["input_pdb"])
    decisions_dir = project / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    archived_decisions = decisions_dir / "protonation_decisions.json"
    if decisions != archived_decisions:
        shutil.copy2(decisions, archived_decisions)
    final_pdb = project / "structures" / "protonated.pdb"
    command = [
        "--json", str(decisions), "--pdb", str(pdb), "--out", str(final_pdb), "--ph", str(args.ph or metadata["analysis_ph"]),
    ]
    if args.strip_ppm:
        command.append("--strip-ppm")
    if args.remove_heterogens:
        command.append("--remove-heterogens")
    if args.keep_water:
        command.append("--keep-water")
    for forcefield in args.forcefield or []:
        command.extend(["--forcefield", forcefield])
    protonate_main(command)

    report = build_preparation_report(
        archived_decisions,
        final_pdb,
        analysis_csv=_metadata_path(project, metadata["analysis_csv"]),
        input_pdb=pdb,
    )
    report_dir = project / "report"
    write_preparation_report(
        report,
        report_dir / "preparation_report.json",
        report_dir / "preparation_report.html",
    )
    print(f"[pkaraptor] Final structure: {final_pdb}")
    print(f"[pkaraptor] Preparation report: {report_dir / 'preparation_report.html'}")
    return final_pdb


def run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run analysis and build a structured pKaRaptor project.")
    parser.add_argument("pdb")
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--project")
    parser.add_argument("--no-open", action="store_true")
    _add_prediction_arguments(parser)
    run_workflow(parser.parse_args(argv))


def finalize_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply decisions, protonate, validate, and report.")
    parser.add_argument("decisions")
    parser.add_argument("--project", required=True)
    parser.add_argument("--pdb")
    parser.add_argument("--ph", type=float)
    parser.add_argument("--strip-ppm", action="store_true")
    parser.add_argument("--remove-heterogens", action="store_true")
    parser.add_argument("--keep-water", action="store_true")
    parser.add_argument("--forcefield", action="append", default=[])
    finalize_workflow(parser.parse_args(argv))
