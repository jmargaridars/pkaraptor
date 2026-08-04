from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FORMAL_CHARGES = {
    "ASP": -1,
    "ASH": 0,
    "GLU": -1,
    "GLH": 0,
    "HIS": 0,
    "HID": 0,
    "HIE": 0,
    "HIP": 1,
    "CYS": 0,
    "CYM": -1,
    "CYX": 0,
    "LYS": 1,
    "LYN": 0,
    "ARG": 1,
    "ARN": 0,
    "TYR": 0,
    "TYM": -1,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _pdb_residue_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
                continue
            try:
                keys.add((line[21].strip(), int(line[22:26].strip())))
            except ValueError:
                continue
    return keys


def build_preparation_report(
    decisions_path: Path,
    final_pdb: Path,
    *,
    analysis_csv: Path | None = None,
    input_pdb: Path | None = None,
) -> dict[str, Any]:
    decisions = _load_json(decisions_path)
    selections = [item for item in decisions.get("selections", []) if isinstance(item, dict)]
    pdb_keys = _pdb_residue_keys(final_pdb)

    selected_counts = Counter(
        str(item.get("selected_resname", "")).upper() for item in selections if item.get("selected_resname")
    )
    missing_mapping = []
    unsupported = []
    initial_charge = 0
    final_charge = 0
    changed = 0
    overrides = 0

    for item in selections:
        chain = str(item.get("chain", "") or "")
        try:
            resnum = int(item.get("resnum"))
        except (TypeError, ValueError):
            continue
        original = str(item.get("resname_original", "") or "").upper()
        selected = str(item.get("selected_resname", "") or "").upper()
        if (chain, resnum) not in pdb_keys:
            missing_mapping.append(str(item.get("Residue", f"{chain}:{resnum}")))
        if selected and selected not in FORMAL_CHARGES:
            unsupported.append(selected)
        initial_charge += FORMAL_CHARGES.get(original, 0)
        final_charge += FORMAL_CHARGES.get(selected, 0)
        changed += int(bool(selected) and selected != original)
        overrides += int(str(item.get("decision_source", "")) in {"user", "user_override"})

    analysis_summary: dict[str, Any] = {}
    if analysis_csv is not None and analysis_csv.exists():
        frame = pd.read_csv(analysis_csv)
        analysis_summary = {
            "titratable_residues": int(len(frame)),
            "membrane_associated_residues": int(
                frame.get("In_membrane_slab", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()
            ),
        }

    validation = {
        "final_pdb_exists": final_pdb.exists() and final_pdb.stat().st_size > 0,
        "residue_mapping": "passed" if not missing_mapping else "warning",
        "missing_selection_targets": missing_mapping,
        "unsupported_selected_resnames": sorted(set(unsupported)),
    }
    return {
        "schema": "pkaraptor.preparation_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "structure": str(input_pdb or decisions.get("pdb", "")),
            "analysis_ph": decisions.get("ph"),
            "decisions": str(decisions_path),
        },
        "assessment": analysis_summary,
        "assignments": {
            "total_selections": len(selections),
            "assignments_changed": changed,
            "user_overrides": overrides,
            "selected_resname_counts": dict(sorted(selected_counts.items())),
        },
        "charge": {
            "initial_estimated_sidechain_charge": initial_charge,
            "final_estimated_sidechain_charge": final_charge,
            "change": final_charge - initial_charge,
            "note": "Side-chain formal-charge estimate; termini, ligands, ions, and cofactors are excluded.",
        },
        "output": {"final_structure": str(final_pdb)},
        "validation": validation,
    }


def write_preparation_report(report: dict[str, Any], json_path: Path, html_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def row(label: str, value: object) -> str:
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>"

    assessment = report.get("assessment", {})
    assignments = report.get("assignments", {})
    charge = report.get("charge", {})
    validation = report.get("validation", {})
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>pKaRaptor preparation report</title>
<style>body{{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;color:#172033}}
h1,h2{{color:#183b66}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}
th,td{{border:1px solid #d7deea;padding:9px;text-align:left}}th{{width:42%;background:#f4f7fb}}</style>
</head><body><h1>pKaRaptor Preparation Report</h1>
<h2>Input</h2><table>{row("Structure", report["input"]["structure"])}{row("Analysis pH", report["input"]["analysis_ph"])}</table>
<h2>Assessment</h2><table>{''.join(row(k.replace('_', ' ').title(), v) for k, v in assessment.items())}</table>
<h2>Assignments</h2><table>{''.join(row(k.replace('_', ' ').title(), v) for k, v in assignments.items())}</table>
<h2>Charge</h2><table>{''.join(row(k.replace('_', ' ').title(), v) for k, v in charge.items())}</table>
<h2>Validation</h2><table>{''.join(row(k.replace('_', ' ').title(), v) for k, v in validation.items())}</table>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")
