from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


VARIANT_TABLE: Dict[str, set[str]] = {
    "HIS": {"HID", "HIE", "HIP"},
    "ASP": {"ASH"},
    "GLU": {"GLH"},
    "LYS": {"LYN"},
    "ARG": {"ARN"},
    "TYR": {"TYM"},
    "CYS": {"CYM", "CYX"},
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pdb_atomname(line: str) -> str:
    if len(line) < 16:
        return ""
    return line[12:16].strip().upper()


def pdb_resname(line: str) -> str:
    if len(line) < 20:
        return ""
    return line[17:20].strip().upper()


def pdb_chain(line: str) -> str:
    if len(line) < 22:
        return ""
    return (line[21:22] or "").strip()


def pdb_resseq(line: str) -> int | None:
    if len(line) < 26:
        return None
    raw = line[22:26].strip()
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def pdb_icode(line: str) -> str:
    if len(line) < 27:
        return ""
    return (line[26:27] or "").strip()


def pdb_element(line: str) -> str:
    if len(line) < 78:
        return ""
    return line[76:78].strip().upper()


def is_atom_record(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("ANISOU")


def is_ppm_dummy_atom_line(line: str) -> bool:
    if not is_atom_record(line):
        return False
    atom = pdb_atomname(line)
    resn = pdb_resname(line)
    elem = pdb_element(line)
    if "DUM" in atom:
        return True
    if "DUM" in resn:
        return True
    if elem == "DU":
        return True
    return False


def strip_ppm_dummy_atoms(pdb_in: Path, pdb_out: Path) -> None:
    dropped = 0
    written = 0
    with pdb_in.open("r", encoding="utf-8", errors="replace") as fin, pdb_out.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if is_ppm_dummy_atom_line(line):
                dropped += 1
                continue
            fout.write(line)
            written += 1
    print(f"[pkaraptor] Stripped PPM dummy atoms: dropped={dropped}, written={written} -> {pdb_out}")


def build_mapping(doc: Dict[str, Any], validate_allowed: bool) -> Dict[Tuple[str, int], str]:
    schema = str(doc.get("schema", "") or "")
    if schema and schema != "pkaraptor.resname_selection.v1":
        raise ValueError(f"Unsupported JSON schema: {schema}")

    selections = doc.get("selections", [])
    if not isinstance(selections, list):
        raise ValueError("JSON must contain 'selections' as a list")

    mapping: Dict[Tuple[str, int], str] = {}
    for i, item in enumerate(selections):
        if not isinstance(item, dict):
            continue

        chain = str(item.get("chain", "") or "").strip()
        resnum_raw = item.get("resnum", None)
        if resnum_raw is None:
            continue

        try:
            resnum = int(resnum_raw)
        except Exception:
            raise ValueError(f"Selection #{i}: invalid resnum={resnum_raw!r}")

        selected = str(item.get("selected_resname", "") or "").strip().upper()
        if not selected:
            continue

        if validate_allowed:
            allowed = item.get("allowed_resnames", [])
            if isinstance(allowed, list):
                allowed_set = {str(x).strip().upper() for x in allowed if str(x).strip()}
                if allowed_set and selected not in allowed_set:
                    residue_label = item.get("Residue", f"{chain}:{resnum}")
                    raise ValueError(
                        f"Selection '{residue_label}': selected_resname={selected} not in allowed_resnames={sorted(allowed_set)}"
                    )

        mapping[(chain, resnum)] = selected

    return mapping


def pdbfixer_prepare(
    pdb_in: Path,
    pdb_out: Path,
    remove_heterogens: bool,
    keep_water: bool,
) -> None:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except Exception as e:
        raise SystemExit(
            "Install PDBFixer:\n\n"
            "  python -m pip install pdbfixer\n\n"
            "or:\n\n"
            "  conda install -c conda-forge pdbfixer\n\n"
            f"Import error: {e}"
        )

    fixer = PDBFixer(filename=str(pdb_in))

    if remove_heterogens:
        fixer.removeHeterogens(keepWater=bool(keep_water))

    fixer.findMissingResidues()
    fixer.missingResidues = {}

    try:
        fixer.findDisulfideBonds()
    except Exception:
        pass

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with pdb_out.open("w", encoding="utf-8") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    print(
        f"[pkaraptor] PDBFixer prepared PDB: {pdb_out} (remove_heterogens={remove_heterogens}, keep_water={keep_water})"
    )


def _parse_residue_id_to_num(res_id: str) -> int | None:
    s = str(res_id or "").strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _canonical_and_variant(selected: str, current_name: str) -> Tuple[str, str | None]:
    sel = str(selected or "").strip().upper()
    cur = str(current_name or "").strip().upper()

    if sel in VARIANT_TABLE:
        return sel, None

    for canon, variants in VARIANT_TABLE.items():
        if sel in variants:
            return canon, sel

    if sel == cur:
        return cur, None
    return sel, None


def openmm_add_hydrogens_from_topology(
    pdb_in: Path,
    pdb_out: Path,
    forcefield_xmls: List[str],
    ph_for_openmm: float,
    mapping: Dict[Tuple[str, int], str],
) -> None:
    try:
        from openmm.app import ForceField, Modeller, PDBFile
    except Exception as e:
        raise SystemExit(
            "OpenMM is not available. Install it with:\n\n"
            "  python -m pip install openmm\n\n"
            f"Import error: {e}"
        )

    pdb = PDBFile(str(pdb_in))
    modeller = Modeller(pdb.topology, pdb.positions)

    try:
        modeller.deleteHydrogens()
        print("[pkaraptor] Deleted existing hydrogens before adding new ones.")
    except Exception:
        print("[pkaraptor] Note: could not delete hydrogens (maybe none present). Continuing.")

    ff = ForceField(*forcefield_xmls)

    residues = list(modeller.topology.residues())
    variants_list: List[str | None] = [None] * len(residues)

    renamed_to_canonical = 0
    applied_variants = Counter()
    applied_total = 0

    for idx, res in enumerate(residues):
        ch_id = str(res.chain.id or "").strip()
        rn = _parse_residue_id_to_num(res.id)
        if rn is None:
            continue

        key = (ch_id, int(rn))
        if key not in mapping:
            continue

        selected = mapping[key].strip().upper()
        canon, var = _canonical_and_variant(selected, res.name)

        if res.name.strip().upper() != canon:
            res.name = canon
            renamed_to_canonical += 1

        if var is not None:
            variants_list[idx] = var
            applied_variants[var] += 1

        applied_total += 1

    print(
        f"[pkaraptor] Topology selections applied={applied_total}, renamed_to_canonical={renamed_to_canonical}, "
        f"variants={dict(applied_variants)}"
    )

    modeller.addHydrogens(ff, pH=float(ph_for_openmm), variants=variants_list)

    with pdb_out.open("w", encoding="utf-8") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    print(f"[pkaraptor] Written PDB with hydrogens (OpenMM templates): {pdb_out}")


def apply_resname_labels_from_json(pdb_in: Path, pdb_out: Path, mapping: Dict[Tuple[str, int], str]) -> None:
    changed = 0
    hits = 0

    with pdb_in.open("r", encoding="utf-8", errors="replace") as fin, pdb_out.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not is_atom_record(line):
                fout.write(line)
                continue

            ch = pdb_chain(line)
            rn = pdb_resseq(line)
            if rn is None:
                fout.write(line)
                continue

            key = (ch, int(rn))
            if key not in mapping:
                fout.write(line)
                continue

            target = str(mapping[key]).strip().upper()
            if len(target) != 3:
                fout.write(line)
                continue

            hits += 1
            current = pdb_resname(line)
            if current != target:
                line = line[:17] + f"{target:>3s}" + line[20:]
                changed += 1

            fout.write(line)

    print(f"[pkaraptor] Applied JSON residue labels: matched_atoms={hits}, changed={changed} -> {pdb_out}")


def enforce_histidines_from_resname(
    pdb_in: Path,
    pdb_out: Path,
    report_csv: Path,
) -> None:
    """
    Enforce tautomer by residue name in the PDB itself (robust to renumbering/mapping mismatches):
      HID: delete HE2
      HIE: delete HD1
      HIP: keep both
    Also produces a report of which ring Hs existed and what was removed.
    """
    lines = pdb_in.read_text(encoding="utf-8", errors="replace").splitlines(True)

    grouped: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        if not is_atom_record(line):
            continue
        ch = pdb_chain(line)
        rn = pdb_resseq(line)
        if rn is None:
            continue
        ic = pdb_icode(line)
        grouped[(ch, int(rn), ic)].append(i)

    drop_idx: set[int] = set()
    report_rows: List[Dict[str, Any]] = []

    for (ch, rn, ic), idxs in grouped.items():
        resn = None
        atomset = set()
        for i in idxs:
            if not is_atom_record(lines[i]):
                continue
            resn = pdb_resname(lines[i])
            atomset.add(pdb_atomname(lines[i]))

        if resn not in {"HID", "HIE", "HIP", "HIS"}:
            continue

        had_hd1 = "HD1" in atomset
        had_he2 = "HE2" in atomset

        removed_hd1 = 0
        removed_he2 = 0

        if resn == "HID":
            for i in idxs:
                if pdb_atomname(lines[i]) == "HE2":
                    drop_idx.add(i)
                    removed_he2 += 1
        elif resn == "HIE":
            for i in idxs:
                if pdb_atomname(lines[i]) == "HD1":
                    drop_idx.add(i)
                    removed_hd1 += 1
        elif resn == "HIP":
            pass
        else:
            pass

        report_rows.append(
            {
                "chain": ch,
                "resnum": rn,
                "icode": ic,
                "resname": resn,
                "had_HD1": int(had_hd1),
                "had_HE2": int(had_he2),
                "removed_HD1": removed_hd1,
                "removed_HE2": removed_he2,
            }
        )

    out_lines = []
    dropped = 0
    for i, line in enumerate(lines):
        if i in drop_idx:
            dropped += 1
            continue
        out_lines.append(line)

    pdb_out.write_text("".join(out_lines), encoding="utf-8")

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chain",
                "resnum",
                "icode",
                "resname",
                "had_HD1",
                "had_HE2",
                "removed_HD1",
                "removed_HE2",
            ],
        )
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print(f"[pkaraptor] HIS enforcement (by resname) dropped_lines={dropped} -> {pdb_out}")
    print(f"[pkaraptor] HIS report written: {report_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pkaraptor selections (JSON) to a PDB and add hydrogens using OpenMM templates."
    )
    parser.add_argument("--json", required=True, help="Selections JSON exported from the dashboard")
    parser.add_argument("--pdb", help="Input PDB file (overrides JSON 'pdb' field if provided)")
    parser.add_argument("--out", required=True, help="Output PDB file (with hydrogens)")

    parser.add_argument("--strip-ppm", action="store_true", help="Remove PPM/OPM dummy atoms (DUM/DU)")
    parser.add_argument(
        "--no-validate-allowed",
        action="store_true",
        help="Do not validate selected_resname ∈ allowed_resnames",
    )

    parser.add_argument("--ph", type=float, default=7.0, help="pH used by OpenMM to place hydrogens")
    parser.add_argument(
        "--forcefield",
        action="append",
        default=[],
        help="OpenMM ForceField XML file(s). Repeatable. Default: amber14-all.xml",
    )

    parser.add_argument("--remove-heterogens", action="store_true", help="Remove heterogens before hydrogenation.")
    parser.add_argument("--keep-water", action="store_true", help="When removing heterogens, keep water molecules.")

    args = parser.parse_args()

    doc = load_json(Path(args.json))
    pdb_from_json = str(doc.get("pdb", "") or "").strip()
    pdb_in_path = Path(args.pdb) if args.pdb else Path(pdb_from_json)

    if not str(pdb_in_path).strip():
        raise SystemExit("No PDB provided. Use --pdb or ensure JSON has a 'pdb' field.")
    if not pdb_in_path.exists():
        raise SystemExit(f"Input PDB not found: {pdb_in_path}")

    mapping = build_mapping(doc, validate_allowed=(not args.no_validate_allowed))
    out_path = Path(args.out)

    ff_xmls = list(args.forcefield) if args.forcefield else ["amber14-all.xml"]

    report_csv = out_path.with_suffix("").with_name(out_path.stem + "_his_report.csv")

    with tempfile.TemporaryDirectory(prefix="pkaraptor_openmm_") as td:
        td_path = Path(td)

        working_pdb = td_path / "working.pdb"
        fixed_pdb = td_path / "fixed.pdb"
        openmm_out = td_path / "openmm_out.pdb"
        labeled_out = td_path / "labeled_out.pdb"
        enforced_out = td_path / "enforced_out.pdb"

        if args.strip_ppm:
            strip_ppm_dummy_atoms(pdb_in_path, working_pdb)
        else:
            working_pdb.write_text(pdb_in_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

        pdbfixer_prepare(
            pdb_in=working_pdb,
            pdb_out=fixed_pdb,
            remove_heterogens=bool(args.remove_heterogens),
            keep_water=bool(args.keep_water),
        )

        openmm_add_hydrogens_from_topology(
            pdb_in=fixed_pdb,
            pdb_out=openmm_out,
            forcefield_xmls=ff_xmls,
            ph_for_openmm=float(args.ph),
            mapping=mapping,
        )

        apply_resname_labels_from_json(
            pdb_in=openmm_out,
            pdb_out=labeled_out,
            mapping=mapping,
        )

        enforce_histidines_from_resname(
            pdb_in=labeled_out,
            pdb_out=enforced_out,
            report_csv=report_csv,
        )

        out_path.write_text(enforced_out.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

        selected_counts = Counter(mapping.values())
        if selected_counts:
            print(f"[pkaraptor] JSON selections summary: {dict(selected_counts)}")

        print(f"[pkaraptor] Written final PDB: {out_path}")


if __name__ == "__main__":
    main()