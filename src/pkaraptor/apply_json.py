from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pdb_chain_from_line(line: str) -> str:
    return line[21].strip() if len(line) > 21 else ""


def pdb_resnum_from_line(line: str) -> int | None:
    if len(line) < 26:
        return None
    raw = line[22:26].strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def pdb_inscode_from_line(line: str) -> str:
    return line[26].strip() if len(line) > 26 else ""


def set_resname_fixed_width(line: str, new_resname: str) -> str:
    new_resname = (str(new_resname).strip().upper() + "   ")[:3]
    if len(line) < 20:
        line = line.rstrip("\n").ljust(80) + "\n"
    return line[:17] + f"{new_resname:>3s}" + line[20:]


def build_mapping(doc: Dict[str, Any], validate_allowed: bool) -> Dict[Tuple[str, int, str], str]:
    schema = str(doc.get("schema", "") or "")
    if schema and schema != "pkaraptor.resname_selection.v1":
        raise ValueError(f"Unsupported JSON schema: {schema}")

    selections = doc.get("selections", [])
    if not isinstance(selections, list):
        raise ValueError("JSON must contain 'selections' as a list")

    mapping: Dict[Tuple[str, int, str], str] = {}

    for i, item in enumerate(selections):
        if not isinstance(item, dict):
            continue

        chain = str(item.get("chain", "") or "")
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

        mapping[(chain, resnum, "")] = selected

    return mapping


def apply_mapping_to_pdb(pdb_in: Path, pdb_out: Path, mapping: Dict[Tuple[str, int, str], str]) -> None:
    records_checked = 0
    records_changed = 0

    with pdb_in.open("r", encoding="utf-8", errors="replace") as fin, pdb_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = line[:6]

            if rec.startswith("ATOM") or rec.startswith("HETATM") or rec.startswith("ANISOU"):
                chain = pdb_chain_from_line(line)
                resnum = pdb_resnum_from_line(line)
                icode = pdb_inscode_from_line(line)

                if resnum is not None:
                    records_checked += 1

                    key_exact = (chain, resnum, icode)
                    key_plain = (chain, resnum, "")

                    if key_exact in mapping:
                        newname = mapping[key_exact]
                        new_line = set_resname_fixed_width(line, newname)
                        if new_line != line:
                            records_changed += 1
                        line = new_line
                    elif key_plain in mapping and icode:
                        newname = mapping[key_plain]
                        new_line = set_resname_fixed_width(line, newname)
                        if new_line != line:
                            records_changed += 1
                        line = new_line

            fout.write(line)

    print(f"[pkaraptor] Written PDB: {pdb_out} (records checked={records_checked}, records changed={records_changed})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pkaraptor dashboard residue-name selections (JSON) to a PDB by renaming residue names."
    )
    parser.add_argument("--json", required=True, help="Selections JSON exported from the dashboard")
    parser.add_argument("--pdb", help="Input PDB file (overrides JSON 'pdb' field if provided)")
    parser.add_argument("--out", required=True, help="Output protonated/renamed PDB file")
    parser.add_argument(
        "--no-validate-allowed",
        action="store_true",
        help="Disable validation that selected_resname is present in allowed_resnames",
    )

    args = parser.parse_args()

    json_path = Path(args.json)
    doc = load_json(json_path)

    pdb_from_json = str(doc.get("pdb", "") or "").strip()
    pdb_in_path = Path(args.pdb) if args.pdb else Path(pdb_from_json)

    if not pdb_in_path or str(pdb_in_path).strip() in {"", "."}:
        raise SystemExit("No PDB provided. Use --pdb or ensure JSON has a 'pdb' field.")

    if not pdb_in_path.exists():
        raise SystemExit(f"Input PDB not found: {pdb_in_path}")

    pdb_out_path = Path(args.out)

    mapping = build_mapping(doc, validate_allowed=(not args.no_validate_allowed))
    apply_mapping_to_pdb(pdb_in_path, pdb_out_path, mapping)


if __name__ == "__main__":
    main()