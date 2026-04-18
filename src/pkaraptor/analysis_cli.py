from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .env_analysis import compute_environment_features
from .pka_methods import attach_pka


def compute_protonation_fraction(pka: float, ph: float) -> float:
    if np.isnan(pka):
        return np.nan
    try:
        return 1.0 / (1.0 + 10.0 ** (ph - pka))
    except OverflowError:
        return 0.0 if ph > pka else 1.0


def classify_from_fraction(frac: float) -> str:
    if np.isnan(frac):
        return "Ambiguous"
    if frac >= 0.5:
        return "Protonated"
    if frac <= 0.1:
        return "Deprotonated"
    return "Ambiguous"


def parse_opm_residues(s: str) -> Set[int]:
    out: Set[int] = set()
    if not s:
        return out
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(tok))
            except Exception:
                continue
    return out


def parse_chain_embedded(arg: str) -> tuple[str, Set[int]]:
    s = str(arg).strip()
    if not s or ":" not in s:
        raise ValueError("Invalid --opm-embedded format. Expected like 'A:48-66,69,426'.")
    chain, rest = s.split(":", 1)
    chain = chain.strip()
    if not chain:
        raise ValueError("Invalid --opm-embedded: missing chain label before ':'")
    return chain, parse_opm_residues(rest)


def parse_opm_embedded_args(args_list: Optional[List[str]]) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    if not args_list:
        return out
    for item in args_list:
        chain, rs = parse_chain_embedded(item)
        out.setdefault(chain, set()).update(rs)
    return out


def _drop_duplicate_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    keep: list[bool] = []
    for c in df.columns:
        key = str(c).casefold()
        if key in seen:
            keep.append(False)
        else:
            seen.add(key)
            keep.append(True)
    return df.loc[:, keep].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Generate protonation_environment_analysis.csv from a PDB using pKa sources and environment descriptors."
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--ph", type=float, default=7.0, help="pH at which to compute protonation fractions")
    parser.add_argument("--out", default="protonation_environment_analysis.csv", help="Output CSV file name")

    parser.add_argument("--no-propka", action="store_true", help="Disable local PROPKA execution")
    parser.add_argument("--pypka-csv", help="PyPka server CSV (semicolon-separated; Chain;Type;Number;pKa)")
    parser.add_argument("--deepka-csv", help='DeepKa server CSV (columns like "Chain", "Res ID", "Res Name", "Predict pKa")')

    parser.add_argument("--opm-id", default="", help="Optional OPM identifier for downstream visualization.")
    parser.add_argument("--opm-zmin", type=float, default=np.nan, help="Lower Z boundary of membrane slab in Angstrom.")
    parser.add_argument("--opm-zmax", type=float, default=np.nan, help="Upper Z boundary of membrane slab in Angstrom.")
    parser.add_argument("--opm-pdb", help="Oriented PDB from PPM/OPM (used only for environment feature calculation).")
    parser.add_argument("--opm-residues", help='(Legacy) residue ranges without chain (e.g. "48-66,69,426"). Applies to all chains.')
    parser.add_argument(
        "--opm-embedded",
        action="append",
        help='Chain-aware embedded residues. Repeatable. Example: --opm-embedded "A:48-66,69,426" --opm-embedded "B:48-66,69,426,461"',
    )

    args = parser.parse_args()

    pdb_path = str(args.pdb)
    ph = float(args.ph)

    opm_id = str(args.opm_id).strip()
    opm_zmin = float(args.opm_zmin) if not np.isnan(args.opm_zmin) else None
    opm_zmax = float(args.opm_zmax) if not np.isnan(args.opm_zmax) else None

    chain_embedded = parse_opm_embedded_args(args.opm_embedded)

    legacy_res_set: Optional[Set[int]] = None
    if args.opm_residues:
        legacy_res_set = parse_opm_residues(args.opm_residues)

    pdb_for_env = str(args.opm_pdb) if args.opm_pdb else pdb_path

    env_df = compute_environment_features(
        pdb_for_env,
        opm_id=opm_id if opm_id else None,
        opm_zmin_A=opm_zmin,
        opm_zmax_A=opm_zmax,
    )
    if env_df.empty:
        raise SystemExit("No titratable residues found or environment analysis failed.")

    if chain_embedded and ("resnum" in env_df.columns) and ("chain" in env_df.columns):

        def _flag_row(r: pd.Series) -> bool:
            try:
                ch = str(r.get("chain", "") or "")
                rn = int(r.get("resnum"))
            except Exception:
                return False
            return rn in chain_embedded.get(ch, set())

        env_df["In_membrane_slab"] = env_df.apply(_flag_row, axis=1)
        if not opm_id:
            opm_id = "PPM3_manual"
            env_df["OPM_id"] = opm_id

    elif legacy_res_set is not None and "resnum" in env_df.columns:
        env_df["In_membrane_slab"] = env_df["resnum"].apply(
            lambda x: (int(x) in legacy_res_set) if (x is not None and not pd.isna(x)) else False
        )
        if not opm_id:
            opm_id = "PPM3_manual"
            env_df["OPM_id"] = opm_id

    methods: List[str] = []
    external_csv: Dict[str, str] = {}

    if not args.no_propka:
        methods.append("propka")
    if args.pypka_csv:
        methods.append("pypka_csv")
        external_csv["pypka_csv"] = str(args.pypka_csv)
    if args.deepka_csv:
        methods.append("deepka")
        external_csv["deepka"] = str(args.deepka_csv)

    if not methods:
        raise SystemExit("No pKa sources selected. Use PROPKA (default) or provide --pypka-csv / --deepka-csv.")

    df = attach_pka(
        env_df,
        pdb_path,
        methods=tuple(methods),
        ffinput=None,
        external_csv=external_csv if external_csv else None,
    )

    fractions: List[float] = []
    states: List[str] = []
    for _, row in df.iterrows():
        try:
            pka = float(row.get("Effective_pKa", np.nan))
        except Exception:
            pka = np.nan
        frac = compute_protonation_fraction(pka, ph)
        fractions.append(frac)
        states.append(classify_from_fraction(frac))

    df["Protonation_fraction"] = fractions
    df["Protonation_state"] = states

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _drop_duplicate_columns_case_insensitive(df)

    if "pypka_pKa" not in df.columns and "pypka_csv_pKa" in df.columns:
        df["pypka_pKa"] = df["pypka_csv_pKa"]

    if "propka_pKa" not in df.columns:
        for c in list(df.columns):
            if str(c).casefold() == "propka_pka":
                df["propka_pKa"] = df[c]
                if c != "propka_pKa":
                    df = df.drop(columns=[c])
                break

    clean_cols = [
        "Residue",
        "chain",
        "resname",
        "resnum",
        "Disulfide_bridge",
        "Disulfide_partner",
        "propka_pKa",
        "propka_pKa_raw",
        "pypka_pKa",
        "deepka_pKa",
        "Effective_pKa",
        "Average_pKa",
        "SASA_frac",
        "SASA",
        "Exposure",
        "Neighbors_within_5A",
        "HBond_shell_donors_count",
        "HBond_shell_acceptors_count",
        "HBond_shell_donors",
        "HBond_shell_acceptors",
        "z_CA_A",
        "In_membrane_slab",
        "OPM_id",
        "OPM_zmin_A",
        "OPM_zmax_A",
        "Protonation_fraction",
        "Protonation_state",
    ]
    cols_present = [c for c in clean_cols if c in df.columns]
    df_out = df[cols_present].copy()

    df_out = df_out.loc[:, ~df_out.columns.duplicated()].copy()
    df_out = _drop_duplicate_columns_case_insensitive(df_out)

    out_path = Path(args.out)
    df_out.to_csv(out_path, index=False)
    print(f"[pkaraptor] Written CSV: {out_path}")


if __name__ == "__main__":
    main()