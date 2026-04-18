from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

STANDARD_TITRATABLE = {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG"}

try:
    from propka import run as propka_run  # type: ignore
except ImportError:
    propka_run = None


ChainKey = str
PkaKey = Tuple[ChainKey, str, int]  # (chain, resname, resnum)


def _norm_chain(chain: str | None) -> str:
    if chain is None:
        return ""
    return str(chain).strip()


def _parse_propka_file(text: str) -> Dict[Tuple[str, int], float]:
    values: Dict[Tuple[str, int], list] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) < 3:
            continue

        resname = tokens[0].upper()
        if resname not in STANDARD_TITRATABLE:
            continue

        try:
            resnum = int(tokens[1])
        except ValueError:
            continue

        numeric = []
        for tok in tokens[2:]:
            try:
                numeric.append(float(tok))
            except ValueError:
                continue

        if not numeric:
            continue

        pka = numeric[0]
        key = (resname, resnum)
        values.setdefault(key, []).append(pka)

    return {k: float(np.mean(v)) for k, v in values.items()}


def run_propka(pdb_path: str) -> Dict[PkaKey, float]:
    if propka_run is None:
        raise RuntimeError(
            "PROPKA is not available. Install it with `pip install propka` "
            "or use the 'propka' extra (pip install .[propka])."
        )

    propka_run.single(pdb_path, optargs=(), write_pka=True)

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    candidates = [f"{base}.pka", f"{base.lower()}.pka", f"{base.upper()}.pka"]
    pka_file = None
    for c in candidates:
        if os.path.exists(c):
            pka_file = c
            break

    if pka_file is None:
        raise FileNotFoundError(
            f"could not find propka .pka file for {pdb_path}. looked for: "
            + ", ".join(candidates)
        )

    text = open(pka_file, "r", encoding="utf-8").read()
    raw = _parse_propka_file(text)

    out: Dict[PkaKey, float] = {}
    for (rn, num), val in raw.items():
        out[("*", rn, int(num))] = float(val)
    return out


def parse_pypka_server_csv(path: str) -> Dict[PkaKey, float]:
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["-", "–", "NA", "NaN", "nan", ""],
        keep_default_na=True,
    )

    required = {"Type", "Number", "pKa"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Not a recognized PyPka server CSV. Expected columns like: Chain;Type;Number;pKa."
        )

    out: Dict[PkaKey, float] = {}
    for _, r in df.iterrows():
        chain = _norm_chain(r.get("Chain", ""))
        rn = str(r["Type"]).strip().upper()
        if rn not in STANDARD_TITRATABLE and rn not in {"NTR", "CTR"}:
            continue

        try:
            num = int(r["Number"])
        except Exception:
            continue

        pka = pd.to_numeric(r.get("pKa", np.nan), errors="coerce")
        if pd.isna(pka):
            continue

        out[(chain if chain else "*", rn, num)] = float(pka)

    return out


def parse_deepka_csv(path: str) -> Dict[PkaKey, float]:
    df = pd.read_csv(path)

    required = {"Res ID", "Res Name", "Predict pKa"}
    if not required.issubset(df.columns):
        raise ValueError(
            'Not a recognized DeepKa CSV. Expected columns including "Res ID", "Res Name", "Predict pKa".'
        )

    out: Dict[PkaKey, float] = {}
    for _, r in df.iterrows():
        chain = _norm_chain(r.get("Chain", ""))
        rn = str(r["Res Name"]).strip().upper()
        if rn not in STANDARD_TITRATABLE:
            continue

        try:
            num = int(r["Res ID"])
        except Exception:
            continue

        try:
            pka = float(r["Predict pKa"])
        except Exception:
            continue

        out[(chain if chain else "*", rn, num)] = pka

    return out


def _lookup_pka(d: Dict[PkaKey, float], chain: str, rn: str, num: int) -> float:
    chain = _norm_chain(chain)
    key_exact = (chain, rn, num)
    if key_exact in d:
        return float(d[key_exact])

    key_wild = ("*", rn, num)
    if key_wild in d:
        return float(d[key_wild])

    matches = [v for (ch, rname, rnum), v in d.items() if rname == rn and rnum == num]
    if len(matches) == 1:
        return float(matches[0])
    return float("nan")


def _lookup_pka_by_resnum_only(
    d: Dict[PkaKey, float],
    chain: str,
    num: int,
    *,
    exclude_resnames: set[str] | None = None,
) -> float:
    chain = _norm_chain(chain)
    excl = exclude_resnames or set()

    hits = [v for (ch, rname, rnum), v in d.items() if ch == chain and rnum == num and rname not in excl]
    if len(hits) == 1:
        return float(hits[0])

    hits = [v for (ch, rname, rnum), v in d.items() if ch == "*" and rnum == num and rname not in excl]
    if len(hits) == 1:
        return float(hits[0])

    hits = [v for (ch, rname, rnum), v in d.items() if rnum == num and rname not in excl]
    if len(hits) == 1:
        return float(hits[0])

    return float("nan")


def _is_propka_disulfide_sentinel(v: float) -> bool:
    if not np.isfinite(v):
        return False
    return abs(float(v) - 99.99) < 1e-6 or float(v) >= 90.0


def _sanitize_pka(v: float, *, treat_propka_ss: bool = False) -> float:
    if pd.isna(v):
        return float("nan")
    vv = float(v)

    if treat_propka_ss and _is_propka_disulfide_sentinel(vv):
        return float("nan")

    if vv <= 0.0 or vv >= 14.0:
        return float("nan")

    return vv


def _drop_duplicate_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    keep_mask: list[bool] = []
    for c in df.columns:
        key = str(c).casefold()
        if key in seen:
            keep_mask.append(False)
        else:
            seen.add(key)
            keep_mask.append(True)
    return df.loc[:, keep_mask].copy()


def attach_pka(
    env_df: pd.DataFrame,
    pdb_path: str,
    methods: Iterable[str] = ("propka",),
    ffinput: str | None = None,
    external_csv: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    df = env_df.copy()

    methods_norm: list[str] = []
    for m in methods:
        mm = str(m).strip().lower()
        if mm == "pypka_csv":
            mm = "pypka"
        methods_norm.append(mm)
    methods_t = tuple(methods_norm)

    if "chain" not in df.columns or "resname" not in df.columns or "resnum" not in df.columns:
        chains: list[str] = []
        resnames: list[str] = []
        resnums: list[int | None] = []
        for lab in df["Residue"]:
            s = str(lab).strip()
            chain = ""
            if ":" in s:
                chain, s2 = s.split(":", 1)
                s = s2.strip()
            m = re.match(r"([A-Za-z]+)\s*([0-9]+)", s)
            if not m:
                chains.append(chain)
                resnames.append("")
                resnums.append(None)
            else:
                chains.append(chain)
                resnames.append(m.group(1).upper())
                resnums.append(int(m.group(2)))
        df["chain"] = chains
        df["resname"] = resnames
        df["resnum"] = resnums

    method_dicts: dict[str, Dict[PkaKey, float]] = {}

    if "propka" in methods_t:
        try:
            method_dicts["propka"] = run_propka(pdb_path)
        except Exception as exc:
            print(f"[pkaraptor] WARNING: PROPKA failed: {exc}")

    if external_csv:
        pypka_path = external_csv.get("pypka") or external_csv.get("pypka_csv")
        if "pypka" in methods_t and pypka_path:
            try:
                method_dicts["pypka"] = parse_pypka_server_csv(pypka_path)
            except Exception as exc:
                print(f"[pkaraptor] WARNING: PyPka CSV import failed: {exc}")

        if "deepka" in methods_t and "deepka" in external_csv:
            try:
                method_dicts["deepka"] = parse_deepka_csv(external_csv["deepka"])
            except Exception as exc:
                print(f"[pkaraptor] WARNING: DeepKa CSV import failed: {exc}")

    # Keep track of PROPKA SS sentinels, but DO NOT DROP these residues (dashboard wants them).
    propka_ss_mask = pd.Series(False, index=df.index)

    for name, d in method_dicts.items():
        col = f"{name}_pKa"

        vals: list[float] = []
        raw_vals: list[float] = []
        clean_vals: list[float] = []
        hits = 0

        for i, (ch, rn, num) in enumerate(zip(df["chain"], df["resname"], df["resnum"])):
            if rn is None or num is None or str(rn).strip() == "" or pd.isna(num):
                vals.append(np.nan)
                if name == "propka":
                    raw_vals.append(np.nan)
                    clean_vals.append(np.nan)
                continue

            chs = str(ch)
            rns = str(rn).upper()
            numi = int(num)

            v = _lookup_pka(d, chs, rns, numi)

            if name == "pypka" and np.isnan(v):
                v = _lookup_pka_by_resnum_only(d, chs, numi, exclude_resnames={"NTR", "CTR"})

            if not np.isnan(v):
                hits += 1

            if name == "propka":
                if rns == "CYS" and (not np.isnan(v)) and _is_propka_disulfide_sentinel(float(v)):
                    propka_ss_mask.iloc[i] = True

                raw_vals.append(v)

                v_clean = _sanitize_pka(v, treat_propka_ss=True)
                clean_vals.append(v_clean)
                vals.append(v_clean)
            else:
                vals.append(_sanitize_pka(v, treat_propka_ss=False))

        df[col] = vals
        if name == "propka":
            df["propka_pKa_raw"] = raw_vals
            df["propka_pKa_clean"] = clean_vals

        if name == "pypka":
            df["pypka_csv_pKa"] = vals

        print(f"[pkaraptor] {name}: attached {hits}/{len(df)} pKa values")

    def combined(row) -> float:
        vals: list[float] = []
        ch = str(row.get("chain", "")).strip()
        rn = str(row.get("resname", "")).strip().upper()
        num = row.get("resnum", None)

        if rn == "" or num is None or pd.isna(num):
            return float("nan")
        numi = int(num)

        for name in methods_t:
            d = method_dicts.get(name)
            if d is None:
                continue

            v = _lookup_pka(d, ch, rn, numi)
            if name == "pypka" and np.isnan(v):
                v = _lookup_pka_by_resnum_only(d, ch, numi, exclude_resnames={"NTR", "CTR"})

            if name == "propka":
                v = _sanitize_pka(v, treat_propka_ss=True)
            else:
                v = _sanitize_pka(v, treat_propka_ss=False)

            if not np.isnan(v):
                vals.append(float(v))

        return float(np.mean(vals)) if vals else float("nan")

    df["Effective_pKa"] = df.apply(combined, axis=1)

    # Keep residues if they have any pKa OR they are disulfide cysteines (dashboard wants them visible).
    disulfide_keep = df.get("Disulfide_bridge", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pka_cols_present = [c for c in ("propka_pKa", "pypka_pKa", "deepka_pKa", "Effective_pKa") if c in df.columns]
    if pka_cols_present:
        any_pka_mask = ~df[pka_cols_present].isna().all(axis=1)
        df = df.loc[any_pka_mask | disulfide_keep].copy()

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _drop_duplicate_columns_case_insensitive(df)

    return df