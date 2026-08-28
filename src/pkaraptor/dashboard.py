from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return min(max(v, 0.0), 1.0)


def _lerp_rgb(c1, c2, t: float) -> np.ndarray:
    t = float(t)
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    rgb = (c1 + (c2 - c1) * t).round().astype(int)
    return rgb


def color_r2y2g(frac: float) -> str:
    f = clamp01(frac)
    if f <= 0.5:
        t = f / 0.5
        c1 = [239, 68, 68]
        c2 = [250, 204, 21]
    else:
        t = (f - 0.5) / 0.5
        c1 = [250, 204, 21]
        c2 = [34, 197, 94]

    rgb = _lerp_rgb(c1, c2, t)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def color_cb_magma_light(frac: float) -> str:
    f = clamp01(frac)
    stops = [
        (0.00, [0x6A, 0x00, 0xA8]),
        (0.35, [0xB1, 0x2A, 0x90]),
        (0.70, [0xF1, 0x60, 0x5D]),
        (1.00, [0xFE, 0xE0, 0x8B]),
    ]

    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        if f <= x1:
            t = 0.0 if x1 == x0 else (f - x0) / (x1 - x0)
            rgb = _lerp_rgb(c0, c1, t)
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    rgb = np.array(stops[-1][1], dtype=int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _to_float_or_none(x):
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _clean_text(x, default: str = "") -> str:
    if x is None or (not isinstance(x, (list, dict)) and pd.isna(x)):
        return default
    text = str(x).strip()
    return default if text.lower() in {"", "nan", "none"} else text


def _normalize_sasa_exposure(value) -> str:
    """Use SASA-specific wording without implying that accessible surface contacts water."""
    text = _clean_text(value)
    aliases = {
        "solvent-exposed": "SASA-exposed",
        "partially exposed": "partially SASA-exposed",
        "buried": "SASA-buried",
    }
    return aliases.get(text.lower(), text)


def _extract_resnum_from_label(label: str) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 0


def _is_disulfide_propka(resname: str, propka_val: float | None) -> bool:
    if resname.upper() != "CYS":
        return False
    if propka_val is None:
        return False
    return propka_val >= 90.0


def prepare_df_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Protonation_state" in out.columns:
        out["Protonation_state"] = out["Protonation_state"].replace({"Ambiguous": "Mixed"})
    if "Exposure" in out.columns:
        out["Exposure"] = out["Exposure"].map(_normalize_sasa_exposure)

    for c in (
        "propka_pKa",
        "propka_pKa_raw",
        "pypka_pKa",
        "deepka_pKa",
        "Effective_pKa",
        "SASA_frac",
        "Protonation_fraction",
        "Average_pKa",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "resname" not in out.columns:
        out["resname"] = out["Residue"].astype(str).str.extract(r":?([A-Za-z]{3})\d+")[0].fillna("")

    resname = out["resname"].astype(str)

    if "Disulfide_bridge" in out.columns:
        out["Disulfide_bridge"] = out["Disulfide_bridge"].fillna(False).astype(bool)
    else:
        if "propka_pKa_raw" in out.columns:
            propka_for_ss = out["propka_pKa_raw"]
        else:
            propka_for_ss = out["propka_pKa"] if "propka_pKa" in out.columns else pd.Series(np.nan, index=out.index)

        disulfide_mask = (resname.str.upper() == "CYS") & (propka_for_ss >= 90.0)
        out["Disulfide_bridge"] = disulfide_mask.fillna(False)

    out["propka_pKa_for_avg"] = out.get("propka_pKa", np.nan)
    out.loc[out["Disulfide_bridge"] == True, "propka_pKa_for_avg"] = np.nan  # noqa: E712

    cols_for_avg = []
    if "propka_pKa_for_avg" in out.columns:
        cols_for_avg.append("propka_pKa_for_avg")
    if "pypka_pKa" in out.columns:
        cols_for_avg.append("pypka_pKa")
    if "deepka_pKa" in out.columns:
        cols_for_avg.append("deepka_pKa")

    if cols_for_avg:
        out["Average_pKa"] = out[cols_for_avg].mean(axis=1, skipna=True)
    else:
        out["Average_pKa"] = np.nan

    if "Effective_pKa" not in out.columns or out["Effective_pKa"].isna().all():
        out["Effective_pKa"] = out["Average_pKa"]

    if "resnum" not in out.columns:
        out["resnum"] = out["Residue"].astype(str).apply(_extract_resnum_from_label).astype(int)

    if "chain" not in out.columns:
        out["chain"] = (
            out["Residue"]
            .astype(str)
            .str.split(":")
            .str[0]
            .where(out["Residue"].astype(str).str.contains(":"), "")
        )
    out["chain"] = out["chain"].fillna("").astype(str)

    if "In_membrane_slab" in out.columns:
        out["In_membrane_slab"] = out["In_membrane_slab"].fillna(False).astype(bool)
    else:
        out["In_membrane_slab"] = False

    if "OPM_id" not in out.columns:
        out["OPM_id"] = ""

    if "Disulfide_partner" not in out.columns:
        out["Disulfide_partner"] = ""

    return out


def build_plot(df: pd.DataFrame, ph: float) -> str:
    work = df.copy()
    work["_chain"] = work.get("chain", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_chain"] = work["_chain"].replace({"": "(blank)"})
    work["_resnum"] = pd.to_numeric(
        work.get("resnum", pd.Series(range(len(work)), index=work.index)), errors="coerce"
    )
    work["_avg"] = pd.to_numeric(
        work.get("Average_pKa", work.get("Effective_pKa", pd.Series(np.nan, index=work.index))),
        errors="coerce",
    )
    work = work.dropna(subset=["_resnum"])
    chains = sorted(work["_chain"].unique().tolist())
    observed_numbers = sorted(work["_resnum"].astype(int).unique().tolist())
    residue_numbers = (
        list(range(observed_numbers[0], observed_numbers[-1] + 1)) if observed_numbers else []
    )
    chain_index = {chain: i for i, chain in enumerate(chains)}
    residue_index = {resnum: i for i, resnum in enumerate(residue_numbers)}
    z = [[None for _ in residue_numbers] for _ in chains]
    custom = [[["", None, "", "No"] for _ in residue_numbers] for _ in chains]

    for _, row in work.iterrows():
        chain = str(row["_chain"])
        resnum = int(row["_resnum"])
        i = chain_index[chain]
        j = residue_index[resnum]
        avg = row["_avg"]
        if pd.notna(avg):
            z[i][j] = float(avg) - float(ph)
        custom[i][j] = [
            str(row.get("Residue", f"{chain}:{resnum}")),
            None if pd.isna(avg) else float(avg),
            str(row.get("Exposure", "") or ""),
            "Yes" if bool(row.get("In_membrane_slab", False)) else "No",
        ]

    fig = go.Figure(
        data=go.Heatmap(
            x=residue_numbers,
            y=chains,
            z=z,
            customdata=custom,
            zmin=-5,
            zmax=5,
            zmid=0,
            colorscale=[[0.0, "#2563eb"], [0.5, "#f8fafc"], [1.0, "#dc2626"]],
            colorbar=dict(title="Δ(pKa−pH)", thickness=16, len=0.78),
            xgap=1,
            ygap=2,
            hoverongaps=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Chain: %{y}<br>Residue number: %{x}<br>"
                "pKa (avg): %{customdata[1]:.2f}<br>"
                "Δ(pKa−pH): %{z:.2f}<br>"
                "SASA exposure: %{customdata[2]}<br>"
                "Membrane-embedded: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"<b>Chain-aware pKa − pH heatmap</b><br><sup>Reference pH {ph:.2f} • white = pKa near pH • blank = no value • colour saturates beyond ±5</sup>",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        height=max(300, 54 * max(len(chains), 1) + 150),
        margin=dict(l=70, r=40, t=80, b=55),
        xaxis=dict(
            title="Residue number",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Chain",
            showgrid=False,
            autorange="reversed",
        ),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def _parse_neighbor_count(s: str) -> int:
    raw = str(s or "").split(",")
    out = []
    for tok in raw:
        m = "".join(ch for ch in tok if ch.isdigit())
        if m:
            out.append(int(m))
    return len(sorted(set(out)))


def dataframe_to_js(df: pd.DataFrame, ph: float) -> tuple[str, str]:
    records = []
    total = len(df)

    counts_state = {"Protonated": 0, "Deprotonated": 0, "Mixed": 0}
    counts_agree = {"High": 0, "Moderate": 0, "Low": 0}
    counts_methods_present = {0: 0, 1: 0, 2: 0, 3: 0}
    coverage = {"propka": 0, "pypka": 0, "deepka": 0, "propka_ss": 0}

    options_map = {
        "ASP": ["ASP", "ASH"],
        "GLU": ["GLU", "GLH"],
        "HIS": ["HID", "HIE", "HIP"],
        "CYS": ["CYS", "CYM", "CYX"],
        "LYS": ["LYS", "LYN"],
        "ARG": ["ARG", "ARN"],
        "TYR": ["TYR", "TYM"],
    }

    for _, r in df.iterrows():
        residue = str(r.get("Residue", ""))
        resnum = int(r.get("resnum", _extract_resnum_from_label(residue)) or 0)
        chain = str(r.get("chain", "") or "")

        resname = str(r.get("resname", "")).upper()

        propka_val = _to_float_or_none(r.get("propka_pKa_raw", r.get("propka_pKa", np.nan)))
        pypka_val = _to_float_or_none(r.get("pypka_pKa", np.nan))
        deepka_val = _to_float_or_none(r.get("deepka_pKa", np.nan))
        avg_val = _to_float_or_none(r.get("Average_pKa", np.nan))
        frac_val = _to_float_or_none(r.get("Protonation_fraction", np.nan))
        sasa_val = _to_float_or_none(r.get("SASA_frac", np.nan))

        in_mem = bool(r.get("In_membrane_slab", False))
        opm_id = str(r.get("OPM_id", "") or "")

        disulfide = bool(r.get("Disulfide_bridge", False)) or _is_disulfide_propka(resname, propka_val)
        if disulfide and resname == "CYS" and propka_val is not None and propka_val >= 90.0:
            coverage["propka_ss"] += 1

        state = str(r.get("Protonation_state", "Mixed"))
        if state == "Ambiguous":
            state = "Mixed"
        if state in counts_state:
            counts_state[state] += 1

        has_propka_numeric = (propka_val is not None) and (not disulfide)
        has_propka_any = propka_val is not None

        has_pypka = pypka_val is not None
        has_deepka = deepka_val is not None

        coverage["propka"] += int(has_propka_any)
        coverage["pypka"] += int(has_pypka)
        coverage["deepka"] += int(has_deepka)

        numeric_vals = []
        if has_propka_numeric:
            numeric_vals.append(propka_val)
        if has_pypka:
            numeric_vals.append(pypka_val)
        if has_deepka:
            numeric_vals.append(deepka_val)

        methods_present = int(has_propka_numeric) + int(has_pypka) + int(has_deepka)
        counts_methods_present[methods_present] += 1

        pka_spread = None
        agree = None

        if len(numeric_vals) >= 2:
            pka_spread = float(max(numeric_vals) - min(numeric_vals))

            if pka_spread <= 1.0:
                agree = "High"
            elif pka_spread <= 2.0:
                agree = "Moderate"
            else:
                agree = "Low"

        if agree is not None:
            counts_agree[agree] += 1

        neighbor_count = _parse_neighbor_count(r.get("Neighbors_within_5A", ""))

        frac_for_color = frac_val if frac_val is not None else 0.5
        color_default = color_r2y2g(frac_for_color)
        color_cb = color_cb_magma_light(frac_for_color)

        allowed_resnames = options_map.get(resname, [resname] if resname else [""])
        # Protonation and tautomer assignments must come from an explicit user choice.
        default_resname = ""

        interactions_raw = r.get("Interaction_diagram_json", "")
        interactions = []
        if isinstance(interactions_raw, str) and interactions_raw.strip():
            try:
                parsed_interactions = json.loads(interactions_raw)
                if isinstance(parsed_interactions, list):
                    interactions = parsed_interactions
            except (TypeError, ValueError, json.JSONDecodeError):
                interactions = []

        records.append(
            {
                "Residue": residue,
                "chain": chain,
                "resnum": resnum,
                "resname": resname,
                "propka_pKa": _to_float_or_none(r.get("propka_pKa", np.nan)),
                "propka_pKa_raw": propka_val,
                "pypka_pKa": pypka_val,
                "deepka_pKa": deepka_val,
                "avg_pKa": avg_val,
                "Disulfide_bridge": disulfide,
                "Disulfide_partner": str(r.get("Disulfide_partner", "") or ""),
                "Protonation_fraction": frac_val,
                "Protonation_state": state,
                "Exposure": _normalize_sasa_exposure(r.get("Exposure", "")),
                "SASA_frac": sasa_val,
                "Neighbors_within_5A": str(r.get("Neighbors_within_5A", "")),
                "neighbor_count": int(neighbor_count),
                "HBond_shell_donors_count": _safe_int(r.get("HBond_shell_donors_count", np.nan)) or 0,
                "HBond_shell_acceptors_count": _safe_int(r.get("HBond_shell_acceptors_count", np.nan)) or 0,
                "HBond_shell_donors": str(r.get("HBond_shell_donors", "")),
                "HBond_shell_acceptors": str(r.get("HBond_shell_acceptors", "")),
                "Ionizable_group_atoms": _clean_text(r.get("Ionizable_group_atoms", "")),
                "Ionizable_group_z_A": _to_float_or_none(r.get("Ionizable_group_z_A", np.nan)),
                "interactions": interactions,
                "color_default": color_default,
                "color_cb": color_cb,
                "methods_present": methods_present,
                "has_propka": bool(has_propka_any),
                "has_propka_numeric": bool(has_propka_numeric),
                "has_pypka": bool(has_pypka),
                "has_deepka": bool(has_deepka),
                "pka_spread": pka_spread,
                "method_agreement": agree,
                "delta_pka_ph": (avg_val - ph) if avg_val is not None else None,
                "In_membrane_slab": bool(in_mem),
                "OPM_id": opm_id,
                "allowed_resnames": allowed_resnames,
                "default_resname": default_resname,
            }
        )

    mean_pka = float(df["Average_pKa"].dropna().mean()) if "Average_pKa" in df.columns else np.nan
    mean_frac = float(df["Protonation_fraction"].dropna().mean()) if "Protonation_fraction" in df.columns else np.nan

    summary = {
        "total": int(total),
        "protonated": int(counts_state["Protonated"]),
        "deprotonated": int(counts_state["Deprotonated"]),
        "mixed": int(counts_state["Mixed"]),
        "mean_pka": mean_pka if not np.isnan(mean_pka) else None,
        "mean_fraction": mean_frac if not np.isnan(mean_frac) else None,
        "coverage": {
            "propka_any": int(coverage["propka"]),
            "propka_ss": int(coverage["propka_ss"]),
            "pypka": int(coverage["pypka"]),
            "deepka": int(coverage["deepka"]),
        },
        "methods_present": {
            "n0": int(counts_methods_present[0]),
            "n1": int(counts_methods_present[1]),
            "n2": int(counts_methods_present[2]),
            "n3": int(counts_methods_present[3]),
        },
        "method_agreement": {
            "high": int(counts_agree["High"]),
            "moderate": int(counts_agree["Moderate"]),
            "low": int(counts_agree["Low"]),
        },
        "ph": float(ph),
    }

    return json.dumps(records), json.dumps(summary)


def build_dashboard_html(df: pd.DataFrame, pdb_path: str, ph: float) -> str:
    df = prepare_df_for_dashboard(df)

    pdb_raw = Path(pdb_path).read_text(encoding="utf-8")
    pdb_js_literal = json.dumps(pdb_raw)

    js_data, js_summary = dataframe_to_js(df, ph)

    footer_html = """
<footer class="app-footer">
  <div class="footer-inner">
    <div class="footer-left">HPCMM Research Group @ LAQV-REQUIMTE</div>
    <div class="footer-right">Please cite:[DOI]</div>
  </div>
</footer>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Protonation Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
  --card-bg: rgba(15,23,42,0.96);
  --accent-soft: rgba(56,189,248,0.15);
  --border-subtle: rgba(148,163,184,0.25);
  --text-main: #e5e7eb;
  --text-muted: #9ca3af;
  --radius-lg: 18px;
  --radius-md: 12px;
  --shadow-soft: 0 18px 45px rgba(15,23,42,0.7);
  --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
  --page-bg: radial-gradient(circle at top left, #1d2435, #020617 60%);
  --viewer-bg: radial-gradient(circle at top, #f8fafc, #e5e7eb 70%);
  --viewer-stage: #f1f5f9;
}}

body.theme-light {{
  --card-bg: rgba(255,255,255,0.92);
  --accent-soft: rgba(37,99,235,0.12);
  --border-subtle: rgba(15,23,42,0.14);
  --text-main: #0f172a;
  --text-muted: rgba(30,41,59,0.75);
  --shadow-soft: 0 18px 45px rgba(15,23,42,0.12);
  --page-bg: radial-gradient(circle at top left, #f1f5f9, #ffffff 62%);
  --viewer-bg: radial-gradient(circle at top, #ffffff, #e2e8f0 75%);
  --viewer-stage: #ffffff;
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 0;
  font-family: var(--font-sans);
  background: var(--page-bg);
  color: var(--text-main);
}}
code {{ color: rgba(224,242,254,0.95); }}
body.theme-light code {{ color: rgba(30,64,175,0.95); }}

.app-shell {{
  max-width: 1320px;
  margin: 24px auto 32px auto;
  padding: 0 16px;
}}
.header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  gap: 12px;
}}
.header-left h1 {{
  font-size: 1.6rem;
  letter-spacing: 0.02em;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.header-left h1 span.icon {{
  width: 32px;
  height: 32px;
  border-radius: 999px;
  background: radial-gradient(circle at 20% 20%, #38bdf8, #4ade80);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
}}
.header-left h1 span:last-child {{ font-weight: 600; }}
.header-sub {{
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 4px;
}}
.header-actions {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}}
.help-btn {{
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(99,102,241,0.14);
  border: 1px solid rgba(99,102,241,0.35);
  color: rgba(224,231,255,0.95);
  font-size: 0.8rem;
  cursor: pointer;
}}
.palette-btn {{
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.35);
  color: rgba(209,250,229,0.95);
  font-size: 0.8rem;
  cursor: pointer;
}}
.diag-btn {{
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(244,63,94,0.10);
  border: 1px solid rgba(244,63,94,0.35);
  color: rgba(255,228,230,0.95);
  font-size: 0.8rem;
  cursor: pointer;
}}
.theme-btn {{
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(14,165,233,0.12);
  border: 1px solid rgba(14,165,233,0.35);
  color: rgba(224,242,254,0.95);
  font-size: 0.8rem;
  cursor: pointer;
}}
body.theme-light .help-btn {{
  background: rgba(79,70,229,0.10);
  border-color: rgba(79,70,229,0.28);
  color: rgba(15,23,42,0.95);
}}
body.theme-light .palette-btn {{
  background: rgba(16,185,129,0.10);
  border-color: rgba(16,185,129,0.26);
  color: rgba(15,23,42,0.95);
}}
body.theme-light .diag-btn {{
  background: rgba(244,63,94,0.08);
  border-color: rgba(244,63,94,0.22);
  color: rgba(15,23,42,0.95);
}}
body.theme-light .theme-btn {{
  background: rgba(14,165,233,0.10);
  border-color: rgba(14,165,233,0.26);
  color: rgba(15,23,42,0.95);
}}

.badge-ph {{
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: #e0f2fe;
  font-size: 0.8rem;
  border: 1px solid rgba(56,189,248,0.35);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
body.theme-light .badge-ph {{
  color: rgba(15,23,42,0.95);
  border-color: rgba(37,99,235,0.22);
}}

.main-stack {{
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
.card {{
  background: var(--card-bg);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-soft);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.card-header {{
  padding: 12px 16px 8px 16px;
  border-bottom: 1px solid rgba(148,163,184,0.2);
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.card-title {{
  font-size: 0.9rem;
  font-weight: 550;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--text-muted);
}}
.card-body {{
  padding: 12px 16px 14px 16px;
  flex: 1;
}}

.hidden {{ display: none !important; }}

.guide {{ display: none; }}
.guide.visible {{ display: block; }}
.guide ul {{
  margin: 8px 0 0 18px;
  padding: 0;
  color: rgba(226,232,240,0.92);
  line-height: 1.5;
  font-size: 0.92rem;
}}
body.theme-light .guide ul {{ color: rgba(15,23,42,0.92); }}
.guide .small {{
  margin-top: 10px;
  color: rgba(148,163,184,0.95);
  font-size: 0.85rem;
}}
body.theme-light .guide .small {{ color: rgba(30,41,59,0.75); }}

.viewer-shell {{
  position: relative;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.35);
  background: var(--viewer-bg);
  height: 560px;
}}
#viewport {{ width: 100%; height: 100%; }}

#atomHoverTooltip {{
  position: absolute;
  display: none;
  pointer-events: none;
  z-index: 100;
  padding: 5px 8px;
  border-radius: 5px;
  background: rgba(15, 23, 42, 0.92);
  color: #ffffff;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
  box-shadow: 0 3px 10px rgba(0,0,0,0.25);
}}
.viewer-controls {{
  position: absolute;
  left: 12px;
  top: 12px;
  display: flex;
  gap: 8px;
  z-index: 20;
}}
.viewer-btn {{
  appearance: none;
  border: 1px solid rgba(148,163,184,0.7);
  background: rgba(248,250,252,0.9);
  color: #0f172a;
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  cursor: pointer;
  box-shadow: 0 10px 20px rgba(15,23,42,0.15);
}}
.viewer-btn[disabled] {{
  opacity: 0.55;
  cursor: not-allowed;
}}

#infoBox {{
  position: absolute;
  right: 14px;
  top: 14px;
  min-width: 320px;
  max-width: 560px;
  padding: 10px 12px;
  background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(15,23,42,0.86));
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.45);
  box-shadow: 0 15px 40px rgba(15,23,42,0.55);
  font-size: 0.78rem;
  color: #e5e7eb;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.25s ease;
  backdrop-filter: blur(14px);
  z-index: 15;
}}
body.theme-light #infoBox {{
  background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92));
  border-color: rgba(15,23,42,0.16);
  box-shadow: 0 18px 50px rgba(15,23,42,0.12);
  color: #0f172a;
}}
#infoBox.visible {{
  opacity: 1;
  pointer-events: auto;
}}

#infoBox .info-topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}}

#infoBox .info-actions {{
  display: inline-flex;
  gap: 6px;
  align-items: center;
}}

#infoBox .info-min-btn {{
  appearance: none;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(148,163,184,0.10);
  color: rgba(226,232,240,0.95);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 0.72rem;
  cursor: pointer;
  line-height: 1;
}}

body.theme-light #infoBox .info-min-btn {{
  border-color: rgba(15,23,42,0.16);
  background: rgba(15,23,42,0.06);
  color: rgba(15,23,42,0.9);
}}

#infoBox.minimized .info-content {{
  display: none;
}}
#infoBox .info-content {{ max-height: 68vh; overflow-y: auto; padding-right: 4px; }}
.info-grid {{
  display: grid;
  grid-template-columns: auto auto;
  gap: 2px 10px;
  margin-top: 6px;
  color: rgba(148,163,184,0.95);
}}
body.theme-light .info-grid {{ color: rgba(30,41,59,0.75); }}
.info-grid b {{ color: inherit; }}
body.theme-light .info-grid b {{ color: #0f172a; }}
.info-subtitle {{
  margin-top: 10px;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(226,232,240,0.7);
}}
body.theme-light .info-subtitle {{ color: rgba(30,41,59,0.75); }}
.hb-list {{
  margin-top: 6px;
  color: rgba(226,232,240,0.92);
  line-height: 1.35;
  word-break: break-word;
}}
body.theme-light .hb-list {{ color: rgba(15,23,42,0.92); }}
.chip-residue {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(15,118,110,0.18);
  border: 1px solid rgba(45,212,191,0.4);
  font-size: 0.75rem;
  margin-bottom: 6px;
}}
body.theme-light .chip-residue {{
  background: rgba(37,99,235,0.10);
  border-color: rgba(37,99,235,0.22);
}}
.chip-residue span.dot {{
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #22c55e;
}}
.plot-shell {{
  border-radius: var(--radius-md);
  border: 1px solid rgba(148,163,184,0.35);
  background: radial-gradient(circle at top left, rgba(15,23,42,1), rgba(15,23,42,0.96));
  padding: 8px 10px 8px 10px;
}}
body.theme-light .plot-shell {{
  background: radial-gradient(circle at top left, rgba(255,255,255,1), rgba(241,245,249,0.96));
}}
.titration-empty {{ min-height:220px; display:flex; align-items:center; justify-content:center; color:rgba(148,163,184,0.9); font-size:0.85rem; text-align:center; padding:20px; }}
.titration-note {{ margin-top:7px; color:rgba(148,163,184,0.9); font-size:0.68rem; line-height:1.4; }}
body.theme-light .titration-note {{ color:rgba(30,41,59,0.62); }}

.table-tools {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}}
.search-input {{ flex: 1; position: relative; min-width: 260px; }}
.search-input input {{
  width: 100%;
  padding: 7px 32px 7px 28px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.5);
  background: rgba(15,23,42,0.9);
  color: #e5e7eb;
  font-size: 0.8rem;
  outline: none;
}}
body.theme-light .search-input input {{
  background: rgba(255,255,255,0.92);
  color: #0f172a;
  border-color: rgba(15,23,42,0.16);
}}
.search-input span.icon {{
  position: absolute;
  left: 9px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.78rem;
  color: rgba(148,163,184,0.9);
}}
body.theme-light .search-input span.icon {{ color: rgba(30,41,59,0.65); }}
.search-clear {{
  position: absolute;
  right: 7px;
  top: 50%;
  transform: translateY(-50%);
  border: 0;
  background: transparent;
  color: rgba(148,163,184,0.9);
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 2px 4px;
}}
.search-clear[hidden] {{ display: none; }}
body.theme-light .search-clear {{ color: rgba(30,41,59,0.65); }}
.search-count {{
  color: rgba(148,163,184,0.9);
  font-size: 0.72rem;
  white-space: nowrap;
}}
body.theme-light .search-count {{ color: rgba(30,41,59,0.65); }}
.table-policy-note {{ margin:7px 4px 0; color:rgba(148,163,184,0.9); font-size:0.67rem; line-height:1.4; }}
body.theme-light .table-policy-note {{ color:rgba(30,41,59,0.62); }}

.filter-chips {{
  display: inline-flex;
  gap: 6px;
  flex-wrap: wrap;
}}
.filter-chip {{
  padding: 5px 9px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.6);
  background: rgba(15,23,42,0.9);
  font-size: 0.75rem;
  color: rgba(148,163,184,0.95);
  cursor: pointer;
  user-select: none;
}}
body.theme-light .filter-chip {{
  background: rgba(255,255,255,0.92);
  color: rgba(30,41,59,0.75);
  border-color: rgba(15,23,42,0.14);
}}
.filter-chip.active {{
  background: var(--accent-soft);
  border-color: rgba(56,189,248,0.9);
  color: #e0f2fe;
}}
body.theme-light .filter-chip.active {{
  border-color: rgba(37,99,235,0.45);
  color: rgba(15,23,42,0.95);
}}

.table-wrap {{
  border-radius: var(--radius-md);
  border: 1px solid rgba(148,163,184,0.35);
  overflow: hidden;
  background: radial-gradient(circle at top, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
}}
body.theme-light .table-wrap {{
  background: radial-gradient(circle at top, rgba(255,255,255,0.98), rgba(241,245,249,0.96));
}}
.table-inner {{
  max-height: 520px;
  overflow: auto;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}}
thead th {{
  position: sticky;
  top: 0;
  background: rgba(15,23,42,0.98);
  border-bottom: 1px solid rgba(51,65,85,1);
  padding: 7px 9px;
  text-align: left;
  font-weight: 550;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(148,163,184,0.95);
  backdrop-filter: blur(10px);
  z-index: 2;
}}
body.theme-light thead th {{
  background: rgba(255,255,255,0.98);
  border-bottom-color: rgba(15,23,42,0.12);
  color: rgba(30,41,59,0.75);
}}
tbody td {{
  padding: 6px 9px;
  border-bottom: 1px solid rgba(30,41,59,1);
  color: #e5e7eb;
  vertical-align: middle;
}}
body.theme-light tbody td {{
  border-bottom-color: rgba(15,23,42,0.10);
  color: #0f172a;
}}
tbody tr {{
  cursor: pointer;
  transition: background 0.1s ease;
}}
tbody tr:nth-child(2n) {{ background: rgba(15,23,42,0.8); }}
body.theme-light tbody tr:nth-child(2n) {{ background: rgba(241,245,249,0.9); }}
tbody tr:hover {{ background: rgba(30,64,175,0.55); }}
body.theme-light tbody tr:hover {{ background: rgba(37,99,235,0.12); }}
tbody tr.row-selected {{
  outline: 2px solid rgba(56,189,248,0.55);
  outline-offset: -2px;
  background: rgba(30,64,175,0.68) !important;
}}
body.theme-light tbody tr.row-selected {{
  outline-color: rgba(37,99,235,0.35);
  background: rgba(37,99,235,0.14) !important;
}}

.fraction-bar {{
  position: relative;
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: rgba(30,41,59,1);
  overflow: hidden;
  margin-top: 4px;
}}
body.theme-light .fraction-bar {{ background: rgba(226,232,240,1); }}
.fraction-bar-inner {{
  height: 100%;
  border-radius: 999px;
  transition: width 0.3s ease, background 0.2s ease;
}}
.fraction-label {{
  font-size: 0.7rem;
  color: rgba(148,163,184,0.95);
}}
body.theme-light .fraction-label {{ color: rgba(30,41,59,0.70); }}
.small-num {{
  font-variant-numeric: tabular-nums;
  color: rgba(226,232,240,0.92);
}}
body.theme-light .small-num {{ color: rgba(15,23,42,0.92); }}
.membrane-flag {{
  display: inline-flex;
  align-items: center;
  margin-top: 4px;
  padding: 2px 7px;
  border: 1px solid rgba(56,189,248,0.65);
  border-radius: 999px;
  background: rgba(14,116,144,0.20);
  color: #7dd3fc;
  font-size: 0.66rem;
  font-weight: 650;
  letter-spacing: 0.03em;
  white-space: nowrap;
}}
body.theme-light .membrane-flag {{
  border-color: rgba(2,132,199,0.45);
  background: rgba(14,165,233,0.10);
  color: #075985;
}}

.diag-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}}
@media (min-width: 980px) {{
  .diag-grid {{ grid-template-columns: 1fr 1fr; }}
}}
.diag-panel {{
  border-radius: var(--radius-md);
  border: 1px solid rgba(148,163,184,0.30);
  background: radial-gradient(circle at top, rgba(15,23,42,0.98), rgba(15,23,42,0.94));
  overflow: hidden;
}}
body.theme-light .diag-panel {{
  background: radial-gradient(circle at top, rgba(255,255,255,0.98), rgba(241,245,249,0.94));
}}
.diag-panel-header {{
  padding: 10px 12px 8px 12px;
  border-bottom: 1px solid rgba(148,163,184,0.18);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}}
.diag-panel-title {{
  font-size: 0.85rem;
  font-weight: 650;
  letter-spacing: 0.04em;
  color: rgba(226,232,240,0.9);
  text-transform: uppercase;
}}
body.theme-light .diag-panel-title {{ color: rgba(15,23,42,0.9); }}
.diag-panel-sub {{
  font-size: 0.76rem;
  color: rgba(148,163,184,0.95);
}}
body.theme-light .diag-panel-sub {{ color: rgba(30,41,59,0.70); }}
.diag-panel-body {{
  padding: 10px 12px 12px 12px;
}}
.diag-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}}
.diag-chip {{
  padding: 5px 9px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.45);
  background: rgba(15,23,42,0.7);
  color: rgba(226,232,240,0.92);
  font-size: 0.75rem;
  user-select: none;
}}
body.theme-light .diag-chip {{
  background: rgba(255,255,255,0.92);
  border-color: rgba(15,23,42,0.14);
  color: rgba(15,23,42,0.85);
}}
.diag-chip.clickable {{ cursor: pointer; }}
.diag-chip.active {{
  background: var(--accent-soft);
  border-color: rgba(56,189,248,0.9);
  color: #e0f2fe;
}}
body.theme-light .diag-chip.active {{
  border-color: rgba(37,99,235,0.45);
  color: rgba(15,23,42,0.95);
}}
.diag-chip strong {{
  color: #e0f2fe;
  font-weight: 750;
}}
body.theme-light .diag-chip strong {{ color: rgba(37,99,235,0.95); }}

.diag-table-wrap {{
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,0.25);
  overflow: hidden;
  background: rgba(15,23,42,0.55);
}}
body.theme-light .diag-table-wrap {{
  background: rgba(255,255,255,0.75);
  border-color: rgba(15,23,42,0.10);
}}
.diag-table-inner {{
  max-height: 420px;
  overflow: auto;
}}
.mini-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}}
.mini-table th {{
  position: sticky;
  top: 0;
  text-align: left;
  font-size: 0.70rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(148,163,184,0.95);
  padding: 6px 6px;
  border-bottom: 1px solid rgba(51,65,85,1);
  background: rgba(15,23,42,0.92);
  backdrop-filter: blur(10px);
  z-index: 2;
}}
body.theme-light .mini-table th {{
  background: rgba(255,255,255,0.96);
  border-bottom-color: rgba(15,23,42,0.12);
  color: rgba(30,41,59,0.75);
}}
.mini-table td {{
  padding: 6px 6px;
  border-bottom: 1px solid rgba(30,41,59,1);
  color: rgba(226,232,240,0.92);
  vertical-align: middle;
}}
body.theme-light .mini-table td {{
  border-bottom-color: rgba(15,23,42,0.10);
  color: rgba(15,23,42,0.92);
}}
.mini-table tr:hover {{ background: rgba(30,64,175,0.45); }}
body.theme-light .mini-table tr:hover {{ background: rgba(37,99,235,0.10); }}

.badge-ok {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(22,163,74,0.16);
  border: 1px solid rgba(34,197,94,0.55);
  color: #bbf7d0;
  font-weight: 700;
}}
.badge-miss {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(220,38,38,0.16);
  border: 1px solid rgba(248,113,113,0.55);
  color: #fecaca;
  font-weight: 700;
}}
.badge-ss {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(234,179,8,0.18);
  border: 1px solid rgba(250,204,21,0.65);
  color: #fef9c3;
  font-weight: 800;
}}

.app-footer {{
  margin-top: 28px;
  padding: 14px 16px;
  border-top: 1px solid rgba(148,163,184,0.25);
  background: rgba(15,23,42,0.85);
  backdrop-filter: blur(6px);
  font-size: 0.75rem;
  color: var(--text-muted);
}}

body.theme-light .app-footer {{
  background: rgba(255,255,255,0.85);
}}

.footer-inner {{
  max-width: 1320px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}}
.footer-left {{ font-weight: 500; }}
.footer-right {{ font-style: italic; }}
.interaction-modal {{ position:fixed; inset:0; z-index:10000; background:rgba(2,6,23,.78); display:none; align-items:center; justify-content:center; padding:24px; }}
.interaction-modal.visible {{ display:flex; }}
.interaction-dialog {{ width:min(860px,94vw); max-height:90vh; overflow:auto; background:#fff; color:#172033; border-radius:16px; box-shadow:0 24px 80px rgba(0,0,0,.45); }}
.interaction-header {{ display:flex; justify-content:space-between; align-items:center; padding:16px 20px; border-bottom:1px solid #dbe3ef; }}
.interaction-actions {{ display:flex; gap:8px; }}
.interaction-actions button, .interaction-btn {{ border:1px solid #9fb0c7; border-radius:8px; padding:6px 10px; background:#f7faff; color:#183b66; cursor:pointer; font-weight:700; }}
.interaction-canvas {{ padding:8px 16px 2px; min-height:360px; }}
.state-map-warning {{ margin:8px 0 2px; padding:9px 12px; border:1px solid #f59e0b; border-radius:10px; background:#fffbeb; color:#78350f; font-size:.76rem; line-height:1.35; }}
.interaction-note {{ padding:0 20px 18px; color:#53657d; font-size:13px; }}
.interaction-svg {{ display:block; width:100%; height:auto; min-height:0; }}
.interaction-node {{ cursor:pointer; }}
</style>
</head>
<body>
<div class="app-shell">
  <header class="header">
    <div class="header-left">
      <h1><span class="icon">🖥️</span><span>Protonation Dashboard</span></h1>
      <div class="header-sub">
        Interactive pKa & microenvironment view for <code>{Path(pdb_path).name}</code>
      </div>
    </div>
    <div class="header-actions">
      <button class="help-btn" id="toggleGuideBtn" type="button">Guide</button>
      <button class="palette-btn" id="togglePaletteBtn" type="button">Palette: Default</button>
      <button class="diag-btn" id="toggleDiagBtn" type="button">Diagnostics: on</button>
      <button class="theme-btn" id="toggleThemeBtn" type="button">Theme: Dark</button>
      <button class="help-btn" id="downloadJsonBtn" type="button">Download JSON</button>
      <span class="badge-ph"><span>pH</span><strong>{ph:.2f}</strong></span>
    </div>
  </header>

  <div class="main-stack">

    <section class="card guide" id="guideCard">
      <div class="card-header"><div class="card-title">User Guide</div></div>
      <div class="card-body">
        <ul>
          <li><b>Guide</b> (top bar): shows or hides this help panel.</li>
          <li><b>Palette</b> (top bar): switches between default and colorblind-friendly protonation color mapping.</li>
          <li><b>Diagnostics</b> (top bar): shows or hides the diagnostics panels (method agreement, coverage, atypical pKa).</li>
          <li><b>Theme</b> (top bar): toggles between dark and light display themes.</li>
          <li><b>Selection</b>: click any residue in the table or 3D structure to focus it, display its local environment, and update the titration curves.</li>
          <li><b>Reset view</b> (Structure panel): clears the current selection, removes neighbor highlighting, and restores the default camera view.</li>
          <li><b>PPM DUM</b> (Structure panel): toggles display of membrane dummy atoms (when present) as spheres, providing a visual reference for the membrane slab used by OPM/PPM alignment.</li>
          <li><b>Search</b> (table): filters residue identity, chain, number, protonation category, SASA exposure, membrane context, or disulfide status.</li>
          <li><b>State filters</b> (table): restricts the table to residues classified as Protonated, Deprotonated, or Mixed.</li>
          <li><b>Membrane filters</b> (table): restricts the table to residues flagged as membrane-embedded or non-membrane.</li>

          <li><b>How to read this dashboard</b>:
            <ul>
              <li><b>Step 1</b>: choose a residue from the table or click directly on the structure.</li>
              <li><b>Step 2</b>: inspect <b>pKa (Avg.)</b> and compare to the reference <b>pH</b> badge in the header.</li>
              <li><b>Step 3</b>: check <b>Fraction</b> to interpret expected charge state at the reference pH.</li>
              <li><b>Step 4</b>: compare the method-specific <b>titration curves</b> and their protonated fractions at the reference pH.</li>
              <li><b>Step 5</b>: examine <b>SASA exposure</b>, <b>membrane context</b>, and <b>Neighbors</b> to relate pKa shifts to the local microenvironment.</li>
              <li><b>Step 6</b>: use <b>Method diagnostics</b> to assess uncertainty; low agreement indicates predictions diverge and should be interpreted cautiously.</li>
              <li><b>Step 7</b>: use <b>Atypical pKa</b> to inspect residues with large shifts from canonical intrinsic values.</li>
            </ul>
          </li>

          <li><b>Core concepts</b>:
            <ul>
              <li><b>Rule of thumb</b>: if <b>pKa &gt; pH</b>, the residue is predominantly protonated; if <b>pKa &lt; pH</b>, it is predominantly deprotonated.</li>
              <li><b>Protonation fraction</b>: predicted population of the protonated state at the reference pH (0 = deprotonated, 1 = protonated) based on Henderson–Hasselbalch behavior.</li>
              <li><b>Microenvironment effects</b>: burial, nearby charges, hydrogen bonds, and membrane context can shift pKa away from intrinsic reference values.</li>
              <li><b>Structure snapshot</b>: predictions use a single structure; conformational dynamics and coupled titration may not be fully captured.</li>
            </ul>
          </li>

          <li><b>Table columns explained</b>:
            <ul>
              <li><b>Residue</b>: chain identifier, residue name, and residue number.</li>
              <li><b>pKa (PROPKA)</b>: pKa predicted by PROPKA (structure-based method).</li>
              <li><b>pKa (PyPka)</b>: pKa predicted by PyPka (continuum electrostatics-based method).</li>
              <li><b>pKa (DeepKa)</b>: pKa predicted by DeepKa (machine-learning-based method).</li>
              <li><b>pKa (Avg.)</b>: average of available pKa predictions after numeric filtering.</li>
              <li><b>Set residue</b>: a manual residue-state choice for the output JSON. If untouched, non-histidine residues use the declared deprotonated fallback; histidines require an explicit tautomer/state choice. Detected disulfide cysteines are structurally locked as CYX.</li>
              <li><b>Fraction</b>: predicted protonated fraction at the reference pH.</li>
              <li><b>SASA exposure</b>: geometric accessibility category. A <b>Membrane-embedded</b> flag identifies residues supplied through <code>--opm-embedded</code>. Membrane embedding does not establish whether a side chain faces lipid, water, a pore, or a cavity.</li>
              <li><b>Neighbors</b>: residues within 5 Å of the selected residue.</li>
              <li><b>HB D</b>: count of nearby hydrogen-bond donors (filtered list shown in the info box).</li>
              <li><b>HB A</b>: count of nearby hydrogen-bond acceptors (filtered list shown in the info box).</li>
            </ul>
          </li>

          <li><b>Interpretation note</b>: predicted values are computational estimates and should be compared with experimental evidence when available.</li>
        </ul>
        <div class="small">
          Shortcuts: press <b>G</b> to toggle guide · press <b>P</b> to toggle palette · press <b>T</b> to toggle theme · press <b>Esc</b> to reset view.
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-header"><div class="card-title">Structure</div></div>
      <div class="card-body">
        <div class="viewer-shell">
          <div class="viewer-controls">
            <button class="viewer-btn" id="resetViewBtn" type="button">Reset view</button>
            <button class="viewer-btn" id="toggleDumBtn" type="button">PPM DUM: off</button>
            <button class="viewer-btn" id="colorModeBtn" type="button">Coloring: Structure</button>
          </div>
          <div id="viewport"></div>
          <div id="atomHoverTooltip"></div>
          <div id="infoBox"></div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-header"><div class="card-title">Selected-residue titration curves</div></div>
      <div class="card-body">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
          <label for="comparisonPhSlider" style="font-size:0.78rem;">
            Comparison pH:
            <b id="comparisonPhValue">–</b>
          </label>
          <input id="comparisonPhSlider"
                 type="range"
                 min="0"
                 max="14"
                 step="0.1"
                 style="width:260px;">
          <button class="viewer-btn" id="resetComparisonPhBtn" type="button">
            Reset to analysis pH
          </button>
          <span style="font-size:0.72rem;color:var(--text-muted);">
            Analysis pH: <b id="analysisPhValue">–</b>
          </span>
        </div>
        <div class="plot-shell">
          <div id="titrationPlot" class="titration-empty">Select a residue in the table or 3D structure to display its titration curves.</div>
        </div>
        <div class="titration-note">Curves show protonated fractions calculated from each available predicted pKa using the Henderson–Hasselbalch relationship. The comparison pH changes only the visual reference markers and does not modify the analysis pH, protonation-state assessment, or exported assignments.</div>
      </div>
    </section>

    <section class="card">
      <div class="card-header"><div class="card-title">Residue table</div></div>
      <div class="card-body">
        <div class="table-tools">
          <div class="search-input">
            <span class="icon">🔍</span>
            <input id="searchBox" type="search" autocomplete="off" placeholder="Search e.g. A CYS 59, membrane, SASA-exposed..." />
            <button class="search-clear" id="clearSearchBtn" type="button" aria-label="Clear search" hidden>×</button>
          </div>
          <span class="search-count" id="searchCount" aria-live="polite"></span>
          <div class="filter-chips" id="stateFilters">
            <div class="filter-chip active" data-state="all">All</div>
            <div class="filter-chip" data-state="Protonated">Prot.</div>
            <div class="filter-chip" data-state="Deprotonated">Deprot.</div>
            <div class="filter-chip" data-state="Mixed">Mixed</div>
          </div>
          <div class="filter-chips" id="membraneFilters">
            <div class="filter-chip active" data-mem="all">All</div>
            <div class="filter-chip" data-mem="mem">Membrane</div>
            <div class="filter-chip" data-mem="nonmem">Non-mem</div>
          </div>
        </div>

        <div class="table-wrap">
          <div class="table-inner">
            <table id="summary-table">
              <thead>
                <tr>
                  <th>Residue</th>
                  <th>pKa (PROPKA)</th>
                  <th>pKa (PyPka)</th>
                  <th>pKa (DeepKa)</th>
                  <th>pKa (Avg.)</th>
                  <th>Set residue</th>
                  <th>2D interactions</th>
                  <th>Fraction</th>
                  <th>SASA exposure</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
        <div class="table-policy-note">If no state is chosen, the canonical input state is preserved. Histidines require a manual state choice. Detected disulfide cysteines are locked as CYX.</div>
      </div>
    </section>

    <div id="diagnosticsStack">
      <section class="card">
        <div class="card-header"><div class="card-title">Method diagnostics</div></div>
        <div class="card-body">
          <div class="diag-grid">
            <div class="diag-panel">
              <div class="diag-panel-header">
                <div class="diag-panel-title">Method agreement</div>
                <div class="diag-panel-sub">difference between predicted pKa values</div>
              </div>
              <div class="diag-panel-body">
                <div class="diag-chips" id="agreeChips"></div>
                <div class="diag-table-wrap">
                  <div class="diag-table-inner">
                    <table class="mini-table" id="agreeTable">
                      <thead>
                        <tr>
                          <th>Residue</th>
                          <th>Methods</th>
                          <th>Spread</th>
                          <th>Class</th>
                        </tr>
                      </thead>
                      <tbody></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div class="diag-panel">
              <div class="diag-panel-header">
                <div class="diag-panel-title">Method coverage</div>
                <div class="diag-panel-sub">which tools were available</div>
              </div>
              <div class="diag-panel-body">
                <div class="diag-chips" id="covChips"></div>
                <div class="diag-table-wrap">
                  <div class="diag-table-inner">
                    <table class="mini-table" id="covTable">
                      <thead>
                        <tr>
                          <th>Residue</th>
                          <th>P</th>
                          <th>Py</th>
                          <th>D</th>
                          <th>n</th>
                        </tr>
                      </thead>
                      <tbody></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div class="card-title">Atypical pKa and Disulfide Bridges Analysis</div></div>
        <div class="card-body">
          <div class="diag-grid">

            <div class="diag-panel">
              <div class="diag-panel-header">
                <div class="diag-panel-title">Deviations vs intrinsic pKa</div>
                <div class="diag-panel-sub">|pKa(avg) − pKa(ref)|</div>
              </div>
              <div class="diag-panel-body">
                <div class="diag-chips" id="atypChips"></div>
                <div class="diag-table-wrap">
                  <div class="diag-table-inner">
                    <table class="mini-table" id="atypTable">
                      <thead>
                        <tr>
                          <th>Residue</th>
                          <th>Ref</th>
                          <th>Avg</th>
                          <th>|Δ|</th>
                          <th>Class</th>
                        </tr>
                      </thead>
                      <tbody></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div class="diag-panel">
              <div class="diag-panel-header">
                <div class="diag-panel-title">Disulfide bridges</div>
                <div class="diag-panel-sub">Cys–Cys pairs</div>
              </div>
              <div class="diag-panel-body">
                <div class="diag-chips" id="ssChips"></div>
                <div class="diag-table-wrap">
                  <div class="diag-table-inner">
                    <table class="mini-table" id="ssTable">
                      <thead>
                        <tr>
                          <th>Cys</th>
                          <th>Partner</th>
                          <th>Chain</th>
                        </tr>
                      </thead>
                      <tbody></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>
    </div>

  </div>
</div>

{footer_html}

<div class="interaction-modal" id="interactionModal" role="dialog" aria-modal="true" aria-labelledby="interactionTitle">
  <div class="interaction-dialog">
    <div class="interaction-header">
      <strong id="interactionTitle">2D interactions</strong>
      <div class="interaction-actions">
        <button id="downloadInteractionSvg" type="button">Download SVG</button>
        <button id="closeInteractionModal" type="button">Close</button>
      </div>
    </div>
    <div class="interaction-canvas" id="interactionCanvas"></div>
  </div>
</div>

<script>
const pdbText = {pdb_js_literal};
const resData = {js_data};
const summaryStats = {js_summary};

let filteredState = "all";
let searchTerm = "";
let membraneFilter = "all";

const PALETTE_KEY = "prot_dashboard_palette";
let paletteMode = (localStorage.getItem(PALETTE_KEY) || "default");

const THEME_KEY = "prot_dashboard_theme";
let themeMode = (localStorage.getItem(THEME_KEY) || "dark");

const DIAG_KEY = "prot_dashboard_diag";
let diagOn = (localStorage.getItem(DIAG_KEY) || "on") === "on";

const DEFAULT_SCALE = [[0.0, "#ef4444"], [0.5, "#facc15"], [1.0, "#22c55e"]];
const CB_SCALE = [
  [0.00, "#6A00A8"],
  [0.35, "#B12A90"],
  [0.70, "#F1605D"],
  [1.00, "#FEE08B"]
];

let agreementFilter = "all";
let coverageFilter = "all";
let atypFilter = "all";

const SELECTION_KEY = "prot_dashboard_resname_selection_v2_explicit_only";
let selectedMap = {{}};
try {{
  selectedMap = JSON.parse(localStorage.getItem(SELECTION_KEY) || "{{}}") || {{}};
}} catch (e) {{
  selectedMap = {{}};
}}

function rowKey(chain, resnum) {{
  const ch = String(chain || "");
  return ch + ":" + String(resnum);
}}

function getSelected(chain, resnum, fallbackVal) {{
  const key = rowKey(chain, resnum);
  if (selectedMap.hasOwnProperty(key)) return selectedMap[key];
  return fallbackVal;
}}

function setSelected(chain, resnum, val) {{
  const key = rowKey(chain, resnum);
  if (val) selectedMap[key] = val;
  else delete selectedMap[key];
  try {{
    localStorage.setItem(SELECTION_KEY, JSON.stringify(selectedMap));
  }} catch (e) {{
  }}
}}

function fallbackDecision(r) {{
  if (r.Disulfide_bridge) return {{state:"CYX", source:"detected_disulfide"}};
  const fallbacks = {{ASP:"ASP", GLU:"GLU", CYS:"CYS", LYS:"LYS", ARG:"ARG", TYR:"TYR"}};
  const state = fallbacks[String(r.resname || "").toUpperCase()] || "";
  return state ? {{state:state, source:"preserved_canonical_input"}} : {{state:"", source:"unassigned"}};
}}

function decisionForResidue(r) {{
  if (r.Disulfide_bridge) return fallbackDecision(r);
  const explicit = getSelected(r.chain, r.resnum, "");
  if (explicit) return {{state:String(explicit), source:"user"}};
  return fallbackDecision(r);
}}

function downloadSelectionsJson() {{
  const out = [];
  for (let i = 0; i < resData.length; i++) {{
    const r = resData[i];
    const allowed = Array.isArray(r.allowed_resnames) ? r.allowed_resnames : [];
    const decision = decisionForResidue(r);

    out.push({{
      Residue: r.Residue,
      chain: String(r.chain || ""),
      resnum: Number(r.resnum),
      resname_original: String(r.resname || ""),
      allowed_resnames: allowed.slice(),
      selected_resname: decision.state || null,
      decision_source: decision.source,
      predictions: {{
        propka: r.propka_pKa == null ? null : Number(r.propka_pKa),
        pypka: r.pypka_pKa == null ? null : Number(r.pypka_pKa),
        deepka: r.deepka_pKa == null ? null : Number(r.deepka_pKa)
      }},
      structural_context: {{
        exposure: String(r.Exposure || ""),
        in_membrane_slab: !!r.In_membrane_slab,
        ionizable_group_atoms: String(r.Ionizable_group_atoms || "")
      }}
    }});
  }}

  const payload = {{
    schema: "pkaraptor.resname_selection.v2",
    pdb: "{Path(pdb_path).name}",
    ph: summaryStats && summaryStats.ph != null ? Number(summaryStats.ph) : null,
    selections: out
  }};

  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const base = "{Path(pdb_path).name}".replace(/\\.[^.]+$/, "");
  a.download = base + "_pkaraptor_selection.json";
  document.body.appendChild(a);
  a.click();
  setTimeout(function() {{
    URL.revokeObjectURL(a.href);
    document.body.removeChild(a);
  }}, 0);
}}

document.getElementById("downloadJsonBtn").addEventListener("click", function() {{
  downloadSelectionsJson();
}});

function formatPct(frac) {{
  if (frac === null || isNaN(frac)) return "–";
  return (Number(frac) * 100).toFixed(0) + "%";
}}

function fmtPka(val, isSS=false) {{
  if (isSS) return "SS";
  if (val === null || isNaN(val)) return "–";
  return Number(val).toFixed(2);
}}

function passesStateFilter(res) {{
  if (filteredState === "all") return true;
  return res.Protonation_state === filteredState;
}}

function passesMembraneFilter(res) {{
  if (membraneFilter === "all") return true;
  const inMem = !!res.In_membrane_slab;
  if (membraneFilter === "mem") return inMem;
  if (membraneFilter === "nonmem") return !inMem;
  return true;
}}

function passesSearchFilter(res) {{
  if (!searchTerm) return true;
  const identityFields = [
    res.Residue,
    String(res.chain || ""),
    String(res.resnum),
    String(res.resname || ""),
    res.Protonation_state,
    res.Exposure,
    (res.In_membrane_slab ? "membrane membrane-embedded" : "non-membrane"),
    (res.Disulfide_bridge ? "SS disulfide" : ""),
  ].join(" ").toLowerCase().replace(/[^a-z0-9.+-]+/g, " ");
  const tokens = searchTerm.toLowerCase().replace(/[^a-z0-9.+-]+/g, " ").trim().split(/\\s+/).filter(Boolean);
  return tokens.every(function(token) {{ return identityFields.indexOf(token) !== -1; }});
}}

function getFilteredResidues() {{
  const out = [];
  for (let i = 0; i < resData.length; i++) {{
    const r = resData[i];
    if (!passesStateFilter(r)) continue;
    if (!passesMembraneFilter(r)) continue;
    if (!passesSearchFilter(r)) continue;
    out.push(r);
  }}
  out.sort(function(a, b) {{
    const ca = String(a.chain || "");
    const cb = String(b.chain || "");
    if (ca !== cb) return ca.localeCompare(cb);
    return Number(a.resnum) - Number(b.resnum);
  }});
  return out;
}}

function parsePartnerToken(tok) {{
  const s = String(tok || "").trim();
  if (!s) return null;

  const parts = s.split(":");
  let chain = "";
  let mid = "";
  let atom = "";

  if (parts.length === 3) {{
    chain = parts[0];
    mid = parts[1];
    atom = parts[2];
  }} else if (parts.length === 2) {{
    mid = parts[0];
    atom = parts[1];
  }} else {{
    return null;
  }}

  const m = mid.match(/(\\d+)/);
  const resnum = m ? Number(m[1]) : null;
  return {{
    raw: s,
    chain: chain,
    resnum: resnum,
    atom: atom
  }};
}}

function splitPartners(s) {{
  const raw = String(s || "").split(",").map(x => x.trim()).filter(x => x.length > 0);
  const out = [];
  for (let i = 0; i < raw.length; i++) {{
    const p = parsePartnerToken(raw[i]);
    if (p && p.resnum != null) out.push(p);
  }}
  return out;
}}

function filterNonConsecutive(partners, centerResnum, centerChain) {{
  const keep = [];
  for (let i = 0; i < partners.length; i++) {{
    const p = partners[i];
    const sameChain = (centerChain && p.chain) ? (p.chain === centerChain) : true;
    if (!sameChain) continue;
    const diff = Math.abs(Number(p.resnum) - Number(centerResnum));
    if (diff === 1) continue;
    if (diff === 0) continue;
    keep.push(p);
  }}
  return keep;
}}

function hbSummaryForResidue(r) {{
  const donors = filterNonConsecutive(splitPartners(r.HBond_shell_donors || ""), r.resnum, r.chain || "");
  const accs   = filterNonConsecutive(splitPartners(r.HBond_shell_acceptors || ""), r.resnum, r.chain || "");
  return {{ donors: donors, acceptors: accs }};
}}

function parseNeighborsChainAware(s) {{
  const parts = String(s || "").split(",").map(x => x.trim()).filter(Boolean);
  const out = [];
  for (let i = 0; i < parts.length; i++) {{
    const tok = parts[i];
    let chain = "";
    let rest = tok;

    if (tok.indexOf(":") !== -1) {{
      const p = tok.split(":");
      if (p.length >= 2) {{
        chain = String(p[0] || "").trim();
        rest = String(p[p.length - 1] || "").trim();
      }}
    }}

    const m = rest.match(/(\\d+)/);
    if (!m) continue;
    out.push({{ chain: chain, resnum: Number(m[1]) }});
  }}
  return out;
}}

function clearRowSelection() {{
  const rows = document.querySelectorAll("#summary-table tbody tr");
  for (let i = 0; i < rows.length; i++) {{
    rows[i].classList.remove("row-selected");
  }}
}}

function markRowSelected(chain, resnum) {{
  clearRowSelection();
  const key = rowKey(chain, resnum);
  const row = document.querySelector('#summary-table tbody tr[data-key="' + key + '"]');
  if (row) row.classList.add("row-selected");
}}

function paletteLabel() {{
  return paletteMode === "cb" ? "Palette: Colorblind" : "Palette: Default";
}}

function currentBarColor(r) {{
  return paletteMode === "cb" ? (r.color_cb || "#F1605D") : (r.color_default || "#22c55e");
}}

function updatePaletteButton() {{
  const btn = document.getElementById("togglePaletteBtn");
  if (btn) btn.textContent = paletteLabel();
}}

function diagLabel() {{
  return diagOn ? "Diagnostics: on" : "Diagnostics: off";
}}

function updateDiagButton() {{
  const btn = document.getElementById("toggleDiagBtn");
  if (btn) btn.textContent = diagLabel();
}}

function applyDiagVisibility() {{
  const stack = document.getElementById("diagnosticsStack");
  if (!stack) return;
  if (diagOn) stack.classList.remove("hidden");
  else stack.classList.add("hidden");
  updateDiagButton();
}}

function themeLabel() {{
  return themeMode === "light" ? "Theme: Light" : "Theme: Dark";
}}

function updateThemeButton() {{
  const btn = document.getElementById("toggleThemeBtn");
  if (btn) btn.textContent = themeLabel();
}}

function applyTheme() {{
  if (themeMode === "light") {{
    document.body.classList.add("theme-light");
  }} else {{
    document.body.classList.remove("theme-light");
  }}
  updateThemeButton();
  applyThemeToViewer();
  applyThemeToPlot();
}}

function applyPlotPalette() {{
  const plotDivs = document.querySelectorAll(".plotly-graph-div");
  if (!plotDivs || plotDivs.length === 0) return;
  const colorscale = (paletteMode === "cb") ? CB_SCALE : DEFAULT_SCALE;

  for (let i = 0; i < plotDivs.length; i++) {{
    const div = plotDivs[i];
    try {{
      Plotly.restyle(div, {{
        "marker.colorscale": [colorscale],
        "marker.cmin": [0],
        "marker.cmax": [1]
      }});
    }} catch (e) {{
    }}
  }}
}}

function applyThemeToPlot() {{
  const plotDivs = document.querySelectorAll(".plotly-graph-div");
  if (!plotDivs || plotDivs.length === 0) return;

  const isLight = (themeMode === "light");

  const paper = "rgba(0,0,0,0)";
  const plotbg = "rgba(0,0,0,0)";
  const fontColor = isLight ? "rgba(15,23,42,0.95)" : "rgba(229,231,235,0.95)";
  const gridColor = isLight ? "rgba(15,23,42,0.10)" : "rgba(148,163,184,0.18)";

  for (let i = 0; i < plotDivs.length; i++) {{
    const div = plotDivs[i];
    try {{
      Plotly.relayout(div, {{
        "paper_bgcolor": paper,
        "plot_bgcolor": plotbg,
        "font.color": fontColor,
        "xaxis.gridcolor": gridColor,
        "yaxis.gridcolor": gridColor
      }});
    }} catch (e) {{
    }}
  }}
}}

function makeResnameDropdown(r) {{
  const allowed = Array.isArray(r.allowed_resnames) ? r.allowed_resnames : [];
  const def = String(r.default_resname || "");
  const current = getSelected(r.chain, r.resnum, def);

  const sel = document.createElement("select");
  sel.style.width = "100%";
  sel.style.borderRadius = "999px";
  sel.style.padding = "4px 8px";
  sel.style.border = "1px solid rgba(148,163,184,0.55)";
  sel.style.background = "rgba(15,23,42,0.9)";
  sel.style.color = "rgba(229,231,235,0.95)";
  sel.style.fontSize = "0.78rem";
  sel.style.outline = "none";

  if (r.Disulfide_bridge) {{
    const disulfideOption = document.createElement("option");
    disulfideOption.value = "CYX";
    disulfideOption.textContent = "CYX · detected disulfide";
    sel.appendChild(disulfideOption);
    sel.value = "CYX";
    sel.disabled = true;
    sel.title = "Locked structural assignment: this cysteine participates in a detected disulfide bridge.";
    return sel;
  }}

  if (!allowed || allowed.length === 0) {{
    const opt = document.createElement("option");
    opt.value = def;
    opt.textContent = def || "–";
    sel.appendChild(opt);
    sel.disabled = true;
    return sel;
  }}

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "(choose)";
  sel.appendChild(placeholder);

  for (let i = 0; i < allowed.length; i++) {{
    const v = String(allowed[i]);
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }}

  sel.value = allowed.indexOf(current) !== -1 ? current : "";

  sel.addEventListener("click", function(e) {{
    e.stopPropagation();
  }});
  sel.addEventListener("change", function(e) {{
    setSelected(r.chain, r.resnum, e.target.value);
  }});

  return sel;
}}

function renderTable() {{
  const tbody = document.querySelector("#summary-table tbody");
  tbody.innerHTML = "";

  const filtered = getFilteredResidues();
  const searchCount = document.getElementById("searchCount");
  if (searchCount) {{
    searchCount.textContent = filtered.length + " of " + resData.length + " residues";
  }}

  if (filtered.length === 0) {{
    const emptyRow = document.createElement("tr");
    const emptyCell = document.createElement("td");
    emptyCell.colSpan = 9;
    emptyCell.textContent = "No residues match the current search and filters.";
    emptyCell.style.textAlign = "center";
    emptyCell.style.padding = "18px";
    emptyCell.style.color = "#94a3b8";
    emptyRow.appendChild(emptyCell);
    tbody.appendChild(emptyRow);
    return;
  }}

  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];

    const tr = document.createElement("tr");
    tr.setAttribute("data-key", rowKey(r.chain, r.resnum));
    tr.onclick = function() {{ selectResidue(r.resnum, r.chain || ""); }};

    const propkaRaw = r.propka_pKa_raw != null ? Number(r.propka_pKa_raw) : NaN;
    const pypka  = r.pypka_pKa  != null ? Number(r.pypka_pKa)  : NaN;
    const deepka = r.deepka_pKa != null ? Number(r.deepka_pKa) : NaN;
    const avg    = r.avg_pKa    != null ? Number(r.avg_pKa)    : NaN;
    const frac   = r.Protonation_fraction != null ? Number(r.Protonation_fraction) : NaN;

    const hb = hbSummaryForResidue(r);
    const dCount = hb.donors.length;
    const aCount = hb.acceptors.length;

    const tdRes = document.createElement("td");
    tdRes.textContent = r.Residue;
    tr.appendChild(tdRes);

    const tdP1 = document.createElement("td");
    tdP1.textContent = fmtPka(propkaRaw, !!r.Disulfide_bridge);
    tdP1.title = r.Disulfide_bridge ? "Disulfide bridge (PROPKA sentinel present; excluded from average)" : "";
    tr.appendChild(tdP1);

    const tdP2 = document.createElement("td");
    tdP2.textContent = fmtPka(pypka);
    tr.appendChild(tdP2);

    const tdP3 = document.createElement("td");
    tdP3.textContent = fmtPka(deepka);
    tr.appendChild(tdP3);

    const tdAvg = document.createElement("td");
    tdAvg.textContent = fmtPka(avg);
    tr.appendChild(tdAvg);

    const tdSel = document.createElement("td");
    tdSel.appendChild(makeResnameDropdown(r));
    tr.appendChild(tdSel);

    const tdInteractions = document.createElement("td");
    const interactionButton = document.createElement("button");
    interactionButton.type = "button";
    interactionButton.className = "interaction-btn";
    interactionButton.textContent = "View 2D";
    interactionButton.disabled = !Array.isArray(r.interactions) || r.interactions.length === 0;
    interactionButton.addEventListener("click", function(e) {{
      e.stopPropagation();
      openInteractionDiagram(r);
    }});
    tdInteractions.appendChild(interactionButton);
    tr.appendChild(tdInteractions);

    const tdFrac = document.createElement("td");
    const fracLabel = document.createElement("div");
    fracLabel.classList.add("fraction-label");
    fracLabel.textContent = formatPct(frac);
    const bar = document.createElement("div");
    bar.classList.add("fraction-bar");
    const inner = document.createElement("div");
    inner.classList.add("fraction-bar-inner");
    const width = isNaN(frac) ? 0 : Math.max(0, Math.min(1, frac)) * 100;
    inner.style.width = width.toFixed(0) + "%";
    inner.style.background = currentBarColor(r);
    bar.appendChild(inner);
    tdFrac.appendChild(fracLabel);
    tdFrac.appendChild(bar);
    tr.appendChild(tdFrac);

    const tdExp = document.createElement("td");
    const exposureText = document.createElement("div");
    exposureText.textContent = r.Exposure || "–";
    tdExp.appendChild(exposureText);
    if (r.In_membrane_slab) {{
      const membraneFlag = document.createElement("span");
      membraneFlag.className = "membrane-flag";
      membraneFlag.textContent = "Membrane-embedded";
      membraneFlag.title = "User-supplied through --opm-embedded. Membrane embedding does not imply lipid exposure, and SASA accessibility does not imply water exposure.";
      tdExp.appendChild(membraneFlag);
    }}
    tr.appendChild(tdExp);

    tbody.appendChild(tr);
  }}
}}

const SVG_NS = "http://www.w3.org/2000/svg";
let activeInteractionResidue = null;

function svgElement(name, attrs) {{
  const element = document.createElementNS(SVG_NS, name);
  const values = attrs || {{}};
  Object.keys(values).forEach(function(key) {{ element.setAttribute(key, String(values[key])); }});
  return element;
}}

function interactionStyle(type) {{
  const styles = {{
    disulfide: {{color:"#d97706", dash:"", label:"Disulfide"}},
    salt_bridge: {{color:"#dc2626", dash:"7 5", label:"Salt bridge"}},
    potential_hbond: {{color:"#2563eb", dash:"4 5", label:"Potential H-bond"}},
    hydrophobic_contact: {{color:"#7c3aed", dash:"2 5", label:"Hydrophobic"}},
    close_contact: {{color:"#64748b", dash:"2 7", label:"Close contact"}}
  }};
  return styles[type] || styles.close_contact;
}}

function addSvgText(svg, x, y, textValue, attrs) {{
  const textNode = svgElement("text", Object.assign({{x:x, y:y, "text-anchor":"middle", "font-family":"Arial, sans-serif"}}, attrs || {{}}));
  textNode.textContent = textValue;
  svg.appendChild(textNode);
  return textNode;
}}

function residue2DTemplate(resname) {{
  const backbone = {{N:[-62,92], CA:[0,92], C:[62,92], O:[92,60]}};
  const templates = {{
    ASP: {{atoms:{{CB:[0,48],CG:[0,4],OD1:[-40,-30],OD2:[40,-30]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","OD1",2],["CG","OD2",1]]}},
    GLU: {{atoms:{{CB:[0,55],CG:[0,18],CD:[0,-20],OE1:[-40,-52],OE2:[40,-52]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","CD",1],["CD","OE1",2],["CD","OE2",1]]}},
    HIS: {{atoms:{{CB:[0,52],CG:[0,13],ND1:[-38,-12],CE1:[-27,-54],NE2:[27,-54],CD2:[38,-12]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","ND1",1],["ND1","CE1",2],["CE1","NE2",1],["NE2","CD2",2],["CD2","CG",1]]}},
    LYS: {{atoms:{{CB:[0,55],CG:[0,20],CD:[0,-15],CE:[0,-50],NZ:[0,-88]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","CD",1],["CD","CE",1],["CE","NZ",1]]}},
    ARG: {{atoms:{{CB:[0,58],CG:[0,25],CD:[0,-8],NE:[0,-41],CZ:[0,-75],NH1:[-42,-105],NH2:[42,-105]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","CD",1],["CD","NE",1],["NE","CZ",1],["CZ","NH1",2],["CZ","NH2",1]]}},
    CYS: {{atoms:{{CB:[0,45],SG:[0,-4]}},bonds:[["CA","CB",1],["CB","SG",1]]}},
    TYR: {{atoms:{{CB:[0,57],CG:[0,20],CD1:[-38,-2],CE1:[-38,-45],CZ:[0,-67],CE2:[38,-45],CD2:[38,-2],OH:[0,-108]}},bonds:[["CA","CB",1],["CB","CG",1],["CG","CD1",1],["CD1","CE1",2],["CE1","CZ",1],["CZ","CE2",2],["CE2","CD2",1],["CD2","CG",2],["CZ","OH",1]]}}
  }};
  const selected = templates[String(resname || "").toUpperCase()] || {{atoms:{{CB:[0,45]}},bonds:[["CA","CB",1]]}};
  return {{
    atoms:Object.assign({{}},backbone,selected.atoms),
    bonds:[["N","CA",1],["CA","C",1],["C","O",2]].concat(selected.bonds)
  }};
}}

function atomColors(atomName) {{
  const name = String(atomName || "C").toUpperCase();
  const element = name.startsWith("N") ? "N" : (name.startsWith("O") ? "O" : (name.startsWith("S") ? "S" : "C"));
  return {{
    C:{{fill:"#f8fafc",stroke:"#475569",text:"#172033"}},
    N:{{fill:"#dbeafe",stroke:"#2563eb",text:"#1e3a8a"}},
    O:{{fill:"#fee2e2",stroke:"#dc2626",text:"#7f1d1d"}},
    S:{{fill:"#fef3c7",stroke:"#d97706",text:"#78350f"}}
  }}[element];
}}

function drawChemicalBond(svg, a, b, order) {{
  const dx = b.x-a.x;
  const dy = b.y-a.y;
  const length = Math.max(1, Math.sqrt(dx*dx+dy*dy));
  const px = -dy/length*3.2;
  const py = dx/length*3.2;
  (Number(order) === 2 ? [-1,1] : [0]).forEach(function(offset) {{
    svg.appendChild(svgElement("line", {{
      x1:a.x+px*offset, y1:a.y+py*offset, x2:b.x+px*offset, y2:b.y+py*offset,
      stroke:"#334155", "stroke-width":2.4, "stroke-linecap":"round"
    }}));
  }});
}}

function openInteractionDiagram(residue) {{
  activeInteractionResidue = residue;
  const modal = document.getElementById("interactionModal");
  const canvas = document.getElementById("interactionCanvas");
  const title = document.getElementById("interactionTitle");
  title.textContent = residue.Residue + " · atom-level interaction map";
  canvas.innerHTML = "";
  const outputDecision = decisionForResidue(residue);
  if (outputDecision.state) {{
    const warning = document.createElement("div");
    warning.className = "state-map-warning";
    if (outputDecision.source === "detected_disulfide") {{
      warning.textContent = "Detected disulfide: CYX is locked for output. The 2D interaction map remains based on the original input structure.";
    }} else if (outputDecision.source === "user") {{
      warning.textContent = "State selected for output: " + outputDecision.state + ". The 2D interaction map is based on the original input structure and is not recalculated after this change.";
    }} else {{
      warning.textContent = "No manual choice: " + outputDecision.state + " will be used as the defined deprotonated fallback. The 2D interaction map is not recalculated.";
    }}
    canvas.appendChild(warning);
  }}

  const svg = svgElement("svg", {{viewBox:"0 0 840 480", class:"interaction-svg", role:"img", "aria-label":residue.Residue + " atom-level interaction map"}});
  const center = {{x:420,y:238}};
  const records = Array.isArray(residue.interactions) ? residue.interactions : [];
  const typeRank = {{disulfide:0,salt_bridge:1,potential_hbond:2,hydrophobic_contact:3,close_contact:4}};
  const contactMap = {{}};
  records.forEach(function(record) {{
    const key = String(record.partner || "unknown") + "|" + String(record.partner_atom || "?");
    if (!contactMap[key]) contactMap[key] = [];
    contactMap[key].push(record);
  }});
  const allContacts = Object.keys(contactMap).sort(function(a,b) {{
    const ar = Math.min.apply(null,contactMap[a].map(function(item){{return typeRank[item.type]??9;}}));
    const br = Math.min.apply(null,contactMap[b].map(function(item){{return typeRank[item.type]??9;}}));
    if (ar !== br) return ar-br;
    return Math.min.apply(null,contactMap[a].map(function(item){{return Number(item.distance_A);}}))-
      Math.min.apply(null,contactMap[b].map(function(item){{return Number(item.distance_A);}}));
  }});
  const contacts = allContacts.slice(0,10);
  const positions = {{}};
  const leftCount = Math.ceil(contacts.length/2);
  contacts.forEach(function(key,index) {{
    const onLeft = index < leftCount;
    const slot = onLeft ? index : index-leftCount;
    const count = onLeft ? leftCount : contacts.length-leftCount;
    positions[key] = {{x:onLeft?104:736,y:count<=1?center.y:76+slot*(290/(count-1)),onLeft:onLeft}};
  }});

  const template = residue2DTemplate(residue.resname);
  const atomPositions = {{}};
  Object.keys(template.atoms).forEach(function(atomName) {{
    const p = template.atoms[atomName];
    atomPositions[atomName] = {{x:center.x+p[0],y:center.y+p[1]}};
  }});
  template.bonds.forEach(function(bond) {{
    if (atomPositions[bond[0]] && atomPositions[bond[1]]) drawChemicalBond(svg,atomPositions[bond[0]],atomPositions[bond[1]],bond[2]);
  }});

  contacts.forEach(function(key) {{
    const matches = contactMap[key].slice().sort(function(a,b) {{
      return (typeRank[a.type]??9)-(typeRank[b.type]??9) || Number(a.distance_A)-Number(b.distance_A);
    }});
    const primary = matches[0];
    const position = positions[key];
    const style = interactionStyle(primary.type);
    const centralAtom = String(primary.central_atom||"CA").toUpperCase();
    const anchor = atomPositions[centralAtom] || atomPositions.CA || center;
    const startX = position.onLeft ? position.x+72 : position.x-72;
    svg.appendChild(svgElement("line", {{x1:startX,y1:position.y,x2:anchor.x,y2:anchor.y,stroke:style.color,"stroke-width":primary.type==="disulfide"?4:2.5,"stroke-dasharray":style.dash,opacity:0.9}}));
    addSvgText(svg,(startX+anchor.x)/2,(position.y+anchor.y)/2-7,
      String(primary.central_atom||"?")+"–"+String(primary.partner_atom||"?")+" "+Number(primary.distance_A).toFixed(2)+" Å",
      {{"font-size":9.5,fill:style.color,"paint-order":"stroke",stroke:"white","stroke-width":5}});

    const group = svgElement("g",{{class:"interaction-node",tabindex:"0"}});
    group.appendChild(svgElement("rect",{{x:position.x-72,y:position.y-31,width:144,height:62,rx:10,fill:"#f8fafc",stroke:style.color,"stroke-width":2.5}}));
    addSvgText(group,position.x,position.y-9,String(primary.partner||"unknown"),{{"font-size":11.5,"font-weight":700,fill:"#172033"}});
    const colors = atomColors(primary.partner_atom);
    addSvgText(group,position.x,position.y+10,String(primary.partner_atom||"?"),{{"font-size":14,"font-weight":700,fill:colors.text}});
    addSvgText(group,position.x,position.y+24,style.label,{{"font-size":8.5,fill:"#64748b"}});
    group.addEventListener("click",function(){{closeInteractionDiagram();selectResidue(Number(primary.partner_resnum),String(primary.partner_chain||""));}});
    svg.appendChild(group);
  }});

  Object.keys(atomPositions).forEach(function(atomName) {{
    const point = atomPositions[atomName];
    const colors = atomColors(atomName);
    const group = svgElement("g",{{class:"chemical-atom"}});
    group.appendChild(svgElement("circle",{{cx:point.x,cy:point.y,r:15,fill:colors.fill,stroke:colors.stroke,"stroke-width":2.2}}));
    addSvgText(group,point.x,point.y+3.5,atomName,{{"font-size":9.5,"font-weight":700,fill:colors.text}});
    svg.appendChild(group);
  }});

  addSvgText(svg,center.x,24,residue.Residue+" · canonical 2D residue structure",{{"font-size":15,"font-weight":700,fill:"#172033"}});
  addSvgText(svg,center.x,43,"Canonical schematic; contact distances come from the input 3D structure",{{"font-size":9.5,fill:"#64748b"}});
  const shownTypes = new Set();
  contacts.forEach(function(key){{contactMap[key].forEach(function(item){{shownTypes.add(item.type);}});}});
  ["potential_hbond","salt_bridge","hydrophobic_contact","disulfide","close_contact"].filter(function(type){{return shownTypes.has(type);}}).forEach(function(type,index){{
    const style = interactionStyle(type);
    const x = 70+index*155;
    svg.appendChild(svgElement("line",{{x1:x,y1:451,x2:x+42,y2:451,stroke:style.color,"stroke-width":3,"stroke-dasharray":style.dash}}));
    addSvgText(svg,x+91,455,style.label,{{"font-size":11,fill:"#334155"}});
  }});
  if (contacts.length === 0) addSvgText(svg,center.x,405,"No informative contacts are available for this residue.",{{"font-size":14,fill:"#64748b"}});
  if (allContacts.length > contacts.length) addSvgText(svg,center.x,420,"+"+String(allContacts.length-contacts.length)+" weaker atom contacts omitted",{{"font-size":11,fill:"#64748b"}});
  canvas.appendChild(svg);
  modal.classList.add("visible");
}}

function openLegacyInteractionDiagram(residue) {{
  activeInteractionResidue = residue;
  const modal = document.getElementById("interactionModal");
  const canvas = document.getElementById("interactionCanvas");
  const title = document.getElementById("interactionTitle");
  title.textContent = residue.Residue + " · local interaction map";
  canvas.innerHTML = "";

  const svg = svgElement("svg", {{viewBox:"0 0 840 380", class:"interaction-svg", role:"img"}});
  const center = {{x:420, y:165}};
  const records = Array.isArray(residue.interactions) ? residue.interactions : [];
  const typeRank = {{disulfide:0, salt_bridge:1, potential_hbond:2, hydrophobic_contact:3, close_contact:4}};
  const partnerMap = {{}};
  records.forEach(function(record) {{
    const key = String(record.partner || "unknown");
    if (!partnerMap[key]) partnerMap[key] = [];
    partnerMap[key].push(record);
  }});
  const allPartners = Object.keys(partnerMap).sort(function(a, b) {{
    const ar = Math.min.apply(null, partnerMap[a].map(function(item) {{ return typeRank[item.type] ?? 9; }}));
    const br = Math.min.apply(null, partnerMap[b].map(function(item) {{ return typeRank[item.type] ?? 9; }}));
    if (ar !== br) return ar - br;
    const ad = Math.min.apply(null, partnerMap[a].map(function(item) {{ return Number(item.distance_A); }}));
    const bd = Math.min.apply(null, partnerMap[b].map(function(item) {{ return Number(item.distance_A); }}));
    return ad - bd;
  }});
  const partners = allPartners.slice(0, 8);
  const positions = {{}};
  const leftCount = Math.ceil(partners.length / 2);
  partners.forEach(function(partner, index) {{
    const onLeft = index < leftCount;
    const slot = onLeft ? index : index - leftCount;
    const count = onLeft ? leftCount : partners.length - leftCount;
    const y = count <= 1 ? center.y : 60 + slot * (210 / (count - 1));
    positions[partner] = {{x:onLeft ? 130 : 710, y:y, onLeft:onLeft}};
  }});

  partners.forEach(function(partner) {{
    const partnerRecords = partnerMap[partner].slice().sort(function(a, b) {{
      const rankDelta = (typeRank[a.type] ?? 9) - (typeRank[b.type] ?? 9);
      return rankDelta || Number(a.distance_A) - Number(b.distance_A);
    }});
    const primary = partnerRecords[0];
    const position = positions[partner];
    const style = interactionStyle(primary.type);
    const startX = position.onLeft ? position.x + 77 : position.x - 77;
    const endX = position.onLeft ? center.x - 91 : center.x + 91;
    const line = svgElement("line", {{
      x1:startX, y1:position.y, x2:endX, y2:center.y,
      stroke:style.color, "stroke-width":primary.type === "disulfide" ? 4 : 2.5,
      "stroke-dasharray":style.dash, opacity:0.9
    }});
    svg.appendChild(line);
    const midpointX = (startX + endX) / 2;
    const midpointY = (center.y + position.y) / 2;
    const contactText = partnerRecords.slice(0, 2).map(function(record) {{
      return String(record.central_atom || "?") + "–" + String(record.partner_atom || "?") + " " + Number(record.distance_A).toFixed(2) + " Å";
    }}).join(" · ");
    addSvgText(
      svg, midpointX, midpointY - 7, contactText,
      {{"font-size":10, fill:style.color, "paint-order":"stroke", stroke:"white", "stroke-width":5}}
    );
    const sample = primary || {{}};
    const group = svgElement("g", {{class:"interaction-node", tabindex:"0"}});
    group.appendChild(svgElement("rect", {{x:position.x-77, y:position.y-29, width:154, height:58, rx:12, fill:"#f8fafc", stroke:style.color, "stroke-width":2.5}}));
    const line1 = svgElement("text", {{x:position.x, y:position.y - 5, "text-anchor":"middle", "font-family":"Arial", "font-size":13, "font-weight":700, fill:"#172033"}});
    line1.textContent = partner;
    group.appendChild(line1);
    const types = Array.from(new Set(partnerRecords.map(function(item) {{ return interactionStyle(item.type).label; }})));
    const line2 = svgElement("text", {{x:position.x, y:position.y + 14, "text-anchor":"middle", "font-family":"Arial", "font-size":10, fill:"#53657d"}});
    line2.textContent = types.join(" / ");
    group.appendChild(line2);
    group.addEventListener("click", function() {{
      closeInteractionDiagram();
      selectResidue(Number(sample.partner_resnum), String(sample.partner_chain || ""));
    }});
    svg.appendChild(group);
  }});

  const centerGroup = svgElement("g", {{class:"interaction-node"}});
  centerGroup.appendChild(svgElement("rect", {{x:center.x-91, y:center.y-41, width:182, height:82, rx:16, fill:"#183b66", stroke:"#0f2744", "stroke-width":3}}));
  const centerLabel = svgElement("text", {{x:center.x, y:center.y - 7, "text-anchor":"middle", "font-family":"Arial", "font-size":17, "font-weight":700, fill:"white"}});
  centerLabel.textContent = residue.Residue;
  centerGroup.appendChild(centerLabel);
  const atomLabel = svgElement("text", {{x:center.x, y:center.y + 17, "text-anchor":"middle", "font-family":"Arial", "font-size":11, fill:"#dbeafe"}});
  atomLabel.textContent = residue.Ionizable_group_atoms || "ionizable group";
  centerGroup.appendChild(atomLabel);
  svg.appendChild(centerGroup);

  const shownTypes = new Set();
  partners.forEach(function(partner) {{ partnerMap[partner].forEach(function(item) {{ shownTypes.add(item.type); }}); }});
  const legendTypes = ["potential_hbond", "salt_bridge", "hydrophobic_contact", "disulfide", "close_contact"].filter(function(type) {{ return shownTypes.has(type); }});
  legendTypes.forEach(function(type, index) {{
    const style = interactionStyle(type);
    const y = 342;
    const x = 70 + index * 155;
    svg.appendChild(svgElement("line", {{x1:x, y1:y, x2:x+42, y2:y, stroke:style.color, "stroke-width":3, "stroke-dasharray":style.dash}}));
    addSvgText(svg, x+91, y+4, style.label, {{"font-size":11, fill:"#334155"}});
  }});

  if (partners.length === 0) {{
    addSvgText(svg, center.x, 270, "No informative contacts are available for this residue.", {{"font-size":14, fill:"#64748b"}});
  }} else if (allPartners.length > partners.length) {{
    addSvgText(svg, center.x, 304, "+" + String(allPartners.length - partners.length) + " weaker contacts omitted", {{"font-size":11, fill:"#64748b"}});
  }}
  canvas.appendChild(svg);
  modal.classList.add("visible");
}}

function closeInteractionDiagram() {{
  document.getElementById("interactionModal").classList.remove("visible");
}}

function downloadInteractionSvg() {{
  const svg = document.querySelector("#interactionCanvas svg");
  if (!svg || !activeInteractionResidue) return;
  const serialized = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([serialized], {{type:"image/svg+xml"}});
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = String(activeInteractionResidue.Residue || "residue").replace(/[^A-Za-z0-9_-]+/g, "_") + "_interactions.svg";
  document.body.appendChild(anchor);
  anchor.click();
  setTimeout(function() {{ URL.revokeObjectURL(anchor.href); anchor.remove(); }}, 0);
}}

document.getElementById("closeInteractionModal").addEventListener("click", closeInteractionDiagram);
document.getElementById("downloadInteractionSvg").addEventListener("click", downloadInteractionSvg);
document.getElementById("interactionModal").addEventListener("click", function(event) {{
  if (event.target === this) closeInteractionDiagram();
}});

function chip(text, strongVal=null, opts={{}}) {{
  const d = document.createElement("div");
  d.classList.add("diag-chip");
  if (opts.clickable) d.classList.add("clickable");
  if (opts.active) d.classList.add("active");
  if (strongVal === null) {{
    d.textContent = text;
  }} else {{
    d.innerHTML = text + " <strong>" + String(strongVal) + "</strong>";
  }}
  if (opts.onClick) d.addEventListener("click", opts.onClick);
  return d;
}}

function badgeOK(text="✓") {{
  const s = document.createElement("span");
  s.classList.add("badge-ok");
  s.textContent = text;
  return s;
}}

function badgeMiss(text="–") {{
  const s = document.createElement("span");
  s.classList.add("badge-miss");
  s.textContent = text;
  return s;
}}

function badgeSS(text="SS") {{
  const s = document.createElement("span");
  s.classList.add("badge-ss");
  s.textContent = text;
  return s;
}}

function passesAgreementFilter(r) {{
  if (Number(r.methods_present || 0) < 2) return false;
  if (agreementFilter === "all") return true;
  return r.method_agreement === agreementFilter;
}}

function passesCoverageFilter(r) {{
  const n = Number(r.methods_present || 0);
  const hasP = !!r.has_propka;
  const hasPy = !!r.has_pypka;
  const hasD = !!r.has_deepka;

  if (coverageFilter === "all") return true;
  if (coverageFilter === "missing") return !(hasP && hasPy && hasD);
  if (coverageFilter === "n0") return n === 0;
  if (coverageFilter === "n1") return n === 1;
  if (coverageFilter === "n2") return n === 2;
  if (coverageFilter === "n3") return n === 3;
  return true;
}}

function renderDiagnostics() {{
  const filtered = getFilteredResidues();

  // Method agreement is meaningful only when at least two
  // numerical pKa predictions are available.
  const agreementEligible = filtered.filter(
    r => Number(r.methods_present || 0) >= 2
  );

  let nHigh = 0, nMod = 0, nLow = 0;
  let spreads = [];

  for (let i = 0; i < agreementEligible.length; i++) {{
    const r = agreementEligible[i];
    const cls = r.method_agreement;
    if (cls === "High") nHigh++;
    else if (cls === "Moderate") nMod++;
    else if (cls === "Low") nLow++;

    if (r.pka_spread != null && !isNaN(r.pka_spread)) spreads.push(Number(r.pka_spread));
  }}

  let meanSpread = null;
  if (spreads.length > 0) {{
    let sum = 0;
    for (let i = 0; i < spreads.length; i++) sum += spreads[i];
    meanSpread = sum / spreads.length;
  }}

  const agreeChips = document.getElementById("agreeChips");
  agreeChips.innerHTML = "";
  agreeChips.appendChild(chip("All", agreementEligible.length, {{
    clickable: true, active: agreementFilter === "all",
    onClick: function() {{ agreementFilter = "all"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  agreeChips.appendChild(chip("High", nHigh, {{
    clickable: true, active: agreementFilter === "High",
    onClick: function() {{ agreementFilter = "High"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  agreeChips.appendChild(chip("Moderate", nMod, {{
    clickable: true, active: agreementFilter === "Moderate",
    onClick: function() {{ agreementFilter = "Moderate"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  agreeChips.appendChild(chip("Low", nLow, {{
    clickable: true, active: agreementFilter === "Low",
    onClick: function() {{ agreementFilter = "Low"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  agreeChips.appendChild(chip("Mean spread (≥2 methods)", meanSpread === null ? "–" : meanSpread.toFixed(2)));

  const agreeBody = document.querySelector("#agreeTable tbody");
  agreeBody.innerHTML = "";
  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];
    if (Number(r.methods_present || 0) < 2) continue;
    if (!passesAgreementFilter(r)) continue;

    const tr = document.createElement("tr");
    tr.onclick = function() {{ selectResidue(r.resnum, r.chain || ""); }};

    const tdR = document.createElement("td"); tdR.textContent = r.Residue; tr.appendChild(tdR);
    const tdM = document.createElement("td"); tdM.textContent = String(r.methods_present ?? 0); tr.appendChild(tdM);
    const tdS = document.createElement("td");
    tdS.textContent = (r.pka_spread == null || isNaN(r.pka_spread)) ? "–" : Number(r.pka_spread).toFixed(2);
    tr.appendChild(tdS);
    const tdC = document.createElement("td"); tdC.textContent = r.method_agreement || "–"; tr.appendChild(tdC);

    agreeBody.appendChild(tr);
  }}

  const total = filtered.length || 1;
  let cP = 0, cPy = 0, cD = 0, cSS = 0;
  let missingAny = 0;
  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];
    if (r.has_propka) cP++;
    if (r.has_pypka) cPy++;
    if (r.has_deepka) cD++;
    if (r.Disulfide_bridge && r.resname === "CYS") cSS++;
    if (!(r.has_propka && r.has_pypka && r.has_deepka)) missingAny++;
  }}

  const covChips = document.getElementById("covChips");
  covChips.innerHTML = "";
  covChips.appendChild(chip("All", filtered.length, {{
    clickable: true, active: coverageFilter === "all",
    onClick: function() {{ coverageFilter = "all"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  covChips.appendChild(chip("Missing any", missingAny, {{
    clickable: true, active: coverageFilter === "missing",
    onClick: function() {{ coverageFilter = "missing"; renderDiagnostics(); renderAtypical(); renderDisulfides(); }}
  }}));
  covChips.appendChild(chip("PROPKA", ((cP / total) * 100).toFixed(0) + "%"));
  covChips.appendChild(chip("PyPka", ((cPy / total) * 100).toFixed(0) + "%"));
  covChips.appendChild(chip("DeepKa", ((cD / total) * 100).toFixed(0) + "%"));
  covChips.appendChild(chip("SS flagged", cSS));

  const covBody = document.querySelector("#covTable tbody");
  covBody.innerHTML = "";
  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];
    if (!passesCoverageFilter(r)) continue;

    const tr = document.createElement("tr");
    tr.onclick = function() {{ selectResidue(r.resnum, r.chain || ""); }};

    const tdR = document.createElement("td"); tdR.textContent = r.Residue; tr.appendChild(tdR);

    const tdP = document.createElement("td");
    if (r.Disulfide_bridge && r.resname === "CYS" && r.has_propka) tdP.appendChild(badgeSS("SS"));
    else tdP.appendChild(r.has_propka ? badgeOK("✓") : badgeMiss("–"));
    tr.appendChild(tdP);

    const tdPy = document.createElement("td");
    tdPy.appendChild(r.has_pypka ? badgeOK("✓") : badgeMiss("–"));
    tr.appendChild(tdPy);

    const tdD = document.createElement("td");
    tdD.appendChild(r.has_deepka ? badgeOK("✓") : badgeMiss("–"));
    tr.appendChild(tdD);

    const tdN = document.createElement("td");
    tdN.textContent = String(r.methods_present ?? 0);
    tr.appendChild(tdN);

    covBody.appendChild(tr);
  }}
}}

function normalizeForAtypical(val) {{
  const v = Number(val);
  if (!isFinite(v)) return null;
  if (v > 14.0) return null;
  return v;
}}

const REF_PKA = {{
  "ASP": 3.9,
  "GLU": 4.3,
  "HIS": 6.0,
  "CYS": 8.3,
  "TYR": 10.1,
  "LYS": 10.5,
  "ARG": 12.5
}};

function atypicalClass(deltaAbs) {{
  if (!isFinite(deltaAbs)) return "–";
  if (deltaAbs >= 3.0) return "Strong";
  if (deltaAbs >= 2.0) return "Moderate";
  return "Mild";
}}

function passesAtypFilter(deltaAbs) {{
  const cls = atypicalClass(deltaAbs);
  if (atypFilter === "all") return true;
  if (atypFilter === "moderate") return (cls === "Moderate" || cls === "Strong");
  if (atypFilter === "strong") return (cls === "Strong");
  return true;
}}

function renderAtypical() {{
  const filtered = getFilteredResidues();

  const rows = [];
  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];
    const rn = String(r.resname || "").toUpperCase();

    // Exclude disulfide-bonded cysteines from intrinsic-pKa deviation analysis.
    if (r.Disulfide_bridge && rn === "CYS") continue;

    if (!REF_PKA.hasOwnProperty(rn)) continue;

    const ref = REF_PKA[rn];
    const avg = normalizeForAtypical(r.avg_pKa);
    if (avg === null) continue;

    const deltaAbs = Math.abs(avg - ref);
    rows.push({{
      r: r,
      ref: ref,
      avg: avg,
      deltaAbs: deltaAbs,
      cls: atypicalClass(deltaAbs)
    }});
  }}

  rows.sort(function(a, b) {{
    return b.deltaAbs - a.deltaAbs;
  }});

  let nAll = rows.length;
  let nMod = 0;
  let nStrong = 0;
  for (let i = 0; i < rows.length; i++) {{
    if (rows[i].cls === "Moderate") nMod++;
    if (rows[i].cls === "Strong") nStrong++;
  }}

  const atypChips = document.getElementById("atypChips");
  atypChips.innerHTML = "";
  atypChips.appendChild(chip("All", nAll, {{
    clickable: true, active: atypFilter === "all",
    onClick: function() {{ atypFilter = "all"; renderAtypical(); renderDisulfides(); }}
  }}));
  atypChips.appendChild(chip("Moderate", (nMod + nStrong), {{
    clickable: true, active: atypFilter === "moderate",
    onClick: function() {{ atypFilter = "moderate"; renderAtypical(); renderDisulfides(); }}
  }}));
  atypChips.appendChild(chip("Strong", nStrong, {{
    clickable: true, active: atypFilter === "strong",
    onClick: function() {{ atypFilter = "strong"; renderAtypical(); renderDisulfides(); }}
  }}));

  const body = document.querySelector("#atypTable tbody");
  body.innerHTML = "";
  for (let i = 0; i < rows.length; i++) {{
    const item = rows[i];
    if (!passesAtypFilter(item.deltaAbs)) continue;

    const tr = document.createElement("tr");
    tr.onclick = function() {{ selectResidue(item.r.resnum, item.r.chain || ""); }};

    const tdR = document.createElement("td"); tdR.textContent = item.r.Residue; tr.appendChild(tdR);
    const tdRef = document.createElement("td"); tdRef.textContent = Number(item.ref).toFixed(2); tr.appendChild(tdRef);
    const tdAvg = document.createElement("td"); tdAvg.textContent = Number(item.avg).toFixed(2); tr.appendChild(tdAvg);
    const tdD = document.createElement("td"); tdD.textContent = Number(item.deltaAbs).toFixed(2); tr.appendChild(tdD);
    const tdC = document.createElement("td"); tdC.textContent = item.cls; tr.appendChild(tdC);

    body.appendChild(tr);
  }}
}}

function parseDisulfidePartnerLabel(s) {{
  const tok = String(s || "").trim();
  if (!tok) return null;

  let chain = "";
  let rest = tok;

  if (tok.indexOf(":") !== -1) {{
    const parts = tok.split(":");
    if (parts.length >= 2) {{
      chain = String(parts[0] || "").trim();
      rest = String(parts[parts.length - 1] || "").trim();
    }}
  }}

  const m = rest.match(/(\\d+)/);
  if (!m) return null;

  return {{ chain: chain, resnum: Number(m[1]), raw: tok }};
}}

let comp = null;
let cartoonRepr = null;
let focusRepr = null;
let neighborRepr = null;
let structureColorMode = "structure";

let dumRepr = null;
let dumOn = false;
let dumAvailable = false;
let dumSele = null;

const pkaraptorChainColorScheme = NGL.ColormakerRegistry.addScheme(function(params) {{
  this.atomColor = function(atom) {{
    const chain = String(atom.chainname || atom.chainid || "").trim();

    if (chain === "A") return 0xff6a00;
    if (chain === "B") return 0x00a63d;
    if (chain === "C") return 0x8b20d4;

    return 0x475569;
  }};
}}, "pKaRaptor chain colors");

const stage = new NGL.Stage("viewport", {{
  backgroundColor: "#f1f5f9"
}});
// Keep NGL's native hover positioning, but use clean pKaRaptor labels.
stage.signals.hovered.add(function(pickingProxy) {{
  if (!pickingProxy) return;

  window.requestAnimationFrame(function() {{
    const tooltip = stage.tooltip;
    if (!tooltip) return;

    function atomLabel(atom) {{
      if (!atom) return "";

      const chain = String(atom.chainname || atom.chainid || "").trim();
      const resname = String(atom.resname || "").trim();
      const resno = atom.resno != null ? String(atom.resno) : "";
      const atomname = String(atom.atomname || "").trim();

      return (
        (chain ? chain + ":" : "") +
        resname +
        resno +
        (atomname ? " · " + atomname : "")
      );
    }}

    if (pickingProxy.atom) {{
      tooltip.textContent = "atom: " + atomLabel(pickingProxy.atom);
      return;
    }}

    if (pickingProxy.bond) {{
      const bond = pickingProxy.bond;
      const a1 = bond.atom1;
      const a2 = bond.atom2;

      tooltip.textContent =
        "bond: " +
        atomLabel(a1) +
        " — " +
        atomLabel(a2);
    }}
  }});
}});

const colorModeBtn = document.getElementById("colorModeBtn");

if (colorModeBtn) {{
  colorModeBtn.addEventListener("click", function() {{
    setStructureColorMode(
      structureColorMode === "structure" ? "chain" : "structure"
    );
  }});
}}

updateStructureColorButton();

window.addEventListener("resize", function() {{ stage.handleResize(); }}, false);

function applyThemeToViewer() {{
  const bg = (themeMode === "light") ? "#ffffff" : "#f1f5f9";
  try {{
    stage.setParameters({{ backgroundColor: bg }});
  }} catch (e) {{
  }}
}}

function updateStructureColorButton() {{
  const btn = document.getElementById("colorModeBtn");
  if (!btn) return;

  btn.textContent =
    structureColorMode === "chain"
      ? "Coloring: Chain"
      : "Coloring: Structure";
}}

function setStructureColorMode(mode) {{
  structureColorMode = mode === "chain" ? "chain" : "structure";

  if (cartoonRepr) {{
    if (structureColorMode === "chain") {{
      cartoonRepr.setParameters({{
        colorScheme: pkaraptorChainColorScheme
      }});
    }} else {{
      cartoonRepr.setParameters({{
        colorScheme: "uniform",
        colorValue: 0x334155
      }});
    }}
  }}

  updateStructureColorButton();
}}



function safeAutoView(selection) {{
  if (!comp) return;
  try {{ comp.autoView(selection); }}
  catch (e) {{ comp.autoView(); }}
}}

function buildResidueSele(chain, resnum) {{
  const ch = String(chain || "").trim();
  const rn = String(resnum);
  if (ch) {{
    return ":" + ch + " and " + rn;
  }}
  return rn;
}}

function buildNeighborsSele(neighList, fallbackChain="") {{
  const groups = {{}};

  for (let i = 0; i < neighList.length; i++) {{
    const item = neighList[i] || {{}};
    let ch = String(item.chain || "").trim();
    const rn = Number(item.resnum);

    if (!ch) ch = String(fallbackChain || "").trim();
    if (!Number.isFinite(rn)) continue;

    const key = ch;
    if (!groups[key]) groups[key] = [];
    groups[key].push(rn);
  }}

  const parts = [];
  for (const ch in groups) {{
    const uniq = Array.from(new Set(groups[ch].map(x => Number(x)))).sort((a,b) => a-b);
    if (uniq.length === 0) continue;

    const orExpr = uniq.map(n => String(n)).join(" or ");
    if (ch) {{
      parts.push("(:" + ch + " and (" + orExpr + "))");
    }} else {{
      parts.push("(" + orExpr + ")");
    }}
  }}

  if (parts.length === 0) return "none";
  return parts.join(" or ");
}}

function updateDumButton() {{
  const btn = document.getElementById("toggleDumBtn");
  if (!btn) return;

  if (!dumAvailable) {{
    btn.textContent = "Membrane: none";
    btn.disabled = true;
    return;
  }}
  btn.disabled = false;
  btn.textContent = dumOn ? "Membrane: on" : "Membrane: off";
}}

function setDumVisibility(on) {{
  dumOn = !!on;
  if (dumRepr) dumRepr.setVisibility(dumOn);
  updateDumButton();
}}

function buildDumSelectionByScan(o) {{
  const idx = [];
  try {{
    o.structure.eachAtom(function(a) {{
      const an = String(a.atomname || "").toUpperCase();
      const rn = String(a.resname || "").toUpperCase();
      const el = String(a.element || "").toUpperCase();
      if (an.indexOf("DUM") !== -1 || rn.indexOf("DUM") !== -1 || el === "DU") {{
        idx.push(a.index);
      }}
    }});
  }} catch (e) {{
    return null;
  }}

  if (!idx || idx.length === 0) return null;

  const maxN = 8000;
  const use = idx.slice(0, maxN);
  return "@" + use.join(",");
}}

stage.loadFile(new Blob([pdbText], {{ type: "text/plain" }}), {{ ext: "pdb" }}).then(function(o) {{
  comp = o;

  cartoonRepr = o.addRepresentation("cartoon", {{
    colorScheme: "uniform",
    color: "#334155",
    opacity: 0.95,
    quality: "high",
    smoothSheet: true,
    pickable: false,
    flatShaded: true,
    roughness: 1.0,
    metalness: 0.0,
    disableImpostor: false
  }});

  focusRepr = o.addRepresentation("ball+stick", {{
    sele: "none",
    colorScheme: "element",
    scale: 2.1,
    radiusScale: 1.6,
    opacity: 1.0
  }});

  neighborRepr = o.addRepresentation("licorice", {{
    sele: "none",
    colorScheme: "element",
    scale: 0.7,
    radiusScale: 0.7,
    opacity: 0.65
  }});

  dumSele = buildDumSelectionByScan(o);
  dumAvailable = !!dumSele;

  if (dumAvailable) {{
    dumRepr = o.addRepresentation("spacefill", {{
      sele: dumSele,
      opacity: 0.60,
      scale: 0.55,
      color: "#60a5fa"
    }});
    dumRepr.setVisibility(false);
  }}
  updateDumButton();

  stage.setParameters({{
    cameraType: "orthographic",
    lightColor: 0xffffff,
    ambientColor: 0xffffff,
    ambientIntensity: 0.60,
    lightIntensity: 1.05
  }});

  applyThemeToViewer();
  o.autoView();

  stage.signals.clicked.add(function(p) {{
    if (p && p.atom) {{
      const ch = p.atom.chainname || "";
      selectResidue(p.atom.resno, ch);
    }}
  }});
}});

function findResidue(resnum, chain="") {{
  const targetChain = String(chain || "");
  let firstMatch = null;

  for (let i = 0; i < resData.length; i++) {{
    const r = resData[i];
    if (Number(r.resnum) !== Number(resnum)) continue;
    if (!firstMatch) firstMatch = r;
    if (String(r.chain || "") === targetChain) return r;
  }}
  return firstMatch;
}}

function selectDisulfidePair(r) {{
  if (!r || !comp) return;

  const p = parseDisulfidePartnerLabel(r.Disulfide_partner || "");
  const sel1 = buildResidueSele(r.chain || "", r.resnum);

  let sel2 = "none";
  if (p && p.resnum != null) {{
    const ch = p.chain ? p.chain : (r.chain || "");
    sel2 = buildResidueSele(ch, p.resnum);
  }}

  const both = (sel2 !== "none") ? ("(" + sel1 + ") or (" + sel2 + ")") : sel1;
  if (focusRepr) focusRepr.setSelection(both);
  if (neighborRepr) neighborRepr.setSelection("none");
  safeAutoView(both);

  selectResidue(r.resnum, r.chain || "");
}}

function renderDisulfides() {{
  const filtered = getFilteredResidues();

  const pairs = [];
  const seen = new Set();

  for (let i = 0; i < filtered.length; i++) {{
    const r = filtered[i];
    if (!r.Disulfide_bridge) continue;
    if (String(r.resname || "").toUpperCase() !== "CYS") continue;

    const p = parseDisulfidePartnerLabel(r.Disulfide_partner || "");
    if (!p) continue;

    const ch1 = String(r.chain || "");
    const n1 = Number(r.resnum);
    const ch2 = String(p.chain || ch1);
    const n2 = Number(p.resnum);

    const a = ch1 + ":" + n1;
    const b = ch2 + ":" + n2;
    const key = (a < b) ? (a + "|" + b) : (b + "|" + a);
    if (seen.has(key)) continue;
    seen.add(key);

    pairs.push({{ r: r, partner: {{ chain: ch2, resnum: n2, raw: p.raw }} }});
  }}

  const ssChips = document.getElementById("ssChips");
  if (ssChips) {{
    ssChips.innerHTML = "";
    ssChips.appendChild(chip("Pairs", pairs.length));
  }}

  const body = document.querySelector("#ssTable tbody");
  if (!body) return;
  body.innerHTML = "";

  pairs.sort(function(a, b) {{
    const ca = String(a.r.chain || "");
    const cb = String(b.r.chain || "");
    if (ca !== cb) return ca.localeCompare(cb);
    return Number(a.r.resnum) - Number(b.r.resnum);
  }});

  for (let i = 0; i < pairs.length; i++) {{
    const item = pairs[i];

    const tr = document.createElement("tr");
    tr.onclick = function() {{ selectDisulfidePair(item.r); }};

    const td1 = document.createElement("td");
    td1.textContent = String(item.r.Residue || ("CYS" + item.r.resnum));
    tr.appendChild(td1);

    const td2 = document.createElement("td");
    td2.textContent = String(item.partner.raw || (item.partner.chain + ":CYS" + item.partner.resnum));
    tr.appendChild(td2);

    const td3 = document.createElement("td");
    td3.textContent = String(item.r.chain || "–");
    tr.appendChild(td3);

    body.appendChild(tr);
  }}
}}

let comparisonPh = null;
let currentTitrationResidue = null;

setTimeout(function() {{
  initializeComparisonPhControls();
}}, 0);

function getAnalysisPh() {{
  return Number(summaryStats && summaryStats.ph != null ? summaryStats.ph : 7.0);
}}

function getComparisonPh() {{
  return comparisonPh == null ? getAnalysisPh() : Number(comparisonPh);
}}

function updateComparisonPhControls() {{
  const analysisPh = getAnalysisPh();

  if (comparisonPh == null) comparisonPh = analysisPh;

  const slider = document.getElementById("comparisonPhSlider");
  const value = document.getElementById("comparisonPhValue");
  const analysisValue = document.getElementById("analysisPhValue");

  if (slider) slider.value = String(comparisonPh);
  if (value) value.textContent = Number(comparisonPh).toFixed(1);
  if (analysisValue) analysisValue.textContent = analysisPh.toFixed(1);
}}

function showTitrationMessage(message) {{
  const plot = document.getElementById("titrationPlot");
  if (!plot) return;
  try {{ Plotly.purge(plot); }} catch (e) {{}}
  plot.classList.add("titration-empty");
  plot.textContent = message;
}}

function initializeComparisonPhControls() {{
  updateComparisonPhControls();

  const slider = document.getElementById("comparisonPhSlider");
  const resetBtn = document.getElementById("resetComparisonPhBtn");

  if (slider) {{
    slider.addEventListener("input", function() {{
      comparisonPh = Number(slider.value);
      updateComparisonPhControls();

      if (currentTitrationResidue) {{
        renderTitrationPlot(currentTitrationResidue);
      }}
    }});
  }}

  if (resetBtn) {{
    resetBtn.addEventListener("click", function() {{
      comparisonPh = getAnalysisPh();
      updateComparisonPhControls();

      if (currentTitrationResidue) {{
        renderTitrationPlot(currentTitrationResidue);
      }}
    }});
  }}
}}

function renderTitrationPlot(r) {{
  currentTitrationResidue = r;
  updateComparisonPhControls();

  const plot = document.getElementById("titrationPlot");
  if (!plot) return;
  if (!r) {{
    showTitrationMessage("Select a residue in the table or 3D structure to display its titration curves.");
    return;
  }}
  if (r.Disulfide_bridge) {{
    showTitrationMessage("Titration curves are not applicable: this cysteine participates in a detected disulfide bridge.");
    return;
  }}

  const methodValues = [
    {{name:"PROPKA", value:r.propka_pKa, color:"#56B4E9", dash:"solid", symbol:"circle"}},
    {{name:"PyPka", value:r.pypka_pKa, color:"#E69F00", dash:"dash", symbol:"square"}},
    {{name:"DeepKa", value:r.deepka_pKa, color:"#CC79A7", dash:"dot", symbol:"triangle-up"}}
  ].filter(function(item) {{
    const value = Number(item.value);
    return item.value != null && isFinite(value) && value >= 0 && value <= 14;
  }});
  if (methodValues.length === 0) {{
    showTitrationMessage("No valid pKa predictions are available for " + String(r.Residue || "this residue") + ".");
    return;
  }}

  const phValues = [];
  for (let i = 0; i <= 140; i++) phValues.push(i / 10);
  const referencePh = getComparisonPh();
  const traces = [];
  methodValues.forEach(function(method) {{
    const pka = Number(method.value);
    const fractions = phValues.map(function(value) {{ return 1 / (1 + Math.pow(10, value - pka)); }});
    const referenceFraction = 1 / (1 + Math.pow(10, referencePh - pka));
    traces.push({{
      x:phValues, y:fractions, type:"scatter", mode:"lines", name:method.name,
      line:{{color:method.color, width:3, dash:method.dash}},
      customdata:phValues.map(function() {{ return pka; }}),
      hovertemplate:"<b>" + method.name + "</b><br>pKa: %{{customdata:.2f}}<br>pH: %{{x:.1f}}<br>Protonated fraction: %{{y:.1%}}<extra></extra>"
    }});
    traces.push({{
      x:[referencePh], y:[referenceFraction], type:"scatter", mode:"markers", showlegend:false,
      marker:{{color:method.color, size:11, symbol:method.symbol, line:{{color:"#ffffff", width:1.5}}}},
      customdata:[pka],
      hovertemplate:"<b>" + method.name + " at comparison pH</b><br>pKa: %{{customdata:.2f}}<br>pH: %{{x:.2f}}<br>Protonated fraction: %{{y:.1%}}<extra></extra>"
    }});
  }});

  const wasMessage = plot.classList.contains("titration-empty");
  plot.classList.remove("titration-empty");
  if (wasMessage) plot.textContent = "";
  const isLight = themeMode === "light";
  const referenceColor = isLight ? "#334155" : "#f8fafc";
  Plotly.react(plot, traces, {{
    title:{{text:"<b>" + String(r.Residue || "Residue") + "</b> · predicted protonated fraction", x:0.04}},
    height:380,
    margin:{{l:65,r:28,t:65,b:58}},
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{{family:"Inter, Arial, sans-serif", color:isLight?"#0f172a":"#e5e7eb"}},
    xaxis:{{title:"pH", range:[0,14], dtick:1, gridcolor:isLight?"rgba(15,23,42,0.10)":"rgba(148,163,184,0.18)"}},
    yaxis:{{title:"Protonated fraction", range:[0,1], tickformat:".0%", gridcolor:isLight?"rgba(15,23,42,0.10)":"rgba(148,163,184,0.18)"}},
    legend:{{orientation:"h", x:0.5, xanchor:"center", y:1.13}},
    shapes:[
      {{type:"line",x0:referencePh,x1:referencePh,y0:0,y1:1,line:{{color:referenceColor,width:2,dash:"dashdot"}}}},
      {{type:"line",x0:0,x1:14,y0:0.5,y1:0.5,line:{{color:"#94a3b8",width:1,dash:"dot"}}}}
    ],
    annotations:[{{x:referencePh,y:1,xref:"x",yref:"paper",text:"comparison pH " + referencePh.toFixed(2),showarrow:false,yshift:12,font:{{color:referenceColor,size:11}}}}]
  }}, {{responsive:true, displaylogo:false}});
}}

function selectResidue(n, chain="") {{
  const info = document.getElementById("infoBox");
  const r = findResidue(Number(n), chain);
  if (!r) return;
  renderTitrationPlot(r);
  if (!comp) return;

  markRowSelected(r.chain, r.resnum);

  const neigh = parseNeighborsChainAware(r.Neighbors_within_5A || "");

  const focusSel = buildResidueSele(r.chain, r.resnum);
  focusRepr.setSelection(focusSel);

  const neighSel = buildNeighborsSele(neigh, r.chain);
  neighborRepr.setSelection(neighSel);

  safeAutoView(focusSel);

  const hb = hbSummaryForResidue(r);
  const dCount = hb.donors.length;
  const aCount = hb.acceptors.length;
  const donorsList = hb.donors.map(x => x.raw).join(", ") || "–";
  const accList = hb.acceptors.map(x => x.raw).join(", ") || "–";

  const propkaRaw = r.propka_pKa_raw != null ? Number(r.propka_pKa_raw) : NaN;
  const pypka  = r.pypka_pKa  != null ? Number(r.pypka_pKa)  : NaN;
  const deepka = r.deepka_pKa != null ? Number(r.deepka_pKa) : NaN;
  const avg    = r.avg_pKa    != null ? Number(r.avg_pKa)    : NaN;
  const frac   = r.Protonation_fraction != null ? Number(r.Protonation_fraction) : NaN;

  const fracText = formatPct(frac);

  const p1 = fmtPka(propkaRaw, !!r.Disulfide_bridge);
  const p2 = fmtPka(pypka);
  const p3 = fmtPka(deepka);
  const p4 = fmtPka(avg);

  let ssLine = "";
  if (r.Disulfide_bridge) {{
    const partner = String(r.Disulfide_partner || "").trim();
    const partnerTxt = partner ? ("Partner: <b>" + partner + "</b>.") : "";
    ssLine =
      '<div class="info-subtitle">Disulfide</div>' +
      '<div class="hb-list">⚑ Cys flagged as disulfide. ' + partnerTxt + '</div>';
  }}

  const spreadText = (r.pka_spread == null || isNaN(r.pka_spread)) ? "–" : Number(r.pka_spread).toFixed(2);
  const deltaText = (r.delta_pka_ph == null || isNaN(r.delta_pka_ph)) ? "–" : Number(r.delta_pka_ph).toFixed(2);

  const memText = r.In_membrane_slab ? "Embedded (user-supplied)" : "Not flagged";

  info.innerHTML =
    '<div class="info-topbar">' +
      '<div class="chip-residue" style="margin-bottom:0;"><span class="dot"></span><span>' + r.Residue + '</span></div>' +
      '<div class="info-actions">' +
        '<button class="info-min-btn" id="infoMinBtn" type="button" aria-label="Minimize info">' +
          (info.classList.contains("minimized") ? "Expand" : "Minimize") +
        '</button>' +
      '</div>' +
    '</div>' +
    '<div class="info-content">' +
      '<div class="info-grid">' +
        '<div>pKa (PROPKA)</div><b>' + p1 + '</b>' +
        '<div>pKa (PyPka)</div><b>' + p2 + '</b>' +
        '<div>pKa (DeepKa)</div><b>' + p3 + '</b>' +
        '<div>pKa (Avg.)</div><b>' + p4 + '</b>' +
        '<div>Δ(pKa − pH)</div><b>' + deltaText + '</b>' +
        '<div>Chain</div><b>' + String(r.chain || "–") + '</b>' +
        '<div>Methods present</div><b>' + String(r.methods_present ?? 0) + '</b>' +
        '<div>pKa spread</div><b>' + spreadText + '</b>' +
        '<div>Method agreement</div><b>' + (r.method_agreement || "–") + '</b>' +
        '<div>Protonation</div><b>' + r.Protonation_state + '</b>' +
        '<div>Fraction (prot.)</div><b>' + fracText + '</b>' +
        '<div>SASA exposure</div><b>' + (r.Exposure || "–") + '</b>' +
        '<div>Membrane context</div><b>' + memText + '</b>' +
        '<div>Neighbors</div><b>' + (r.Neighbors_within_5A || "–") + '</b>' +
        '<div>HBond shell (filtered)</div><b>D ' + dCount + ' · A ' + aCount + '</b>' +
      '</div>' +
      '<div class="info-subtitle">HBond donors (filtered)</div>' +
      '<div class="hb-list">' + donorsList + '</div>' +
      '<div class="info-subtitle">HBond acceptors (filtered)</div>' +
      '<div class="hb-list">' + accList + '</div>' +
      ssLine +
    '</div>';

  info.classList.add("visible");

  const minBtn = document.getElementById("infoMinBtn");
  if (minBtn) {{
    minBtn.addEventListener("click", function(e) {{
      e.stopPropagation();
      info.classList.toggle("minimized");
      minBtn.textContent = info.classList.contains("minimized") ? "Expand" : "Minimize";
    }});
  }}
}}

function resetViewer() {{
  if (!comp) return;
  if (focusRepr) focusRepr.setSelection("none");
  if (neighborRepr) neighborRepr.setSelection("none");

  clearRowSelection();

  const info = document.getElementById("infoBox");
  if (info) {{
    info.classList.remove("visible");
    info.innerHTML = "";
  }}

  renderTitrationPlot(null);

  safeAutoView();
}}

document.getElementById("resetViewBtn").addEventListener("click", function() {{
  resetViewer();
}});

document.getElementById("toggleDumBtn").addEventListener("click", function() {{
  if (!dumAvailable) return;
  setDumVisibility(!dumOn);
}});

const searchBox = document.getElementById("searchBox");
const clearSearchBtn = document.getElementById("clearSearchBtn");

function updateSearch(value) {{
  searchTerm = String(value || "").trim();
  clearSearchBtn.hidden = searchTerm.length === 0;
  renderTable();
}}

searchBox.addEventListener("input", function(e) {{
  updateSearch(e.target.value);
}});

searchBox.addEventListener("keydown", function(e) {{
  if (e.key === "Escape" && searchBox.value) {{
    e.preventDefault();
    e.stopPropagation();
    searchBox.value = "";
    updateSearch("");
  }}
}});

clearSearchBtn.addEventListener("click", function() {{
  searchBox.value = "";
  updateSearch("");
  searchBox.focus();
}});

const chips = document.querySelectorAll("#stateFilters .filter-chip");
for (let i = 0; i < chips.length; i++) {{
  chips[i].addEventListener("click", function() {{
    for (let j = 0; j < chips.length; j++) chips[j].classList.remove("active");
    this.classList.add("active");
    filteredState = this.getAttribute("data-state");
    renderTable();
    renderDiagnostics();
    renderAtypical();
    renderDisulfides();
  }});
}}

const memChips = document.querySelectorAll("#membraneFilters .filter-chip");
for (let i = 0; i < memChips.length; i++) {{
  memChips[i].addEventListener("click", function() {{
    for (let j = 0; j < memChips.length; j++) memChips[j].classList.remove("active");
    this.classList.add("active");
    membraneFilter = this.getAttribute("data-mem");
    renderTable();
    renderDiagnostics();
    renderAtypical();
    renderDisulfides();
  }});
}}

function toggleGuide() {{
  const card = document.getElementById("guideCard");
  if (!card) return;
  card.classList.toggle("visible");
}}

document.getElementById("toggleGuideBtn").addEventListener("click", function() {{
  toggleGuide();
}});

function togglePalette() {{
  paletteMode = (paletteMode === "cb") ? "default" : "cb";
  localStorage.setItem(PALETTE_KEY, paletteMode);
  updatePaletteButton();
  renderTable();
  renderAtypical();
  renderDisulfides();
  applyPlotPalette();
}}

document.getElementById("togglePaletteBtn").addEventListener("click", function() {{
  togglePalette();
}});

function toggleDiagnostics() {{
  diagOn = !diagOn;
  localStorage.setItem(DIAG_KEY, diagOn ? "on" : "off");
  applyDiagVisibility();
}}

document.getElementById("toggleDiagBtn").addEventListener("click", function() {{
  toggleDiagnostics();
}});

function toggleTheme() {{
  themeMode = (themeMode === "light") ? "dark" : "light";
  localStorage.setItem(THEME_KEY, themeMode);
  applyTheme();
}}

document.getElementById("toggleThemeBtn").addEventListener("click", function() {{
  toggleTheme();
}});

window.addEventListener("keydown", function(e) {{
  if (e.key === "g" || e.key === "G") toggleGuide();
  else if (e.key === "p" || e.key === "P") togglePalette();
  else if (e.key === "t" || e.key === "T") toggleTheme();
  else if (e.key === "Escape") {{
    const interactionModal = document.getElementById("interactionModal");
    if (interactionModal && interactionModal.classList.contains("visible")) closeInteractionDiagram();
    else resetViewer();
  }}
}});

updatePaletteButton();
applyDiagVisibility();
applyTheme();
renderTitrationPlot(null);
renderTable();
renderDiagnostics();
renderAtypical();
renderDisulfides();
applyPlotPalette();
setTimeout(function() {{
  applyPlotPalette();
  applyThemeToPlot();
}}, 250);
</script>
</body>
</html>
"""
    return html_doc
