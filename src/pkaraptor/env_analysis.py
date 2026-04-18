# env_analysis.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import mdtraj as md

TITRATABLE_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR"}

WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP3", "SOL"}
MEMBRANE_MARKER_RESNAMES = {"MEM", "DUM"}


def _chain_label(chain: md.Topology.Chain) -> str:
    for attr in ("id", "chain_id", "pdbid", "name"):
        if hasattr(chain, attr):
            v = getattr(chain, attr)
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != "none":
                return s
    if hasattr(chain, "index"):
        return str(getattr(chain, "index"))
    return ""


def _res_label(chain: str, resname: str, resseq: int) -> str:
    chain_s = chain.strip() if chain else ""
    if chain_s:
        return f"{chain_s}:{resname}{resseq}"
    return f"{resname}{resseq}"


def _atom_label(chain: str, resname: str, resseq: int, atomname: str) -> str:
    base = _res_label(chain, resname, resseq)
    return f"{base}:{atomname}"


def _donor_atom(atom: md.Topology.Atom) -> bool:
    if not atom.residue.is_protein:
        return False
    if atom.element is None:
        return False
    sym = str(atom.element.symbol).upper()
    if sym not in {"N", "S"}:
        return False

    resn = str(atom.residue.name).upper()
    an = atom.name.upper()

    if an == "N":
        return resn != "PRO"

    if resn == "LYS" and an == "NZ":
        return True
    if resn == "ARG" and an in {"NE", "NH1", "NH2"}:
        return True
    if resn == "HIS" and an in {"ND1", "NE2"}:
        return True
    if resn == "TRP" and an == "NE1":
        return True
    if resn == "ASN" and an == "ND2":
        return True
    if resn == "GLN" and an == "NE2":
        return True
    if resn == "SER" and an == "OG":
        return True
    if resn == "THR" and an == "OG1":
        return True
    if resn == "TYR" and an == "OH":
        return True
    if resn == "CYS" and an == "SG":
        return True

    return False


def _acceptor_atom(atom: md.Topology.Atom) -> bool:
    if not atom.residue.is_protein:
        return False
    if atom.element is None:
        return False
    sym = str(atom.element.symbol).upper()

    resn = str(atom.residue.name).upper()
    an = atom.name.upper()

    if sym == "O":
        if an in {"O", "OXT"}:
            return True
        if resn in {"SER", "THR", "TYR"} and an in {"OG", "OG1", "OH"}:
            return True
        return True

    if sym == "S":
        if resn == "MET" and an == "SD":
            return True
        if resn == "CYS" and an == "SG":
            return True
        return False

    if sym == "N":
        if an == "N":
            return False
        if resn == "HIS" and an in {"ND1", "NE2"}:
            return True
        if resn in {"LYS", "ARG"}:
            return False
        return False

    return False


def _is_water_res(res: md.Topology.Residue) -> bool:
    return str(res.name).strip().upper() in WATER_RESNAMES


def _is_membrane_marker_res(res: md.Topology.Residue) -> bool:
    return str(res.name).strip().upper() in MEMBRANE_MARKER_RESNAMES


def _res_key(res: md.Topology.Residue) -> Tuple[str, int, str]:
    chain = _chain_label(res.chain)
    return (chain, int(res.resSeq), str(res.name).strip().upper())


def clean_pka_columns(
    df: pd.DataFrame,
    cols: List[str],
    min_valid: float = 0.0,
    max_valid: float = 14.0,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[(out[c] <= float(min_valid)) | (out[c] >= float(max_valid)), c] = np.nan
    return out


def _compute_disulfides(
    top: md.Topology,
    xyz_nm: np.ndarray,
    *,
    sg_cutoff_A: float = 2.35,
) -> tuple[dict[int, bool], dict[int, str]]:
    """
    Detect disulfide bonds by SG–SG distance.

    Returns:
      disulfide_by_res_index: res.index -> bool
      partner_label_by_res_index: res.index -> "A:CYS12" style label of partner(s), comma-separated
    """
    sg_atoms: dict[int, int] = {}  # res.index -> atom.index (SG)
    for res in top.residues:
        if not res.is_protein:
            continue
        if str(res.name).strip().upper() != "CYS":
            continue
        for a in res.atoms:
            if a.name.upper() == "SG":
                sg_atoms[res.index] = a.index
                break

    disulfide = {ri: False for ri in sg_atoms.keys()}
    partners: dict[int, list[int]] = {ri: [] for ri in sg_atoms.keys()}

    res_ids = sorted(sg_atoms.keys())
    if len(res_ids) < 2:
        return {}, {}

    cutoff_nm = float(sg_cutoff_A) / 10.0

    for ii in range(len(res_ids)):
        for jj in range(ii + 1, len(res_ids)):
            ri = res_ids[ii]
            rj = res_ids[jj]
            ai = sg_atoms[ri]
            aj = sg_atoms[rj]
            d = float(np.linalg.norm(xyz_nm[ai] - xyz_nm[aj]))
            if d <= cutoff_nm:
                disulfide[ri] = True
                disulfide[rj] = True
                partners[ri].append(rj)
                partners[rj].append(ri)

    partner_labels: dict[int, str] = {}
    for ri, plist in partners.items():
        if not plist:
            continue
        r = top.residue(ri)
        chain = _chain_label(r.chain)
        labels = []
        for pj in sorted(set(plist)):
            rr = top.residue(pj)
            ch2 = _chain_label(rr.chain)
            labels.append(_res_label(ch2, str(rr.name).strip().upper(), int(rr.resSeq)))
        partner_labels[ri] = ", ".join(labels)

    return disulfide, partner_labels


def compute_environment_features(
    pdb_path: str,
    opm_id: Optional[str] = None,
    opm_zmin_A: Optional[float] = None,
    opm_zmax_A: Optional[float] = None,
    neighbors_cutoff_A: float = 5.0,
    neighbors_scheme: str = "closest",
    sasa_include_waters: bool = False,
    neighbors_include_waters: bool = True,
    neighbors_include_membrane_markers: bool = False,
    disulfide_sg_cutoff_A: float = 2.35,
) -> pd.DataFrame:
    traj = md.load_pdb(pdb_path)
    top = traj.topology

    if neighbors_scheme not in {"closest", "closest-heavy"}:
        raise ValueError("neighbors_scheme must be 'closest' or 'closest-heavy'")

    xyz_nm = traj.xyz[0]

    # Disulfide detection (geometry)
    ss_by_res_index, ss_partner_by_res_index = _compute_disulfides(
        top, xyz_nm, sg_cutoff_A=float(disulfide_sg_cutoff_A)
    )

    sasa_by_res_index: Dict[int, float] = {}

    if sasa_include_waters:
        sasa_res = md.shrake_rupley(traj, mode="residue")[0]
        for res in top.residues:
            sasa_by_res_index[res.index] = float(sasa_res[res.index])
    else:
        prot_atoms = [a.index for a in top.atoms if a.residue.is_protein]
        traj_prot = traj.atom_slice(prot_atoms)
        top_prot = traj_prot.topology

        sasa_prot = md.shrake_rupley(traj_prot, mode="residue")[0]
        sasa_by_key: Dict[Tuple[str, int, str], float] = {}
        for r2 in top_prot.residues:
            sasa_by_key[_res_key(r2)] = float(sasa_prot[r2.index])

        for res in top.residues:
            if not res.is_protein:
                continue
            k = _res_key(res)
            if k in sasa_by_key:
                sasa_by_res_index[res.index] = float(sasa_by_key[k])

    rep_atom_indices: List[int] = []
    for res in top.residues:
        ca_atoms = [a.index for a in res.atoms if a.name == "CA"]
        if ca_atoms:
            rep_atom_indices.append(ca_atoms[0])
        else:
            rep_atom_indices.append(next(res.atoms).index)

    rep_xyz_nm = xyz_nm[rep_atom_indices]
    rep_z_A = rep_xyz_nm[:, 2] * 10.0

    cutoff_nm = float(neighbors_cutoff_A) / 10.0

    neighbors: Dict[int, List[str]] = defaultdict(list)
    neighbor_res_indices: Dict[int, Set[int]] = defaultdict(set)

    residues_all = list(top.residues)
    n_res = len(residues_all)
    pairs: List[Tuple[int, int]] = [(i, j) for i in range(n_res) for j in range(i + 1, n_res)]
    dists_nm, _ = md.compute_contacts(traj, contacts=pairs, scheme=neighbors_scheme)

    def keep_neighbor_pair(r: md.Topology.Residue) -> bool:
        if _is_membrane_marker_res(r) and (not neighbors_include_membrane_markers):
            return False
        if _is_water_res(r) and (not neighbors_include_waters):
            return False
        return True

    for k, (i, j) in enumerate(pairs):
        if float(dists_nm[0, k]) > cutoff_nm:
            continue

        res_i = top.residue(i)
        res_j = top.residue(j)

        if not keep_neighbor_pair(res_i) or not keep_neighbor_pair(res_j):
            continue

        chain_i = _chain_label(res_i.chain)
        chain_j = _chain_label(res_j.chain)

        lab_j = _res_label(chain_j, str(res_j.name).strip().upper(), int(res_j.resSeq))
        lab_i = _res_label(chain_i, str(res_i.name).strip().upper(), int(res_i.resSeq))

        neighbors[res_i.index].append(lab_j)
        neighbors[res_j.index].append(lab_i)

        neighbor_res_indices[res_i.index].add(res_j.index)
        neighbor_res_indices[res_j.index].add(res_i.index)

    donors_by_res: Dict[int, List[md.Topology.Atom]] = defaultdict(list)
    acceptors_by_res: Dict[int, List[md.Topology.Atom]] = defaultdict(list)

    for atom in top.atoms:
        if not atom.residue.is_protein:
            continue
        if atom.element is None:
            continue
        if _donor_atom(atom):
            donors_by_res[atom.residue.index].append(atom)
        if _acceptor_atom(atom):
            acceptors_by_res[atom.residue.index].append(atom)

    rows = []
    for res in top.residues:
        resname = str(res.name).strip().upper()
        if resname not in TITRATABLE_RESIDUES:
            continue

        resseq = int(res.resSeq)
        chain = _chain_label(res.chain)
        label = _res_label(chain, resname, resseq)

        sasa_value = float(sasa_by_res_index.get(res.index, np.nan))
        neighbor_list = neighbors.get(res.index, [])
        neighbors_str = ", ".join(neighbor_list)

        z_val_A = float(rep_z_A[res.index]) if res.index < len(rep_z_A) else np.nan

        in_mem = False
        if opm_zmin_A is not None and opm_zmax_A is not None and not np.isnan(z_val_A):
            zlo = min(opm_zmin_A, opm_zmax_A)
            zhi = max(opm_zmin_A, opm_zmax_A)
            in_mem = (z_val_A >= zlo) and (z_val_A <= zhi)

        shell_res_idx = neighbor_res_indices.get(res.index, set())

        donors_shell: List[str] = []
        acceptors_shell: List[str] = []

        for ridx in sorted(shell_res_idx):
            nres = top.residue(ridx)
            if not nres.is_protein:
                continue

            nchain = _chain_label(nres.chain)
            nresn = str(nres.name).strip().upper()
            nresseq = int(nres.resSeq)

            for a in donors_by_res.get(ridx, []):
                donors_shell.append(_atom_label(nchain, nresn, nresseq, a.name))
            for a in acceptors_by_res.get(ridx, []):
                acceptors_shell.append(_atom_label(nchain, nresn, nresseq, a.name))

        rows.append(
            {
                "Residue": label,
                "chain": chain,
                "resname": resname,
                "resnum": resseq,
                "SASA": sasa_value,
                "Neighbors_within_5A": neighbors_str,
                "HBond_shell_donors_count": int(len(donors_shell)),
                "HBond_shell_acceptors_count": int(len(acceptors_shell)),
                "HBond_shell_donors": ", ".join(donors_shell),
                "HBond_shell_acceptors": ", ".join(acceptors_shell),
                "z_CA_A": z_val_A,
                "In_membrane_slab": bool(in_mem),
                "OPM_id": opm_id or "",
                "OPM_zmin_A": float(opm_zmin_A) if opm_zmin_A is not None else np.nan,
                "OPM_zmax_A": float(opm_zmax_A) if opm_zmax_A is not None else np.nan,
                "Disulfide_bridge": bool(ss_by_res_index.get(res.index, False)),
                "Disulfide_partner": str(ss_partner_by_res_index.get(res.index, "")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["SASA_frac"] = df.groupby("resname")["SASA"].transform(
        lambda x: x / (x.max() if x.max() > 0 else 1.0)
    )

    df["Exposure"] = np.where(
        df["SASA_frac"] >= 0.66,
        "solvent-exposed",
        np.where(df["SASA_frac"] >= 0.33, "partially exposed", "buried"),
    )

    pka_cols = ["propka_pKa", "pypka_pKa", "deepka_pKa", "Effective_pKa", "Average_pKa"]
    df = clean_pka_columns(df, pka_cols, min_valid=0.0, max_valid=14.0)

    return df