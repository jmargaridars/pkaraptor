from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


IONIZABLE_GROUP_ATOMS: Mapping[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "LYS": ("NZ",),
    "ARG": ("CZ", "NH1", "NH2"),
    "CYS": ("SG",),
    "TYR": ("OH",),
}


def ionizable_group_position_nm(
    resname: str,
    atom_names: Sequence[str],
    atom_positions_nm: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the ionizable-group centroid and a label describing the atoms used."""
    names = [str(name).strip().upper() for name in atom_names]
    xyz = np.asarray(atom_positions_nm, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(names) != len(xyz):
        raise ValueError("atom_names and atom_positions_nm must describe the same N x 3 atoms")
    if not names:
        raise ValueError("at least one atom is required")

    wanted = IONIZABLE_GROUP_ATOMS.get(str(resname).strip().upper(), ())
    selected = [i for i, name in enumerate(names) if name in wanted]
    if selected:
        used = [names[i] for i in selected]
        return xyz[selected].mean(axis=0), ",".join(used)

    if "CA" in names:
        index = names.index("CA")
        return xyz[index].copy(), "CA (fallback)"

    return xyz.mean(axis=0), "all atoms (fallback)"
