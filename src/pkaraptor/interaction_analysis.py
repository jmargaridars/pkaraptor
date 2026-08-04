from __future__ import annotations


ACIDIC_RESIDUES = {"ASP", "GLU"}
BASIC_RESIDUES = {"ARG", "LYS", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR"}


def classify_interaction(
    central_resname: str,
    partner_resname: str,
    central_atom: str,
    partner_atom: str,
    distance_A: float,
    *,
    central_is_donor: bool = False,
    central_is_acceptor: bool = False,
    partner_is_donor: bool = False,
    partner_is_acceptor: bool = False,
) -> str:
    """Classify a local contact for visualization using transparent geometric rules."""
    central_resname = str(central_resname).upper()
    partner_resname = str(partner_resname).upper()
    central_atom = str(central_atom).upper()
    partner_atom = str(partner_atom).upper()
    distance_A = float(distance_A)

    if (
        central_resname == "CYS"
        and partner_resname == "CYS"
        and central_atom == "SG"
        and partner_atom == "SG"
        and distance_A <= 2.35
    ):
        return "disulfide"

    opposite_charges = (
        central_resname in ACIDIC_RESIDUES and partner_resname in BASIC_RESIDUES
    ) or (central_resname in BASIC_RESIDUES and partner_resname in ACIDIC_RESIDUES)
    if opposite_charges and distance_A <= 4.0:
        return "salt_bridge"

    donor_acceptor = (central_is_donor and partner_is_acceptor) or (
        partner_is_donor and central_is_acceptor
    )
    if donor_acceptor and distance_A <= 3.5:
        return "potential_hbond"

    if (
        central_resname in HYDROPHOBIC_RESIDUES
        or partner_resname in HYDROPHOBIC_RESIDUES
    ) and central_atom.startswith("C") and partner_atom.startswith("C"):
        if distance_A <= 4.0:
            return "hydrophobic_contact"

    return "close_contact"
