from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


_COLOR = ("singlet", "triplet", "adjoint")
_GEN = (0, 1, 2, 3)
_CHIRALITY = ("none", "L", "R")


@dataclass(frozen=True)
class PhysProps:
    spin: float
    charge: float
    t3: float
    hypercharge: float
    color: str
    generation: int
    fermion_number: float
    is_self_conjugate: bool
    chirality: str = "none"


PHYS_VEC_DIM = 1 + 1 + 1 + 1 + len(_COLOR) + len(_GEN) + 1 + 1 + 1 + len(_CHIRALITY) + 1
assert PHYS_VEC_DIM == 18, PHYS_VEC_DIM


def _phys_to_tensor(p: PhysProps) -> torch.Tensor:
    v = torch.zeros(PHYS_VEC_DIM, dtype=torch.float32)
    i = 0
    v[i] = float(p.spin); i += 1
    v[i] = float(p.charge); i += 1
    v[i] = float(p.t3); i += 1
    v[i] = float(p.hypercharge); i += 1
    v[i + _COLOR.index(p.color)] = 1.0; i += len(_COLOR)
    v[i + _GEN.index(p.generation)] = 1.0; i += len(_GEN)
    v[i] = float(p.fermion_number); i += 1
    v[i] = 1.0 if p.spin >= 1 - 1e-9 and abs(p.fermion_number) < 1e-9 else 0.0; i += 1
    v[i] = 1.0 if p.is_self_conjugate else 0.0; i += 1
    v[i + _CHIRALITY.index(p.chirality)] = 1.0; i += len(_CHIRALITY)
    v[i] = 0.0; i += 1
    return v


_UNKNOWN_VEC = torch.zeros(PHYS_VEC_DIM, dtype=torch.float32)
_UNKNOWN_VEC[-1] = 1.0


_QUARK_UP = dict(spin=0.5, charge=+2/3, t3=+0.5, hypercharge=+1/3,
                 color="triplet", fermion_number=+1, is_self_conjugate=False)
_QUARK_DN = dict(spin=0.5, charge=-1/3, t3=-0.5, hypercharge=+1/3,
                 color="triplet", fermion_number=+1, is_self_conjugate=False)
_LEPTON = dict(spin=0.5, charge=-1, t3=-0.5, hypercharge=-1,
               color="singlet", fermion_number=+1, is_self_conjugate=False)
_NEUTRINO = dict(spin=0.5, charge=0, t3=+0.5, hypercharge=-1,
                 color="singlet", fermion_number=+1, is_self_conjugate=False)

PHOTON = PhysProps(spin=1, charge=0, t3=0, hypercharge=0, color="singlet",
                   generation=0, fermion_number=0, is_self_conjugate=True)
GLUON = PhysProps(spin=1, charge=0, t3=0, hypercharge=0, color="adjoint",
                  generation=0, fermion_number=0, is_self_conjugate=True)
W_BOSON = PhysProps(spin=1, charge=+1, t3=+1, hypercharge=0, color="singlet",
                    generation=0, fermion_number=0, is_self_conjugate=False)
Z_BOSON = PhysProps(spin=1, charge=0, t3=0, hypercharge=0, color="singlet",
                    generation=0, fermion_number=0, is_self_conjugate=True)
HIGGS = PhysProps(spin=0, charge=0, t3=-0.5, hypercharge=+1, color="singlet",
                  generation=0, fermion_number=0, is_self_conjugate=True)


U_QUARK = PhysProps(**_QUARK_UP, generation=1)
C_QUARK = PhysProps(**_QUARK_UP, generation=2)
T_QUARK = PhysProps(**_QUARK_UP, generation=3)
D_QUARK = PhysProps(**_QUARK_DN, generation=1)
S_QUARK = PhysProps(**_QUARK_DN, generation=2)
B_QUARK = PhysProps(**_QUARK_DN, generation=3)


ELECTRON = PhysProps(**_LEPTON, generation=1)
MUON = PhysProps(**_LEPTON, generation=2)
TAU = PhysProps(**_LEPTON, generation=3)


NU_E = PhysProps(**_NEUTRINO, generation=1)


def _with_chirality(p: PhysProps, chi: str) -> PhysProps:
    if chi == "R":

        return PhysProps(
            spin=p.spin, charge=p.charge, t3=0.0, hypercharge=2 * p.charge,
            color=p.color, generation=p.generation,
            fermion_number=p.fermion_number,
            is_self_conjugate=p.is_self_conjugate, chirality="R",
        )
    if chi == "L":
        return PhysProps(
            spin=p.spin, charge=p.charge, t3=p.t3, hypercharge=p.hypercharge,
            color=p.color, generation=p.generation,
            fermion_number=p.fermion_number,
            is_self_conjugate=p.is_self_conjugate, chirality="L",
        )
    return p


_QED_TABLE: Dict[str, PhysProps] = {
    "e":  ELECTRON, "mu": MUON, "t":  TAU,
    "u":  U_QUARK, "c":  C_QUARK, "tt": T_QUARK,
    "d":  D_QUARK, "s":  S_QUARK, "b":  B_QUARK,
    "A":  PHOTON,
}

_QCD_TABLE: Dict[str, PhysProps] = {
    "u": U_QUARK, "c": C_QUARK, "t": T_QUARK,
    "d": D_QUARK, "s": S_QUARK, "b": B_QUARK,
    "G": GLUON,
}

_EW_TABLE: Dict[str, PhysProps] = {
    "e":    ELECTRON,
    "e_L":  _with_chirality(ELECTRON, "L"),
    "e_R":  _with_chirality(ELECTRON, "R"),
    "nue_L": _with_chirality(NU_E, "L"),
    "u":    U_QUARK,
    "u_L":  _with_chirality(U_QUARK, "L"),
    "u_R":  _with_chirality(U_QUARK, "R"),
    "d":    D_QUARK,
    "d_L":  _with_chirality(D_QUARK, "L"),
    "d_R":  _with_chirality(D_QUARK, "R"),
    "W":    W_BOSON,
    "Z":    Z_BOSON,
    "A":    PHOTON,
    "h":    HIGGS,
}


TABLES: Dict[str, Dict[str, PhysProps]] = {
    "QED": _QED_TABLE,
    "QCD": _QCD_TABLE,
    "EW":  _EW_TABLE,
}


def phys_props(name: str, model: Optional[str] = None) -> Optional[PhysProps]:
    if model is not None and model in TABLES:
        p = TABLES[model].get(name)
        if p is not None:
            return p


    for m in ("QED", "QCD", "EW"):
        p = TABLES[m].get(name)
        if p is not None:
            return p
    return None


def phys_vec(name: str, antiparticle: bool = False,
             model: Optional[str] = None) -> torch.Tensor:
    p = phys_props(name, model=model)
    if p is None:
        return _UNKNOWN_VEC.clone()

    v = _phys_to_tensor(p)
    if antiparticle and not p.is_self_conjugate:

        v[1] = -v[1]
        v[2] = -v[2]
        v[3] = -v[3]

        fn_idx = 4 + len(_COLOR) + len(_GEN)
        v[fn_idx] = -v[fn_idx]
    return v
