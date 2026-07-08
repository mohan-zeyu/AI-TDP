"""Radiometric systematics: what the camera indicates when the surface emissivity
differs from the camera's global emissivity setting (T⁴ gray-body model)."""

from __future__ import annotations


def _t4(t_c: float) -> float:
    return (t_c + 273.15) ** 4


def indicated_temp_c(
    t_true_c: float,
    eps_true: float,
    eps_set: float = 0.93,
    t_refl_c: float = 25.0,
) -> float:
    """Camera-indicated temperature for a surface of true emissivity `eps_true`
    when the camera assumes `eps_set` and reflected temperature `t_refl_c`.

    Radiance balance: eps_true·T_true⁴ + (1−eps_true)·T_refl⁴
                    = eps_set·T_ind⁴ + (1−eps_set)·T_refl⁴
    """
    t_ind4 = (eps_true * _t4(t_true_c) + (eps_set - eps_true) * _t4(t_refl_c)) / eps_set
    return t_ind4 ** 0.25 - 273.15


def tape_bias_c(t_true_c: float, eps_tape: float = 0.95, eps_set: float = 0.93,
                t_refl_c: float = 25.0) -> float:
    """Indicated-minus-true bias on a black-tape surface (ε≈0.95, camera set to 0.93)."""
    return indicated_temp_c(t_true_c, eps_tape, eps_set, t_refl_c) - t_true_c
