"""Evaluation protocols (M8). Discipline shared by every protocol:
identical seeded sensor sets for all methods; trusted-pixel metrics on real
data; model selection stays on validation patches (nothing here re-selects)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from tdp.eval.baselines import bicubic_interp, gp_interp, pinn_percase, rbf_interp
from tdp.eval.metrics import field_metrics
from tdp.model.normalization import cond_vector, px_to_xy
from tdp.sim.fdm import bilinear
from tdp.sim.scenarios import BoardState, sample_context, sample_sensors
from tdp.train.finetune import (
    RealCase,
    make_sensor_tensor,
    read_sites,
    sites_px_for,
    source_free_sampler,
)


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def ours_predict_T(model, s_xy, s_vals, amb, aspect, q_xy,
                   card=None, ctx=None, ctx_state=None, chunk=4096) -> np.ndarray:
    """Real-data prediction in °C. card -> board tokens + cond; ctx -> context."""
    sensors, dT = make_sensor_tensor(s_xy, s_vals, amb)
    cond = card.cond(aspect) if card is not None else None
    tokens = card.tokens if card is not None else None
    ctx_t = torch.from_numpy(ctx)[None] if ctx is not None else None
    st_t = torch.from_numpy(ctx_state)[None] if ctx_state is not None else None
    outs = []
    for i in range(0, len(q_xy), chunk):
        q = torch.from_numpy(q_xy[i:i + chunk].astype(np.float32))[None]
        outs.append(model(sensors, q, cond, context=ctx_t, context_state=st_t,
                          board_tokens=tokens)[0].numpy())
    return np.concatenate(outs) * dT + amb


def grid_xy(shape: tuple[int, int]) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return px_to_xy(rr.ravel(), cc.ravel(), n_rows=shape[0], n_cols=shape[1])


def context_from_field(field: np.ndarray, amb: float, n=192, seed=0):
    rng = np.random.default_rng(seed)
    h, w = field.shape
    rows, cols = rng.integers(0, h, n), rng.integers(0, w, n)
    ex = np.clip(field[rows, cols] - amb, 0, None)
    s_ctx = max(ex.max(), 1e-6)
    pts = np.concatenate([px_to_xy(rows, cols, n_rows=h, n_cols=w),
                          (ex / s_ctx)[:, None]], -1).astype(np.float32)
    return pts, np.zeros(n, dtype=np.int64)


@torch.no_grad()
def _synth_predict(model, xy, vals, scn_aspect, cond_vec, q_xy, ctx=None, chunk=8192):
    scale = float(max(vals.max(), 1e-6))
    sensors = torch.from_numpy(
        np.concatenate([xy, (vals / scale)[:, None]], -1).astype(np.float32))[None]
    cond = torch.from_numpy(cond_vec)[None] if cond_vec is not None else None
    ctx_t = torch.from_numpy(ctx[0])[None] if ctx is not None else None
    st_t = torch.from_numpy(ctx[1])[None] if ctx is not None else None
    outs = []
    for i in range(0, len(q_xy), chunk):
        q = torch.from_numpy(q_xy[i:i + chunk].astype(np.float32))[None]
        outs.append(model(sensors, q, cond, context=ctx_t, context_state=st_t)[0].numpy())
    return np.concatenate(outs), scale


def _board_grid(scn: BoardState, stride=4):
    ny, nx = scn.theta.shape
    xx, yy = np.meshgrid(np.linspace(0, scn.aspect, nx), np.linspace(0, 1, ny))
    q = np.stack([xx.ravel(), yy.ravel()], -1)[::stride]
    truth = scn.theta.ravel()[::stride]
    return q, truth


# ---------------------------------------------------------------- synthetic
def synthetic_kcurves(main, twin, boards, ks=(2, 3, 4, 6, 8, 12, 16),
                      n_boards=16, trials=3, seed=1234) -> dict:
    out: dict = {m: {k: [] for k in ks} for m in
                 ("ours_ctx", "ours_noctx", "twin_ctx", "twin_noctx", "rbf", "gp")}
    for k in ks:
        for bi, board in enumerate(boards[:n_boards]):
            scn = board.states[0]
            cond = cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin)
            q_xy, truth = _board_grid(scn)
            for tr in range(trials):
                rng = np.random.default_rng((seed, k, bi, tr))
                xy, vals = sample_sensors(scn, rng, (k, k), layout=None,
                                          layout_prob=0.0, noise_theta_max=0.0)
                ctx = sample_context(board.states[1], rng, 128) \
                    if len(board.states) > 1 else None
                for name, model, use_ctx in (("ours_ctx", main, True),
                                             ("ours_noctx", main, False),
                                             ("twin_ctx", twin, True),
                                             ("twin_noctx", twin, False)):
                    pred, scale = _synth_predict(model, xy, vals, scn.aspect, cond,
                                                 q_xy, ctx=ctx if use_ctx else None)
                    out[name][k].append(float(np.sqrt(np.mean(
                        (pred - truth / scale) ** 2))))
                scale = float(max(vals.max(), 1e-6))
                for name, fn in (("rbf", rbf_interp), ("gp", gp_interp)):
                    pred = fn(xy, vals, q_xy) / scale
                    out[name][k].append(float(np.sqrt(np.mean(
                        (pred - truth / scale) ** 2))))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in out.items()}


def amplitude_ood(main, boards, scales=(1, 2, 4, 8), k=8, n_boards=16, seed=99) -> dict:
    out = {s: [] for s in scales}
    for s in scales:
        for bi, board in enumerate(boards[:n_boards]):
            base = board.states[0]
            scn = BoardState(layout=base.layout, amps=base.amps * s,
                             theta=base.theta * s)
            cond = cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin)
            rng = np.random.default_rng((seed, s, bi))
            xy, vals = sample_sensors(scn, rng, (k, k), layout=None,
                                      layout_prob=0.0, noise_theta_max=0.0)
            ctx_state = board.states[1]
            ctx = sample_context(BoardState(ctx_state.layout, ctx_state.amps * s,
                                            ctx_state.theta * s), rng, 128)
            q_xy, truth = _board_grid(scn)
            pred, scale = _synth_predict(main, xy, vals, scn.aspect, cond, q_xy, ctx=ctx)
            out[s].append(float(np.sqrt(np.mean((pred - truth / scale) ** 2))))
    return {str(s): float(np.mean(v)) for s, v in out.items()}


def layout_ood(main, ood_boards, ref_boards, k=8, seed=55) -> dict:
    def _run(boards, tag_seed):
        errs = []
        for bi, board in enumerate(boards):
            scn = board.states[0]
            cond = cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin)
            rng = np.random.default_rng((seed, tag_seed, bi))
            xy, vals = sample_sensors(scn, rng, (k, k), layout=None,
                                      layout_prob=0.0, noise_theta_max=0.0)
            ctx = sample_context(board.states[1], rng, 128)
            q_xy, truth = _board_grid(scn)
            pred, scale = _synth_predict(main, xy, vals, scn.aspect, cond, q_xy, ctx=ctx)
            errs.append(float(np.sqrt(np.mean((pred - truth / scale) ** 2))))
        return float(np.mean(errs))

    return {"in_dist_1to5_sources": _run(ref_boards[:24], 0),
            "ood_6to7_sources": _run(ood_boards[:24], 1)}


# ---------------------------------------------------------------- real board
def real_headline(model_main, model_twin, cards, case: RealCase, session: str,
                  sites, rects, ctx_field, ctx_amb, pinn_budget=60.0,
                  scratch=None) -> tuple[dict, dict]:
    """All methods on one untouched case. Returns (metrics per method, preds)."""
    spx = sites_px_for(sites, session, case.shape)
    xy, vals = read_sites(case.field, spx)
    q_xy = grid_xy(case.shape)
    ctx, ctx_state = context_from_field(ctx_field, ctx_amb)
    sampler = source_free_sampler(rects, case.aspect)

    preds = {
        "bicubic": bicubic_interp(xy, vals, q_xy),
        "rbf": rbf_interp(xy, vals, q_xy),
        "gp": gp_interp(xy, vals, q_xy),
        "pinn": pinn_percase(xy, vals, q_xy, case.aspect, sampler,
                             budget_s=pinn_budget),
        "zeroshot_noctx": ours_predict_T(model_main, xy, vals, case.amb,
                                         case.aspect, q_xy),
        "zeroshot_ctx": ours_predict_T(model_main, xy, vals, case.amb, case.aspect,
                                       q_xy, ctx=ctx, ctx_state=ctx_state),
        "tier1": ours_predict_T(model_main, xy, vals, case.amb, case.aspect, q_xy,
                                card=cards["main"]),
        "tier1_twin": ours_predict_T(model_twin, xy, vals, case.amb, case.aspect,
                                     q_xy, card=cards["twin"]),
    }
    if scratch is not None:
        preds["tier1_scratch"] = ours_predict_T(scratch[0], xy, vals, case.amb,
                                                case.aspect, q_xy, card=scratch[1])
    metrics = {m: field_metrics(p.reshape(case.shape), case.field, case.trusted)
               for m, p in preds.items()}
    return metrics, preds


def real_k_subsets(model, card, case: RealCase, sites, session="s1",
                   ks=(3, 4, 5, 6), max_subsets=10, seed=7) -> dict:
    spx = sites_px_for(sites, session, case.shape)
    xy_all, vals_all = read_sites(case.field, spx)
    q_xy = grid_xy(case.shape)
    rng = np.random.default_rng(seed)
    out: dict = {m: {} for m in ("ours_card", "rbf", "gp")}
    for k in ks:
        combos = list(itertools.combinations(range(len(vals_all)), k))
        if len(combos) > max_subsets:
            combos = [combos[i] for i in rng.choice(len(combos), max_subsets,
                                                    replace=False)]
        errs = {m: [] for m in out}
        for idx in combos:
            xy, vals = xy_all[list(idx)], vals_all[list(idx)]
            p = ours_predict_T(model, xy, vals, case.amb, case.aspect, q_xy, card=card)
            errs["ours_card"].append(field_metrics(
                p.reshape(case.shape), case.field, case.trusted)["rmse_trust"])
            for m, fn in (("rbf", rbf_interp), ("gp", gp_interp)):
                p = fn(xy, vals, q_xy).reshape(case.shape)
                errs[m].append(field_metrics(p, case.field, case.trusted)["rmse_trust"])
        for m in out:
            out[m][k] = float(np.mean(errs[m]))
    return out


def loso_sites(model, card, case: RealCase, sites, session: str) -> dict:
    active = [s for s in sites if session in s["trusted_sessions"]]
    spx = sites_px_for(sites, session, case.shape)
    xy_all, vals_all = read_sites(case.field, spx)
    out = {}
    for i, site in enumerate(active):
        keep = [j for j in range(len(active)) if j != i]
        xy, vals = xy_all[keep], vals_all[keep]
        q = xy_all[i:i + 1]
        ours = float(ours_predict_T(model, xy, vals, case.amb, case.aspect, q,
                                    card=card)[0] - vals_all[i])
        rbf = float(rbf_interp(xy, vals, q)[0] - vals_all[i])
        out[site["id"]] = {"ours_card": ours, "rbf": rbf,
                           "true_c": float(vals_all[i])}
    return out


def quasi_static(model, card, npz_path: Path, sites, trusted: np.ndarray,
                 session="s2", every=5) -> dict:
    d = np.load(npz_path)
    T, t_rel, amb = d["T"], d["t_rel_s"], d["t_amb"]
    shape = T.shape[1:]
    q_xy = grid_xy(shape)
    spx = sites_px_for(sites, session, shape)
    rows = {"t_min": [], "rmse": [], "max_err": [], "mean_excess": []}
    for i in range(0, len(T), every):
        field = T[i].astype(np.float64)
        xy, vals = read_sites(field, spx)
        pred = ours_predict_T(model, xy, vals, float(amb[i]), shape[1] / shape[0],
                              q_xy, card=card).reshape(shape)
        m = field_metrics(pred, field, trusted)
        rows["t_min"].append(float(t_rel[i] / 60))
        rows["rmse"].append(m["rmse_trust"])
        rows["max_err"].append(m["max_T_err"])
        rows["mean_excess"].append(float(field.mean() - amb[i]))
    return rows


def diode_consistency(model, card, npz_path: Path, sites, tlog_path: Path,
                      soc_log_json: Path, session="s2", every=3) -> dict:
    d = np.load(npz_path)
    T, epochs, amb = d["T"], d["timestamp_epoch"].astype(np.float64), d["t_amb"]
    shape = T.shape[1:]
    offset = json.loads(soc_log_json.read_text("utf-8"))["offset_s"]
    raw = np.loadtxt(tlog_path, delimiter=",")
    t_diode, T_diode = raw[:, 0] + offset, raw[:, 1]

    active = [s for s in sites if session in s["trusted_sessions"]]
    soc_i = next(i for i, s in enumerate(active) if s["id"] == "soc")
    spx = sites_px_for(sites, session, shape)
    keep = [j for j in range(len(active)) if j != soc_i]

    rows = {"t_min": [], "pred_soc": [], "tape_soc": [], "diode": []}
    for i in range(0, len(T), every):
        if epochs[i] < t_diode[0] or epochs[i] > t_diode[-1]:
            continue
        field = T[i].astype(np.float64)
        xy_all, vals_all = read_sites(field, spx)
        pred = float(ours_predict_T(model, xy_all[keep], vals_all[keep],
                                    float(amb[i]), shape[1] / shape[0],
                                    xy_all[soc_i:soc_i + 1], card=card)[0])
        rows["t_min"].append(float((epochs[i] - epochs[0]) / 60))
        rows["pred_soc"].append(pred)
        rows["tape_soc"].append(float(vals_all[soc_i]))
        rows["diode"].append(float(np.interp(epochs[i], t_diode, T_diode)))
    # constant junction-surface offset fitted on the data itself (reported)
    delta = float(np.median(np.asarray(rows["tape_soc"]) - np.asarray(rows["diode"])))
    rows["diode_offset_c"] = delta
    rows["diode_shifted"] = [v + delta for v in rows["diode"]]
    return rows
