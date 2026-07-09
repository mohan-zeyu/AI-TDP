"""Domain-randomized boards, operating states, and sensor sampling (v2.5).

A *board* = layout (source positions/shapes on a rectangular domain) + physics
(ĥ, BC, γ). A board has several *states*: the same layout with independently
re-drawn per-source amplitudes (occasionally a source switched off) — mimicking
workload changes (idle / half / full / USB-load). All states of a board share
one LU factorization, and the linear PDE makes each state θ = Σᵢ aᵢ·φᵢ(x):
in-context conditioning asks the model to read the basis {φᵢ} from context
frames and the coefficients {aᵢ} from the live sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdp.model.normalization import BOARD_ASPECT
from tdp.sim.fdm import bilinear, factorize_steady, grid_coords, solve_with


@dataclass
class ScenarioConfig:
    ny: int = 96                       # grid rows; nx = max(round(ny*aspect), 12)
    aspect_range: tuple = (0.5, 1.0)
    h_hat_logrange: tuple = (0.3, 30.0)   # M2: measured upper bound ĥ ≲ 3 on the Pi 4B
    robin_prob: float = 0.7
    gamma_logrange: tuple = (0.1, 10.0)
    n_sources: tuple = (1, 5)
    gauss_prob: float = 0.7            # else smooth rectangle
    sigma_range: tuple = (0.04, 0.15)  # major axis of Gaussian sources
    sigma_ratio: tuple = (0.4, 1.0)    # minor/major
    rect_half: tuple = (0.03, 0.15)    # half-extents of rectangle sources
    rect_edge: float = 0.012           # tanh edge width (smooth for autograd PDE)
    rel_amp: tuple = (0.2, 1.0)        # per-source amplitude draw, per state
    base_amp: float = 200.0
    margin: float = 0.12               # source centers stay this far (fractional) from edges
    states_per_board: int = 3
    source_off_prob: float = 0.15      # per state, per source (≥1 stays on)


@dataclass
class Source:
    """Unit-amplitude source shape; per-state amplitudes scale it."""

    kind: str          # "gauss" | "rect"
    cx: float
    cy: float
    p1: float          # gauss: sigma_major | rect: half-width x
    p2: float          # gauss: sigma_minor | rect: half-height y
    angle: float = 0.0
    edge: float = 0.012

    def q_unit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.kind == "gauss":
            ca, sa = np.cos(self.angle), np.sin(self.angle)
            u = ca * (x - self.cx) + sa * (y - self.cy)
            v = -sa * (x - self.cx) + ca * (y - self.cy)
            return np.exp(-0.5 * ((u / self.p1) ** 2 + (v / self.p2) ** 2))
        sx = 0.5 * (np.tanh((x - (self.cx - self.p1)) / self.edge)
                    - np.tanh((x - (self.cx + self.p1)) / self.edge))
        sy = 0.5 * (np.tanh((y - (self.cy - self.p2)) / self.edge)
                    - np.tanh((y - (self.cy + self.p2)) / self.edge))
        return sx * sy


@dataclass
class BoardLayout:
    aspect: float
    h_hat: float
    robin: bool
    gamma: float | None
    sources: list[Source] = field(default_factory=list)


@dataclass
class BoardState:
    """One operating state of a board. Delegates layout properties so existing
    code (sensor sampling, k-curves, tests) treats it like the old Scenario."""

    layout: BoardLayout
    amps: np.ndarray          # (n_sources,) absolute amplitudes
    theta: np.ndarray         # (ny, nx) float32

    @property
    def aspect(self) -> float:
        return self.layout.aspect

    @property
    def h_hat(self) -> float:
        return self.layout.h_hat

    @property
    def robin(self) -> bool:
        return self.layout.robin

    @property
    def gamma(self) -> float | None:
        return self.layout.gamma

    def q_at(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[..., 0], xy[..., 1]
        out = np.zeros_like(x, dtype=np.float64)
        for a, s in zip(self.amps, self.layout.sources):
            out += a * s.q_unit(x, y)
        return out


@dataclass
class Board:
    layout: BoardLayout
    states: list[BoardState]


def _random_layout(rng: np.random.Generator, scfg: ScenarioConfig,
                   aspect: float | None) -> BoardLayout:
    if aspect is None:
        aspect = float(rng.uniform(*scfg.aspect_range))
    h_hat = float(np.exp(rng.uniform(*np.log(scfg.h_hat_logrange))))
    robin = bool(rng.random() < scfg.robin_prob)
    gamma = float(np.exp(rng.uniform(*np.log(scfg.gamma_logrange)))) if robin else None
    m = scfg.margin
    sources = []
    for _ in range(rng.integers(scfg.n_sources[0], scfg.n_sources[1] + 1)):
        cx = float(rng.uniform(m * aspect, (1 - m) * aspect))
        cy = float(rng.uniform(m, 1 - m))
        if rng.random() < scfg.gauss_prob:
            s1 = float(rng.uniform(*scfg.sigma_range))
            sources.append(Source("gauss", cx, cy, s1, s1 * float(rng.uniform(*scfg.sigma_ratio)),
                                  angle=float(rng.uniform(0, np.pi))))
        else:
            sources.append(Source("rect", cx, cy,
                                  float(rng.uniform(*scfg.rect_half)),
                                  float(rng.uniform(*scfg.rect_half)),
                                  edge=scfg.rect_edge))
    return BoardLayout(aspect=aspect, h_hat=h_hat, robin=robin, gamma=gamma, sources=sources)


def _random_amps(rng: np.random.Generator, n: int, scfg: ScenarioConfig) -> np.ndarray:
    amps = scfg.base_amp * rng.uniform(*scfg.rel_amp, size=n)
    off = rng.random(n) < scfg.source_off_prob
    if off.all():
        off[rng.integers(n)] = False  # at least one source active
    amps[off] = 0.0
    return amps


def random_board(rng: np.random.Generator, scfg: ScenarioConfig,
                 aspect: float | None = None, n_states: int | None = None) -> Board:
    layout = _random_layout(rng, scfg, aspect)
    n_states = scfg.states_per_board if n_states is None else n_states
    nx = max(int(round(scfg.ny * layout.aspect)), 12)
    X, Y = grid_coords(scfg.ny, nx, layout.aspect)
    unit_grids = np.stack([s.q_unit(X, Y) for s in layout.sources])
    lu, boundary = factorize_steady(scfg.ny, nx, layout.aspect, layout.h_hat,
                                    bc="robin" if layout.robin else "dirichlet",
                                    gamma=layout.gamma if layout.robin else 0.0)
    states = []
    for _ in range(n_states):
        amps = _random_amps(rng, len(layout.sources), scfg)
        q_grid = np.tensordot(amps, unit_grids, axes=1)
        theta = solve_with(lu, boundary, q_grid)
        states.append(BoardState(layout=layout, amps=amps, theta=theta.astype(np.float32)))
    return Board(layout=layout, states=states)


def random_scenario(rng: np.random.Generator, scfg: ScenarioConfig,
                    aspect: float | None = None) -> BoardState:
    """Single-state convenience (tests, no-context paths)."""
    return random_board(rng, scfg, aspect=aspect, n_states=1).states[0]


def generate_boards(n: int, scfg: ScenarioConfig, seed: int,
                    board_aspect_frac: float = 0.25) -> list[Board]:
    """n boards (each with scfg.states_per_board states); a fixed fraction pinned
    to the real board's aspect ratio."""
    rng = np.random.default_rng(seed)
    out = []
    every = max(int(1 / max(board_aspect_frac, 1e-9)), 1)
    for i in range(n):
        aspect = BOARD_ASPECT if (i % every == 0) else None
        out.append(random_board(rng, scfg, aspect=aspect))
    return out


# Fractional (u, v) positions of a plausible 8-site sensor layout on the board
# (u along the short side, v along the long side, image convention v↓).
# PLACEHOLDER until configs/sensors_board.json exists (M3): SoC, RAM, USB ctrl,
# PMIC, far corner, and three spread fill-ins, mirroring the Pi 4B floor plan.
PLACEHOLDER_BOARD_LAYOUT = np.array([
    [0.39, 0.61],  # SoC (hotspot ≈ crop px (95, 40))
    [0.68, 0.61],  # RAM
    [0.73, 0.83],  # USB controller area
    [0.12, 0.70],  # PMIC / power area
    [0.10, 0.08],  # far cold corner
    [0.50, 0.25],  # upper mid board
    [0.85, 0.35],  # right edge mid
    [0.33, 0.93],  # bottom center
])


def sample_sensors(
    scn: BoardState,
    rng: np.random.Generator,
    k_range: tuple[int, int] = (4, 16),
    layout: np.ndarray | None = None,
    layout_prob: float = 0.2,
    layout_jitter: float = 0.013,     # ≈2 px on the 157-px-long board
    noise_theta_max: float = 0.10,
    min_peak_frac: float = 0.25,
    max_tries: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sensor positions (K, 2) and their noisy θ readings (K,).

    80% random placement with random K; 20% the (jittered) board sensor-site
    layout. Resamples up to `max_tries` if no sensor sees ≥ min_peak_frac of the
    field peak — sensor layouts are engineered in practice, and a fully-blind
    set makes the per-frame amplitude scale ΔT_s unidentifiable.
    """
    theta_max = float(scn.theta.max())
    use_layout = layout is not None and rng.random() < layout_prob
    for _ in range(max_tries):
        if use_layout:
            uv = layout + rng.normal(0.0, layout_jitter, layout.shape)
            uv = np.clip(uv, 0.005, 0.995)
            xy = np.stack([uv[:, 0] * scn.aspect, uv[:, 1]], axis=-1)
        else:
            k = int(rng.integers(k_range[0], k_range[1] + 1))
            xy = np.stack([rng.uniform(0.01 * scn.aspect, 0.99 * scn.aspect, k),
                           rng.uniform(0.01, 0.99, k)], axis=-1)
        vals = bilinear(scn.theta, xy, scn.aspect)
        if theta_max <= 0 or vals.max() >= min_peak_frac * theta_max:
            break
    sigma = rng.uniform(0.0, noise_theta_max)
    noisy = vals + rng.normal(0.0, sigma, vals.shape) * max(vals.max(), 1e-9)
    return xy.astype(np.float32), noisy.astype(np.float32)


def sample_context(
    state: BoardState,
    rng: np.random.Generator,
    n_points: int,
    frame_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a reference frame into context points (n, 3) = (x, y, θ/s_ctx).

    Context values are normalized by the frame's *own* scale, so context conveys
    the board's field *shape* (the response basis) while the live sensors carry
    the current state's amplitude. frame_idx (0/1) tags which reference state
    the points came from (context type embedding in the model).
    """
    xy = np.stack([rng.uniform(0.005 * state.aspect, 0.995 * state.aspect, n_points),
                   rng.uniform(0.005, 0.995, n_points)], axis=-1)
    vals = bilinear(state.theta, xy, state.aspect)
    s_ctx = float(max(vals.max(), 1e-6))
    return (np.concatenate([xy, (vals / s_ctx)[:, None]], axis=-1).astype(np.float32),
            np.full(n_points, frame_idx, dtype=np.int64))
