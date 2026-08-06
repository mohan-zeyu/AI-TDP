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
from tdp.sim.fdm import (
    bilinear,
    factorize_steady,
    factorize_steady_variable_k,
    grid_coords,
    solve_with,
)


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
    source_rotated_prob: float = 0.35
    rel_amp: tuple = (0.2, 1.0)        # per-source amplitude draw, per state
    base_amp: float = 200.0
    margin: float = 0.12               # source centers stay this far (fractional) from edges
    states_per_board: int = 3
    # Disabled by default so the v2 generator remains reproducible.  A
    # heterogeneous continuation config turns these regular inclusions on.
    heterogeneous_prob: float = 0.0
    n_material_regions: tuple = (1, 4)
    material_k_logrange: tuple = (0.15, 6.0)
    material_h_logrange: tuple = (0.3, 3.0)
    material_rc_logrange: tuple = (0.001, 0.08)
    material_zero_rc_prob: float = 0.35
    material_half: tuple = (0.04, 0.18)
    material_source_aligned_prob: float = 0.65
    material_source_offset_prob: float = 0.45
    material_source_offset_max: float = 0.08
    material_rotated_prob: float = 0.15
    material_ellipse_prob: float = 0.20
    independent_boundary_prob: float = 0.0
    boundary_side_factor: tuple = (0.25, 4.0)
    local_contact_prob: float = 0.0
    local_contact_fraction: tuple = (0.08, 0.35)
    local_contact_factor: tuple = (2.0, 20.0)
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
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        u = ca * (x - self.cx) + sa * (y - self.cy)
        v = -sa * (x - self.cx) + ca * (y - self.cy)
        sx = 0.5 * (np.tanh((u + self.p1) / self.edge)
                    - np.tanh((u - self.p1) / self.edge))
        sy = 0.5 * (np.tanh((v + self.p2) / self.edge)
                    - np.tanh((v - self.p2) / self.edge))
        return sx * sy


@dataclass
class MaterialRegion:
    """Piecewise-constant rectangular in-plane conductivity inclusion."""

    cx: float
    cy: float
    half_x: float
    half_y: float
    relative_k: float
    relative_h: float = 1.0
    relative_rc: float = 0.0
    angle: float = 0.0
    shape: str = "rect"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        u = ca * (x - self.cx) + sa * (y - self.cy)
        v = -sa * (x - self.cx) + ca * (y - self.cy)
        if self.shape == "ellipse":
            return (u / self.half_x) ** 2 + (v / self.half_y) ** 2 <= 1.0
        return (np.abs(u) <= self.half_x) & (np.abs(v) <= self.half_y)


@dataclass
class BoundarySegment:
    """Override of the edge heat-transfer coefficient on one side.

    ``side`` is 0/1/2/3 = left/right/top/bottom; start/end are fractional
    positions along that side.  Segments describe clips, cables, or local
    contact with a support without tying the universal model to a board type.
    """

    side: int
    start: float
    end: float
    gamma: float


@dataclass
class BoardLayout:
    aspect: float
    h_hat: float
    robin: bool
    gamma: float | None
    sources: list[Source] = field(default_factory=list)
    materials: list[MaterialRegion] = field(default_factory=list)
    side_gamma: tuple[float, float, float, float] | None = None
    boundary_segments: list[BoundarySegment] = field(default_factory=list)

    def conductivity_at(self, xy: np.ndarray) -> np.ndarray:
        """Relative conductivity; later regions override earlier overlaps."""
        x, y = xy[..., 0], xy[..., 1]
        out = np.ones_like(x, dtype=np.float64)
        for material in self.materials:
            out[material.contains(x, y)] = material.relative_k
        return out

    def sink_at(self, xy: np.ndarray) -> np.ndarray:
        """Relative vertical heat-loss coefficient H(x,y)/H_background."""
        x, y = xy[..., 0], xy[..., 1]
        out = np.ones_like(x, dtype=np.float64)
        for material in self.materials:
            out[material.contains(x, y)] = material.relative_h
        return out

    def material_descriptors(self) -> np.ndarray:
        """(R,10): geometry, K/H/contact resistance, and shape code."""
        return np.asarray([
            [m.cx, m.cy, m.half_x, m.half_y, np.sin(m.angle), np.cos(m.angle),
             np.log(m.relative_k), np.log(m.relative_h), np.log1p(m.relative_rc),
             float(m.shape == "ellipse")]
            for m in self.materials
        ], dtype=np.float32).reshape(-1, 10)

    def contact_resistance_faces(self, X: np.ndarray, Y: np.ndarray):
        """Dimensionless interface resistance on x/y grid faces."""
        rx = np.zeros((X.shape[0], X.shape[1] - 1), dtype=np.float64)
        ry = np.zeros((X.shape[0] - 1, X.shape[1]), dtype=np.float64)
        for material in self.materials:
            if material.relative_rc <= 0:
                continue
            inside = material.contains(X, Y)
            rx = np.maximum(rx, (inside[:, :-1] != inside[:, 1:]) * material.relative_rc)
            ry = np.maximum(ry, (inside[:-1, :] != inside[1:, :]) * material.relative_rc)
        return rx, ry

    def boundary_gamma_arrays(self, ny: int, nx: int) -> dict[str, np.ndarray] | float:
        if not self.robin:
            return 0.0
        base = self.side_gamma or (self.gamma,) * 4
        arrays = {
            "left": np.full(ny, base[0], dtype=np.float64),
            "right": np.full(ny, base[1], dtype=np.float64),
            "top": np.full(nx, base[2], dtype=np.float64),
            "bottom": np.full(nx, base[3], dtype=np.float64),
        }
        names = ("left", "right", "top", "bottom")
        for segment in self.boundary_segments:
            values = arrays[names[segment.side]]
            lo = max(0, int(np.floor(segment.start * (len(values) - 1))))
            hi = min(len(values), int(np.ceil(segment.end * (len(values) - 1))) + 1)
            values[lo:hi] = segment.gamma
        return arrays

    def boundary_gamma_at(self, xy: np.ndarray, side: np.ndarray) -> np.ndarray:
        base = self.side_gamma or ((self.gamma or 0.0),) * 4
        out = np.asarray(base, dtype=np.float64)[side]
        along = np.where(side < 2, xy[..., 1], xy[..., 0] / self.aspect)
        for segment in self.boundary_segments:
            hit = ((side == segment.side) & (along >= segment.start)
                   & (along <= segment.end))
            out[hit] = segment.gamma
        return out

    def boundary_descriptors(self) -> np.ndarray:
        """Straight segment descriptors usable for rectangles or polygons."""
        robin = float(self.robin)
        base = self.side_gamma or ((self.gamma or 1.0),) * 4

        def descriptor(side: int, start: float, end: float, gamma: float):
            if side == 0:
                p0, p1, normal = (0.0, start), (0.0, end), (-1.0, 0.0)
            elif side == 1:
                p0, p1, normal = (self.aspect, start), (self.aspect, end), (1.0, 0.0)
            elif side == 2:
                p0, p1, normal = (start * self.aspect, 0.0), (end * self.aspect, 0.0), (0.0, -1.0)
            else:
                p0, p1, normal = (start * self.aspect, 1.0), (end * self.aspect, 1.0), (0.0, 1.0)
            return [*p0, *p1, *normal, np.log(max(gamma, 1e-8)), robin]

        rows = [descriptor(side, 0.0, 1.0, base[side]) for side in range(4)]
        rows.extend(descriptor(s.side, s.start, s.end, s.gamma)
                    for s in self.boundary_segments)
        return np.asarray(rows, dtype=np.float32).reshape(-1, 8)


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

    def conductivity_at(self, xy: np.ndarray) -> np.ndarray:
        return self.layout.conductivity_at(xy)

    def sink_at(self, xy: np.ndarray) -> np.ndarray:
        return self.layout.sink_at(xy)

    def material_descriptors(self) -> np.ndarray:
        return self.layout.material_descriptors()

    def source_descriptors(self) -> np.ndarray:
        """(S,8) source geometry and per-state relative source strength."""
        if len(self.layout.sources) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        scale = max(float(np.max(self.amps)), 1e-8)
        relative = np.clip(self.amps / scale, 1e-4, None)
        return np.asarray([
            [s.cx, s.cy, s.p1, s.p2, np.sin(s.angle), np.cos(s.angle),
             np.log(a), float(s.kind == "gauss")]
            for s, a in zip(self.layout.sources, relative)
        ], dtype=np.float32).reshape(-1, 8)

    def boundary_descriptors(self) -> np.ndarray:
        return self.layout.boundary_descriptors()

    def boundary_gamma_at(self, xy: np.ndarray, side: np.ndarray) -> np.ndarray:
        return self.layout.boundary_gamma_at(xy, side)


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
    side_gamma = None
    boundary_segments = []
    if robin:
        if rng.random() < scfg.independent_boundary_prob:
            factors = np.exp(rng.uniform(*np.log(scfg.boundary_side_factor), size=4))
            side_gamma = tuple(float(gamma * factor) for factor in factors)
        else:
            side_gamma = (gamma,) * 4
        if rng.random() < scfg.local_contact_prob:
            for _ in range(int(rng.integers(1, 4))):
                side = int(rng.integers(4))
                length = float(rng.uniform(*scfg.local_contact_fraction))
                start = float(rng.uniform(0.0, 1.0 - length))
                factor = float(np.exp(rng.uniform(*np.log(scfg.local_contact_factor))))
                boundary_segments.append(BoundarySegment(
                    side, start, start + length, side_gamma[side] * factor
                ))
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
            angle = (float(rng.uniform(0, np.pi))
                     if rng.random() < scfg.source_rotated_prob else 0.0)
            sources.append(Source("rect", cx, cy,
                                  float(rng.uniform(*scfg.rect_half)),
                                  float(rng.uniform(*scfg.rect_half)),
                                  angle=angle,
                                  edge=scfg.rect_edge))
    materials = []
    if scfg.heterogeneous_prob > 0 and rng.random() < scfg.heterogeneous_prob:
        n_materials = int(rng.integers(
            scfg.n_material_regions[0], scfg.n_material_regions[1] + 1
        ))
        for _ in range(n_materials):
            aligned = bool(sources) and rng.random() < scfg.material_source_aligned_prob
            if aligned:
                source = sources[int(rng.integers(len(sources)))]
                cx, cy = source.cx, source.cy
                half_x = float(np.clip(
                    source.p1 * rng.uniform(0.85, 1.45),
                    scfg.material_half[0],
                    scfg.material_half[1],
                ))
                half_y = float(np.clip(
                    source.p2 * rng.uniform(0.85, 1.45),
                    scfg.material_half[0],
                    scfg.material_half[1],
                ))
                angle = source.angle if source.kind == "gauss" else 0.0
                if rng.random() < scfg.material_source_offset_prob:
                    offset = scfg.material_source_offset_max
                    cx += float(rng.uniform(-offset, offset))
                    cy += float(rng.uniform(-offset, offset))
                    cx = float(np.clip(cx, half_x, aspect - half_x))
                    cy = float(np.clip(cy, half_y, 1.0 - half_y))
            else:
                half_x = float(rng.uniform(*scfg.material_half))
                half_y = float(rng.uniform(*scfg.material_half))
                cx = float(rng.uniform(half_x, aspect - half_x))
                cy = float(rng.uniform(half_y, 1.0 - half_y))
                angle = 0.0
            if rng.random() < scfg.material_rotated_prob:
                angle = float(rng.uniform(0.0, np.pi))
            materials.append(MaterialRegion(
                cx=cx,
                cy=cy,
                half_x=half_x,
                half_y=half_y,
                relative_k=float(np.exp(rng.uniform(*np.log(scfg.material_k_logrange)))),
                relative_h=float(np.exp(rng.uniform(*np.log(scfg.material_h_logrange)))),
                relative_rc=(0.0 if rng.random() < scfg.material_zero_rc_prob else
                             float(np.exp(rng.uniform(*np.log(scfg.material_rc_logrange))))),
                angle=angle,
                shape="ellipse" if rng.random() < scfg.material_ellipse_prob else "rect",
            ))
    return BoardLayout(
        aspect=aspect,
        h_hat=h_hat,
        robin=robin,
        gamma=gamma,
        sources=sources,
        materials=materials,
        side_gamma=side_gamma,
        boundary_segments=boundary_segments,
    )


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
    nonuniform_boundary = bool(
        layout.boundary_segments
        or (layout.side_gamma is not None and len(set(layout.side_gamma)) > 1)
    )
    if layout.materials or nonuniform_boundary:
        xy_grid = np.stack([X, Y], axis=-1)
        conductivity = layout.conductivity_at(xy_grid)
        sink = layout.h_hat * layout.sink_at(xy_grid)
        contact_resistance = layout.contact_resistance_faces(X, Y)
        lu, boundary = factorize_steady_variable_k(
            conductivity,
            layout.aspect,
            sink,
            bc="robin" if layout.robin else "dirichlet",
            gamma=layout.boundary_gamma_arrays(scfg.ny, nx),
            contact_resistance=contact_resistance,
        )
    else:
        lu, boundary = factorize_steady(
            scfg.ny,
            nx,
            layout.aspect,
            layout.h_hat,
            bc="robin" if layout.robin else "dirichlet",
            gamma=layout.gamma if layout.robin else 0.0,
        )
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
