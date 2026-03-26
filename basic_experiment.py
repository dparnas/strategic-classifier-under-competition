from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
# need imageio-ffmpeg for mp4
import imageio.v2 as imageio


# -----------------------------
# Label generators (start simple)
# -----------------------------
def deterministic_step_label(x: np.ndarray, label_threshold: float = 0.7) -> np.ndarray:
    """y = 1[x >= label_threshold]. Deterministic and fixed per experiment."""
    return (x >= label_threshold).astype(int)

def deterministic_intervals_label(x: np.ndarray, label_threshold: list[tuple]) -> np.ndarray:
    """y = 1[x in at least one interval]. Deterministic and fixed per experiment."""
    y = np.zeros(shape=x.shape[0])
    for interval in label_threshold:
        mask = (x >= interval[0]) & (x <= interval[1])
        y[mask] = 1
    return (y).astype(int)

def logistic_probability(x: np.ndarray, t: float = 0.7, k: float = 40.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - t)))

def probabilistic_step_label(x: np.ndarray, label_threshold: float = 0.7, k: float = 40.0, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    p = logistic_probability(x, t=label_threshold, k=k)
    return rng.binomial(1, p).astype(int)

# -----------------------------
# Cost functions
# -----------------------------
def weight_l1_cost(weight: float | int):
    def l1_cost(x: np.ndarray, xprime: np.ndarray) -> np.ndarray:
        return weight * np.abs(x - xprime)
    return l1_cost

# -----------------------------
# Supplier hypothesis: reject-all OR right-threshold
# -----------------------------
supplier_colors = ["tab:blue", "tab:orange", "tab:brown", "tab:gray"]
ActionKind = Literal["reject_all", "threshold", "interval"]
DirectionRL = Literal["right", "left"]  # realized directions only

@dataclass(frozen=True)
class SupplierAction:
    kind: ActionKind

    # threshold params
    threshold: Optional[float] = None
    direction: DirectionRL = "right"   # only used if kind == "threshold"

    # interval params
    lower: Optional[float] = None
    upper: Optional[float] = None

    def accepts(self, xprime: np.ndarray) -> np.ndarray:
        if self.kind == "reject_all":
            return np.zeros_like(xprime, dtype=bool)

        if self.kind == "threshold":
            assert self.threshold is not None
            t = float(self.threshold)
            if self.direction == "right":
                return xprime >= t
            else:  # "left"
                return xprime <= t

        if self.kind == "interval":
            assert self.lower is not None and self.upper is not None
            lo, hi = (self.lower, self.upper) if self.lower <= self.upper else (self.upper, self.lower)
            return (xprime >= lo) & (xprime <= hi)

        raise ValueError(f"Unknown kind: {self.kind}")

    def best_xprime_l1(self, x0: np.ndarray) -> np.ndarray:
        """Argmin_{x'} |x'-x0| s.t. accepts(x') == True, in 1D."""
        if self.kind == "reject_all":
            return x0  # infeasible; caller should handle

        if self.kind == "threshold":
            assert self.threshold is not None
            t = float(self.threshold)
            if self.direction == "right":
                return np.where(x0 >= t, x0, t)
            else:  # left
                return np.where(x0 <= t, x0, t)

        if self.kind == "interval":
            assert self.lower is not None and self.upper is not None
            lo, hi = (self.lower, self.upper) if self.lower <= self.upper else (self.upper, self.lower)
            return np.clip(x0, lo, hi)

        raise ValueError(f"Unknown kind: {self.kind}")

ClassifierType = Literal["thr_right", "thr_left", "thr_multi", "interval"]

@dataclass(frozen=True)
class SupplierClassifierSpec:
    ctype: ClassifierType


@dataclass
class RoundSnapshot:
    round_idx: int
    actions: List[SupplierAction]
    x0: np.ndarray
    y: np.ndarray
    x_star: np.ndarray
    chosen: np.ndarray

# -----------------------------
# User strategic response + choice
# -----------------------------
def _best_response_to_actions(
    x0: np.ndarray,
    actions: List[SupplierAction],
    v: float,
    cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
    mode: str = 'retrospective',
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Given CURRENT deployed actions, compute:
      - x_star: where each user moves (possibly stays)
      - chosen_supplier: in {0..m-1} for chosen supplier, or -1 for opt-out

    Rules (per your spec):
      - User moves to be accepted by at least one model with minimal cost.
      - Choice: maximize v*h(x') - c(x0, x').
        Since v is fixed and h is binary, among accepting suppliers this reduces to minimal cost;
        tie-break uniformly among minimal-cost accepting suppliers.
      - Opt-out utility is 0. If no accepting supplier yields positive utility, opt out.
        With v=1 and |.|<=1 on [0,1], usually positive if any supplier can accept.
    """
    n = x0.shape[0]
    m = len(actions)
    chosen = np.full((n, m), -1, dtype=float) if mode == 'prospective' else np.full(n, -1, dtype=int)
    x_star = x0.copy()

    # Precompute which suppliers can ever accept (RejectAll cannot)
    can_accept = np.array([a.kind != "reject_all" for a in actions], dtype=bool)

    # For right-thresholds with L1 cost, minimal-cost move to be accepted is:
    #   x' = x0 if already accepted (x0 >= t), else x' = t (cost t-x0)
    # We'll compute per supplier the best x' and its utility.

    # Utilities per supplier; initialize to -inf
    utilities = np.full((n, m), -np.inf, dtype=float)
    xprime_candidates = np.tile(x0[:, None], (1, m))  # (n, m)

    for j, a in enumerate(actions):
        if a.kind == "reject_all":
            continue
        # candidate x': if x0 >= t stay, else move to t
        #todo: adapt for other classifiers
        xprime_j = a.best_xprime_l1(x0)
        xprime_candidates[:, j] = xprime_j

        accepted = a.accepts(xprime_j)  # should be True by construction unless numerical weirdness
        # utility = v*1 - cost if accepted else 0 - cost (but not relevant here)
        c = cost_fn(x0, xprime_j)
        u = np.where(accepted, v - c, -np.inf)
        utilities[:, j] = u

    # Opt-out utility = 0
    best_u = np.max(utilities, axis=1)
    # If best_u <= 0, opt out (ties at 0 go to opt-out by this rule; adjust if you prefer)
    will_choose = best_u > 0

    if mode == 'retrospective':
        # Among suppliers achieving best utility, tie-break uniformly
        for i in np.where(will_choose)[0]:
            js = np.flatnonzero(np.isclose(utilities[i], best_u[i], atol=1e-12, rtol=0.0))
            j = rng.choice(js)
            chosen[i] = j # random variable
            x_star[i] = xprime_candidates[i, j]
        return x_star, chosen
    elif mode == 'prospective':
        for i in np.where(will_choose)[0]:
            js = np.flatnonzero(np.isclose(utilities[i], best_u[i], atol=1e-12, rtol=0.0))
            chosen[i, js] = 1/len(js) # distribution
        return None, chosen

# -----------------------------
# Metrics
# -----------------------------
def compute_profit(y: np.ndarray, chosen: np.ndarray, supplier_idx: int, alpha: float, beta: float) -> float:
    mask = chosen == supplier_idx
    if mask.sum() == 0:
        return 0.0
    tp = np.sum(y[mask] == 1)
    fp = np.sum(y[mask] == 0)
    return float(alpha * tp - beta * fp)

def compute_expected_profit(y: np.ndarray, chosen: np.ndarray, supplier_idx: int, alpha: float, beta: float) -> float:
    expectation_weights = chosen[:, supplier_idx]
    mask = expectation_weights > 0
    tp = np.sum((y[mask] == 1) * expectation_weights[mask])
    fp = np.sum((y[mask] == 0) * expectation_weights[mask])
    return float(alpha * tp - beta * fp)

def compute_market_share_true_positives(y: np.ndarray, chosen: np.ndarray, supplier_idx: int) -> float:
    tp_total = np.sum(y == 1)
    if tp_total == 0:
        return 0.0
    return float(np.sum((y == 1) & (chosen == supplier_idx)) / tp_total)

def compute_expected_market_share_true_positives(y: np.ndarray, chosen: np.ndarray, supplier_idx: int) -> float:
    tp_total = np.sum(y == 1)
    expectation_weights = chosen[:, supplier_idx]
    mask = expectation_weights > 0
    if tp_total == 0:
        return 0.0
    return float(np.sum((y[mask] == 1) * expectation_weights[mask]) / tp_total)

def compute_accuracy(y: np.ndarray, x: np.ndarray, action: SupplierAction) -> float:
    # accuracy = 1[h(x0)==y], with reject_all meaning h=0 everywhere
    preds = action.accepts(x)
    yhat = preds.astype(int)
    return float(np.mean(yhat == y))

def compute_social_burden_true_positives(y: np.ndarray, x0: np.ndarray, x_star: np.ndarray, cost_fn) -> float:
    mask = y == 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(cost_fn(x0[mask], x_star[mask])))

def compute_user_welfare(x0: np.ndarray, x_star: np.ndarray, chosen: np.ndarray, v: float, cost_fn) -> float:
    # welfare_i = v*1[chosen!=-1] - cost(x0,x_star) if chosen!=-1 else 0
    accepted = chosen != -1
    c = cost_fn(x0, x_star)
    welfare = np.where(accepted, v - c, 0.0)
    return float(np.mean(welfare))


# -----------------------------
# Simulation: 2 suppliers, sequential best responses
# -----------------------------
@dataclass
class ExpConfig:
    n_users: int = 2000
    n_suppliers: int = 2
    supplier_specs: List[SupplierClassifierSpec] = field(default_factory=list)
    label_params: dict = field(default_factory=dict)
    label_generator: Callable = deterministic_step_label
    v: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    grid_size: int = 101
    max_rounds: int = 50
    # mode: Literal["move_after_all", "move_between_deployments"] = "move_after_all"
    seed: int = 0


def simulate_round(
    x0: np.ndarray,
    y: np.ndarray,
    actions: List[SupplierAction],
    cfg: ExpConfig,
    cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
    store_snapshots: bool = False,
    mode: str = 'retrospective',
    **kwargs
):
    """
    Returns (x_star, chosen) after applying the movement/choice rules under the configured mode.
    """
    # Users best-respond once to the full action set
    x_star, chosen = _best_response_to_actions(x0, actions, cfg.v, cost_fn, rng, mode)
    if store_snapshots and mode == 'retrospective':
        snapshots = kwargs.get('snapshots')
        snapshots.append(RoundSnapshot(
            round_idx=kwargs.get("round_idx"),
            actions=list(actions),
            x0=x0.copy(),  # x0 fixed, but copy for safety
            y=y.copy(),
            x_star=x_star.copy(),
            chosen=chosen.copy(),
        ))
        return x_star, chosen, snapshots
    return x_star, chosen


def best_response_for_supplier(
    supplier_idx: int,
    current_actions: list[SupplierAction],
    supplier_spec: SupplierClassifierSpec,
    x0: np.ndarray,
    y: np.ndarray,
    cfg,
    cost_fn,
    rng,
):
    grid = np.linspace(0.0, 1.0, cfg.grid_size)

    best_action = SupplierAction(kind="reject_all")
    best_profit = 0.0

    def eval_action(candidate: SupplierAction):
        nonlocal best_action, best_profit
        actions_try = list(current_actions)
        actions_try[supplier_idx] = candidate

        # prospective evaluation (as you already do)
        _, chosen = simulate_round(x0, y, actions_try, cfg, cost_fn, rng, mode="prospective")
        prof = compute_expected_profit(y, chosen, supplier_idx, cfg.alpha, cfg.beta)

        if prof > best_profit + 1e-12:
            best_profit = prof
            best_action = candidate

    ctype = supplier_spec.ctype

    if ctype in ("thr_right", "thr_multi"):
        for t in grid:
            eval_action(SupplierAction(kind="threshold", threshold=float(t), direction="right"))

    if ctype in ("thr_left", "thr_multi"):
        for t in grid:
            eval_action(SupplierAction(kind="threshold", threshold=float(t), direction="left"))

    if ctype == "interval":
        # O(G^2) intervals; OK for moderate grid_size
        for lo in grid:
            for hi in grid:
                if hi < lo:
                    continue
                eval_action(SupplierAction(kind="interval", lower=float(lo), upper=float(hi)))

    return best_action


# def plot_thresholds(history):
#     rounds = range(len(history["threshold"][0]))
#     number_of_thresholds = len(history["threshold"][0][0])
#     plt.figure(figsize=(8, 4))
#     if number_of_thresholds - 1 == 1:
#         for supplier in history["threshold"].keys():
#             plt.plot(rounds, history["threshold"][supplier], label=f"Supplier {supplier} threshold",
#                      color=supplier_colors[supplier])
#     elif number_of_thresholds - 1 == 2:
#         for supplier in history["threshold"].keys():
#             lower_thresholds = [thresholds[0] for thresholds in history["threshold"][supplier]]
#             upper_thresholds = [thresholds[1] for thresholds in history["threshold"][supplier]]
#             plt.plot(rounds, lower_thresholds, linestyle='--', label=f"Supplier {supplier} threshold",
#                      color=supplier_colors[supplier])
#             plt.plot(rounds, upper_thresholds, linestyle=':', label=f"Supplier {supplier} threshold",
#                      color=supplier_colors[supplier])
#     # plt.plot(rounds, history["threshold"][1], label="Supplier 2 threshold")
#     plt.xlabel("Round")
#     plt.ylabel("Threshold")
#     plt.title("Classifier thresholds over rounds")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def _action_to_band(action, x_min=0.0, x_max=1.0):
    """
    Map a per-round classifier description into an accepted interval [lo, hi].
    action is either:
      (t, "right"), (t, "left"), or (lo, hi, "interval")
    """
    if action is None:
        return (np.nan, np.nan)

    # threshold + direction
    if len(action) == 2:
        t, direction = action
        t = float(t)
        if direction == "right":
            return (t, x_max)
        elif direction == "left":
            return (x_min, t)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    # interval
    if len(action) == 3:
        lo, hi, kind = action
        if kind != "interval":
            raise ValueError(f"Unknown kind: {kind}")
        lo, hi = float(lo), float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    raise ValueError(f"Unexpected action format: {action}")


def plot_thresholds(history, save_dir=None, show_graphs=True, x_min=0.0, x_max=1.0, alpha=0.20):
    """
    For each supplier and each round, history["threshold"][supplier][r] is either:
      (t, "right"), (t, "left"), or (lo, hi, "interval").

    This plots + fills the accepted region [lo, hi] over rounds.
    """
    suppliers = list(history["threshold"].keys())
    n_rounds = len(history["threshold"][suppliers[0]])
    rounds = np.arange(n_rounds)

    plt.figure(figsize=(10, 5))

    for supplier in suppliers:
        actions = history["threshold"][supplier]

        lo = np.empty(n_rounds, dtype=float)
        hi = np.empty(n_rounds, dtype=float)

        for r, a in enumerate(actions):
            lo[r], hi[r] = _action_to_band(a, x_min=x_min, x_max=x_max)

        # Clip to [x_min, x_max] and mask invalid
        lo = np.clip(lo, x_min, x_max)
        hi = np.clip(hi, x_min, x_max)
        valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)

        # Fill accepted band (over rounds)
        plt.fill_between(
            rounds,
            lo,
            hi,
            where=valid,
            interpolate=True,
            alpha=alpha,
            color=supplier_colors[supplier],
            label=f"Supplier {supplier} accepted region",
        )

        # Plot boundaries to make it readable
        plt.plot(rounds[valid], lo[valid], color=supplier_colors[supplier], linewidth=1.5, linestyle="--")
        plt.plot(rounds[valid], hi[valid], color=supplier_colors[supplier], linewidth=1.5, linestyle=":")

    plt.xlabel("Round")
    plt.ylabel("Accepted x-range")
    plt.title("Classifier accepted regions over rounds")
    plt.ylim(x_min, x_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "thresholds.png"))
    if show_graphs:
        plt.show()


def plot_profits(history, w_expectation=False, save_dir=None, show_graphs=True):
    rounds = range(len(history["profit"][0]))
    plt.figure(figsize=(8, 4))
    if w_expectation:
        for supplier in history["profit"].keys():
            plt.plot(rounds[:-1], history["profit"][supplier][:-1], label=f"Supplier {supplier} profit",
                     color=supplier_colors[supplier])
            plt.plot(rounds[-2], history["profit"][supplier][-1], linestyle=None,
                     label=f"Expected Supplier {supplier} profit", marker='x',
                     color=supplier_colors[supplier], alpha=0.5)
    else:
        for supplier in history["threshold"].keys():
            plt.plot(rounds, history["profit"][supplier], label=f"Supplier {supplier} profit",
                     color=supplier_colors[supplier])
    plt.xlabel("Round")
    plt.ylabel("Profit")
    plt.title("Supplier profit over rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "profits.png"))
    if show_graphs:
        plt.show()

def plot_market_share(history, w_expectation=False, save_dir=None, show_graphs=True):
    rounds = range(len(history["mshare"][0]))
    plt.figure(figsize=(8, 4))
    if w_expectation:
        for supplier in history["threshold"].keys():
            plt.plot(rounds[:-1], history["mshare"][supplier][:-1], label=f"Supplier {supplier} TP share",
                     color=supplier_colors[supplier])
        # plt.plot(rounds[:-1], history["mshare"][0][:-1], label="Supplier 1 TP share")
        # plt.plot(rounds[:-1], history["mshare"][1][:-1], label="Supplier 2 TP share")
            plt.plot(rounds[-2], history["mshare"][supplier][-1], linestyle=None,
                     label=f"Expected Supplier {supplier} TP share", marker='x',
                     color=supplier_colors[supplier], alpha=0.5)
        # plt.plot(rounds[-2], history["mshare"][1][-1], linestyle=None,
        #          label="Expected Supplier 2 TP share", marker='x', color='tab:orange', alpha=0.5)
    else:
        for supplier in history["threshold"].keys():
            plt.plot(rounds, history["mshare"][supplier], label=f"Supplier {supplier} TP share",
                     color=supplier_colors[supplier])

    plt.xlabel("Round")
    plt.ylabel("Market share (true positives)")
    plt.title("Market share over rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "market_shares.png"))
    if show_graphs:
        plt.show()

def plot_accuracy(history, save_dir=None, show_graphs=True):
    rounds = range(len(history["accuracy"][0]))
    plt.figure(figsize=(8, 4))
    for supplier in history["accuracy"].keys():
        plt.plot(rounds, history["accuracy"][supplier], label=f"Supplier {supplier} accuracy",
                 color=supplier_colors[supplier])
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Classifier accuracy over rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "accuracy.png"))
    if show_graphs:
        plt.show()

def plot_user_metrics(history, save_dir=None, show_graphs=True):
    rounds = range(len(history["user_welfare"]))
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, history["user_welfare"], label="User welfare", color='gold')
    plt.plot(rounds, history["social_burden"], label="Social burden (TP)", color='firebrick')
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("User welfare and social burden")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "user_metrics.png"))
    if show_graphs:
        plt.show()

def plot_baseline_dashboard(history, w_expectation: bool=False, save_dir: None | str=None, show_graphs: bool=True):
    plot_thresholds(history, save_dir, show_graphs)
    plot_profits(history, w_expectation, save_dir, show_graphs)
    plot_market_share(history, w_expectation, save_dir, show_graphs)
    plot_accuracy(history, save_dir, show_graphs)
    plot_user_metrics(history, save_dir, show_graphs)

def make_snapshot_figure(snapshot, title_extra: str = "", x_min: float = 0.0, x_max: float = 1.0):
    x0 = snapshot.x0
    y = snapshot.y
    x_star = snapshot.x_star
    chosen = snapshot.chosen
    actions = snapshot.actions
    r = snapshot.round_idx

    fig, ax = plt.subplots(figsize=(10, 3.2))

    # --- y-levels for visual layout ---
    y_orig = 0.00
    y_final = 0.15
    y_out = -0.05

    ax.set_ylim(-0.15, 0.55)
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("x")
    ax.set_title(f"Round {r} {title_extra}".strip())

    # --- 1) Fill accepted regions + draw boundary lines (DO THIS FIRST so points appear on top) ---
    xgrid = np.linspace(x_min, x_max, 400)
    y_bottom, y_top = -0.12, 0.52  # vertical span of the shaded acceptance band

    for j, a in enumerate(actions):
        col = supplier_colors[j]

        if a.kind == "reject_all":
            ax.text(0.02 + 0.22 * j, 0.35, f"S{j+1}: RejectAll", va="bottom")
            continue

        # accepted mask over xgrid
        accepted_mask = a.accepts(xgrid)

        # Fill the vertical band where accepted
        ax.fill_between(
            xgrid,
            y_bottom,
            y_top,
            where=accepted_mask,
            interpolate=True,
            alpha=0.10,          # increase if you want more visible shading
            color=col,
            zorder=0,
        )

        # Draw demarcation lines + annotate
        if a.kind == "threshold":
            t = float(a.threshold)
            ax.axvline(t, linestyle="--", linewidth=2.0, alpha=0.95, color=col, zorder=3)
            ax.text(
                t, 0.25 + j * 0.1,
                f"S{j+1}: {a.direction} @ {t:.2f}",
                va="bottom", ha="center", color=col
            )

        elif a.kind == "interval":
            lo = float(a.lower)
            hi = float(a.upper)
            if hi < lo:
                lo, hi = hi, lo

            ax.axvline(lo, linestyle="--", linewidth=2.0, alpha=0.95, color=col, zorder=3)
            ax.axvline(hi, linestyle="--",  linewidth=2.0, alpha=0.95, color=col, zorder=3)
            ax.text(
                (lo + hi) / 2, 0.25 + j * 0.1,
                f"S{j+1}: [{lo:.2f}, {hi:.2f}]",
                va="bottom", ha="center", color=col
            )

        else:
            # In case you add new kinds later
            ax.text(0.02 + 0.22 * j, 0.25 + j * 0.1, f"S{j+1}: {a.kind}", va="bottom", color=col)

    # --- 2) Original points at y=0 ---
    ax.scatter(x0[y == 0], np.full(np.sum(y == 0), y_orig), s=10, alpha=0.6,
               label="y=0 (orig)", color='tab:red', zorder=4)
    ax.scatter(x0[y == 1], np.full(np.sum(y == 1), y_orig), s=10, alpha=0.6,
               label="y=1 (orig)", color='tab:green', zorder=4)

    # --- 3) Movement arrows/segments ---
    moved = chosen != -1
    idxs = np.where(moved)[0]
    for i in idxs:
        ax.plot([x0[i], x_star[i]], [y_orig, y_final],
                linewidth=0.8, alpha=0.25, color="k", zorder=2)

    # --- 4) Final positions by chosen supplier ---
    for j in range(len(actions)):
        mask = chosen == j
        if np.any(mask):
            ax.scatter(x_star[mask], np.full(np.sum(mask), y_final),
                       s=14, alpha=0.85, label=f"chosen {j+1}",
                       color=supplier_colors[j], zorder=5)

    # --- 5) Opt-out ---
    out = chosen == -1
    if np.any(out):
        ax.scatter(x0[out], np.full(np.sum(out), y_out), s=10, alpha=0.6,
                   label="opt-out", color='tab:purple', zorder=4)

    ax.legend(loc="upper center", ncol=4, frameon=False)
    fig.tight_layout()
    return fig

def fig_to_rgb_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

def snapshots_to_mp4(snapshots, mp4_path="market_dynamics.mp4", fps=6):
    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for snap in snapshots:
            fig = make_snapshot_figure(snap)
            frame = fig_to_rgb_array(fig)
            plt.close(fig)
            writer.append_data(frame)
    return mp4_path

def run_baseline_experiment(
    cfg: ExpConfig,
    cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = weight_l1_cost(1.0),
    max_rounds_after_convergence: int = 0,
    store_snapshots: bool = True,
    add_next_expected_metrics: bool = False,
    save_dir: str | None = None
):
    rounds_after_convergence = 0
    rng = np.random.default_rng(cfg.seed)
    snapshots: List[RoundSnapshot] = []

    # Sample users and labels
    x0 = rng.uniform(0.0, 1.0, size=cfg.n_users)
    y = cfg.label_generator(x0, **cfg.label_params)

    if save_dir is not None:
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        sns.histplot(x=x0, hue=y, hue_order=[1,0], palette=['tab:green', 'tab:red'], bins=20, ax=ax[1])
        sns.scatterplot(x=x0, y=y, hue=y, hue_order=[1,0], palette=['tab:green', 'tab:red'], alpha=0.7, s=10, ax=ax[0])
        ax[0].set_xlim((0, 1))
        ax[1].set_xlim((0, 1))
        plt.suptitle("Y Distribution")
        plt.savefig(os.path.join(save_dir, 'y_distribution.png'))

    # Initialize actions (simple start)
    actions = [SupplierAction(kind='reject_all') for i in range(cfg.n_suppliers)]
    # actions = [
    #     SupplierAction(kind="reject_all"),
    #     SupplierAction(kind="reject_all"),
    # ]

    history = {metric: {i: [] for i in range(cfg.n_suppliers)} for metric in [
        "threshold", "profit", "accuracy", "mshare"]}
    history["social_burden"] = []
    history["user_welfare"] = []

    def get_classifier_rule(action: SupplierAction) -> float | tuple:
        if action.kind == "reject_all":
            return float("nan")
        elif action.kind == "threshold":
            return (float(action.threshold), action.direction)
        elif action.kind == "interval":
            return (float(action.lower), float(action.upper), action.kind)

    prev_actions = None

    for r in range(cfg.max_rounds):
        # Sequential best responses
        for j in range(cfg.n_suppliers):
            actions[j] = best_response_for_supplier(
                supplier_idx=j,
                supplier_spec=cfg.supplier_specs[j],
                x0=x0,
                y=y,
                current_actions=actions,
                cfg=cfg,
                cost_fn=cost_fn,
                rng=rng,
            )

        # Evaluate metrics after both updated
        if store_snapshots:
            x_star, chosen, snapshots = simulate_round(x0, y, actions, cfg, cost_fn, rng, store_snapshots, round_idx=r,
                                            snapshots=snapshots, mode='retrospective')
        else:
            x_star, chosen = simulate_round(x0, y, actions, cfg, cost_fn, rng, mode='retrospective')
        sb = compute_social_burden_true_positives(y, x0, x_star, cost_fn)
        uw = compute_user_welfare(x0, x_star, chosen, cfg.v, cost_fn)

        for i in range(cfg.n_suppliers):
            history["threshold"][i].append(get_classifier_rule(actions[i]))
            history["profit"][i].append(compute_profit(y, chosen, i, cfg.alpha, cfg.beta))
            history["accuracy"][i].append(compute_accuracy(y, x_star, actions[i]))
            history["mshare"][i].append(compute_market_share_true_positives(y, chosen, i))
        history["social_burden"].append(sb)
        history["user_welfare"].append(uw)

        # Convergence: actions unchanged (exactly, since chosen from grid / reject_all)
        if prev_actions is not None and actions == prev_actions:
            rounds_after_convergence += 1
            if rounds_after_convergence >= max_rounds_after_convergence:
                break
        prev_actions = list(actions)
    if add_next_expected_metrics:
        _, chosen = simulate_round(x0, y, actions, cfg, cost_fn, rng, mode='prospective')
        for i in range(cfg.n_suppliers):
            history["profit"][i].append(compute_expected_profit(y, chosen, i, cfg.alpha, cfg.beta))
            history["mshare"][i].append(compute_expected_market_share_true_positives(y, chosen, i))
    return (history, snapshots) if store_snapshots else history


# -----------------------------
# Example usage
# -----------------------------
saved_configs = {'simplest_experiment': # two suppliers, y = I[x >= 0.7], models = I[x>=t] -> shows our basic hypothesis
    ExpConfig(
        n_users=50,
        n_suppliers=2,
        supplier_specs=[SupplierClassifierSpec("thr_right"), SupplierClassifierSpec("thr_right")],
        label_params={'label_threshold': 0.7},
        label_generator=deterministic_step_label,
        alpha=1.0,
        beta=1.0,
        grid_size=101,
        max_rounds=10,
        seed=42,
    ), 'deterministic_intervals_experiment': # two suppliers, y = I[x >= 0.7], models = I[t1>x>=t2] -> shows competition "improving" model for consumer.
                                    # Note that this is an artifact of how the best threshold is chosen
    ExpConfig(
        n_users=50,
        n_suppliers=2,
        supplier_specs=[SupplierClassifierSpec("interval"), SupplierClassifierSpec("interval")],
        label_params={'label_threshold': 0.7},
        label_generator=deterministic_step_label,
        alpha=1.0,
        beta=1.0,
        grid_size=101,
        max_rounds=10,
        seed=42,
    ), 'deterministic_two_market_experiment': # two suppliers, y = I[x >= 0.7], models = I[x>t] or I[x<t] -> converge to different markets.
    ExpConfig(
        n_users=50,
        n_suppliers=2,
        supplier_specs=[SupplierClassifierSpec("thr_multi"), SupplierClassifierSpec("thr_multi")],
        label_params={'label_threshold': [(0, 0.3), (0.7, 1.0)]},
        label_generator=deterministic_intervals_label,
        alpha=1.0,
        beta=1.0,
        grid_size=101,
        max_rounds=10,
        seed=42,
    ), 'probabilistic_interval_experiment': # Need to further explore, is the negative profit due to randomness or a problem in the policy
    ExpConfig(
        n_users=50,
        n_suppliers=2,
        supplier_specs=[SupplierClassifierSpec("interval"), SupplierClassifierSpec("interval")],
        label_params={'label_threshold': 0.5, 'k': 1, 'rng': np.random.default_rng(5)},
        label_generator=probabilistic_step_label,
        alpha=1.0,
        beta=1.0,
        grid_size=101,
        max_rounds=15,
        seed=42,
    )

}
if __name__ == "__main__":
    experiment_name = 'probabilistic_interval_experiment'
    cfg = saved_configs[experiment_name]
    save_dir = './' + experiment_name # or None if you don't want to save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    show_graphs = False
    add_next_expected_metrics = True
    max_rounds_after_convergence = 5
    hist, snaps = run_baseline_experiment(cfg, weight_l1_cost(10),
                                          max_rounds_after_convergence=max_rounds_after_convergence,
                                        add_next_expected_metrics=add_next_expected_metrics, save_dir=save_dir)
    print("Rounds:", len(hist["threshold"][0]))
    print("Final thresholds:", *[hist["threshold"][i][-1] for i in range(cfg.n_suppliers)])
    if add_next_expected_metrics:
        print("Final expected profits:", *[hist["profit"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final actual profits:", *[hist["profit"][i][-2] for i in range(cfg.n_suppliers)])
        print("Final expected mshares:", *[hist["mshare"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final actual mshares:", *[hist["mshare"][i][-2] for i in range(cfg.n_suppliers)])
    else:
        print("Final profits:", *[hist["profit"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final mshares:", *[hist["mshare"][i][-1] for i in range(cfg.n_suppliers)])
    print("Final social burden (TP):", hist["social_burden"][-1])


    plot_baseline_dashboard(hist, w_expectation=add_next_expected_metrics, save_dir=save_dir, show_graphs=show_graphs)
    mp4_path = snapshots_to_mp4(snaps, os.path.join(save_dir, f"market_dynamics.mp4"), fps=6)

