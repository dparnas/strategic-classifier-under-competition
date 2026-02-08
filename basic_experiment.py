from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Label generators (start simple)
# -----------------------------
def deterministic_step_label(x: np.ndarray, tau: float = 0.7) -> np.ndarray:
    """y = 1[x >= tau]. Deterministic and fixed per experiment."""
    return (x >= tau).astype(int)


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
@dataclass(frozen=True)
class SupplierAction:
    kind: Literal["reject_all", "right_threshold"]
    threshold: Optional[float] = None  # only used if kind == "right_threshold"

    def accepts(self, xprime: np.ndarray) -> np.ndarray:
        if self.kind == "reject_all":
            return np.zeros_like(xprime, dtype=bool)
        assert self.threshold is not None
        return xprime >= self.threshold

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
        t = float(a.threshold)
        # candidate x': if x0 >= t stay, else move to t
        #todo: adapt for other classifiers
        xprime_j = np.where(x0 >= t, x0, t)
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
class BaselineConfig:
    n_users: int = 2000
    n_suppliers: int = 2
    tau: float = 0.7
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
    cfg: BaselineConfig,
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
    x0: np.ndarray,
    y: np.ndarray,
    current_actions: List[SupplierAction],
    cfg: BaselineConfig,
    cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> SupplierAction:
    """
    Grid-search the supplier's best-response action holding other suppliers fixed.
    Includes RejectAll as a special action (profit 0).
    """
    # Candidate thresholds
    grid = np.linspace(0.0, 1.0, cfg.grid_size)

    best_action = SupplierAction(kind="reject_all", threshold=None)
    best_profit = 0.0  # reject-all baseline

    # Try thresholds
    for t in grid:
        actions_try = list(current_actions)
        actions_try[supplier_idx] = SupplierAction(kind="right_threshold", threshold=float(t))

        x_star, chosen = simulate_round(x0, y, actions_try, cfg, cost_fn, rng, mode='prospective')
        prof = compute_expected_profit(y, chosen, supplier_idx, cfg.alpha, cfg.beta)

        # Must not deploy negative profit (so reject-all beats any negative)
        if prof > best_profit + 1e-12:
            best_profit = prof
            best_action = actions_try[supplier_idx]

    return best_action

def plot_thresholds(history):
    rounds = range(len(history["threshold"][0]))
    plt.figure(figsize=(8, 4))
    for supplier in history["threshold"].keys():
        plt.plot(rounds, history["threshold"][supplier], label=f"Supplier {supplier} threshold",
                 color=supplier_colors[supplier])
    # plt.plot(rounds, history["threshold"][1], label="Supplier 2 threshold")
    plt.xlabel("Round")
    plt.ylabel("Threshold")
    plt.title("Classifier thresholds over rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_profits(history, w_expectation=False):
    rounds = range(len(history["profit"][0]))
    plt.figure(figsize=(8, 4))
    if w_expectation:
        for supplier in history["profit"].keys():
            plt.plot(rounds[:-1], history["profit"][supplier][:-1], label=f"Supplier {supplier} profit",
                     color=supplier_colors[supplier])
        # plt.plot(rounds[:-1], history["profit"][0][:-1], label="Supplier 1 profit")
        # plt.plot(rounds[:-1], history["profit"][1][:-1], label="Supplier 2 profit")
            plt.plot(rounds[-2], history["profit"][supplier][-1], linestyle=None,
                     label=f"Expected Supplier {supplier} profit", marker='x',
                     color=supplier_colors[supplier], alpha=0.5)
        # plt.plot(rounds[-2], history["profit"][1][-1], linestyle=None,
        #          label="Expected Supplier 2 profit", marker='x', color='tab:orange', alpha=0.5)
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
    plt.show()

def plot_market_share(history, w_expectation=True):
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
    plt.show()

def plot_accuracy(history):
    rounds = range(len(history["accuracy"][0]))
    plt.figure(figsize=(8, 4))
    for supplier in history["accuracy"].keys():
        plt.plot(rounds, history["accuracy"][supplier], label=f"Supplier {supplier} accuracy",
                 color=supplier_colors[supplier])
    # plt.plot(rounds, history["acc2"], label="Supplier 2 accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Classifier accuracy over rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_user_metrics(history):
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
    plt.show()

def plot_baseline_dashboard(history, w_expectation=False):
    plot_thresholds(history)
    plot_profits(history, w_expectation)
    plot_market_share(history, w_expectation)
    plot_accuracy(history)
    plot_user_metrics(history)


import numpy as np
import matplotlib.pyplot as plt

def make_snapshot_figure(snapshot, title_extra: str = ""):
    x0 = snapshot.x0
    y = snapshot.y
    x_star = snapshot.x_star
    chosen = snapshot.chosen
    actions = snapshot.actions
    r = snapshot.round_idx

    fig, ax = plt.subplots(figsize=(10, 3.2))

    # original points at y=0
    ax.scatter(x0[y == 0], np.zeros(np.sum(y == 0)), s=10, alpha=0.6, label="y=0 (orig)", color='tab:red')
    ax.scatter(x0[y == 1], np.zeros(np.sum(y == 1)), s=10, alpha=0.6, label="y=1 (orig)", color='tab:green')

    # movement arrows/segments
    moved = chosen != -1
    idxs = np.where(moved)[0]
    for i in idxs:
        ax.plot([x0[i], x_star[i]], [0.0, 0.25], linewidth=0.5, alpha=0.25)

    # final positions by chosen supplier
    chosen_colors = supplier_colors
    for j in range(len(actions)):
        mask = chosen == j
        if np.any(mask):
            ax.scatter(x_star[mask], np.full(np.sum(mask), 0.25), s=14, alpha=0.8, label=f"chosen {j+1}",
                       color=chosen_colors[j])

    # opt-out
    out = chosen == -1
    if np.any(out):
        ax.scatter(x0[out], np.full(np.sum(out), -0.05), s=10, alpha=0.6, label="opt-out", color='tab:purple')

    # classifier thresholds
    for j, a in enumerate(actions):
        if a.kind == "right_threshold":
            ax.axvline(a.threshold, linestyle="--", alpha=0.9, color=chosen_colors[j])
            ax.text(a.threshold, 0.45, f"S{j+1}: {a.threshold:.2f}", rotation=90, va="bottom")
        else:
            ax.text(0.02 + 0.22*j, 0.45, f"S{j+1}: RejectAll", va="bottom")

    ax.set_ylim(-0.15, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_title(f"Round {r} {title_extra}".strip())
    ax.legend(loc="upper center", ncol=4, frameon=False)
    fig.tight_layout()

    return fig


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fig_to_rgb_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

import io
import imageio.v2 as imageio

def snapshots_to_gif(snapshots, gif_path="market_sim.gif", fps=2):
    frames = []
    for snap in snapshots:
        fig = make_snapshot_figure(snap)
        frame = fig_to_rgb_array(fig)
        plt.close(fig)
        frames.append(frame)
    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path


def snapshots_to_mp4(snapshots, mp4_path="market_sim.mp4", fps=6):
    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for snap in snapshots:
            fig = make_snapshot_figure(snap)
            frame = fig_to_rgb_array(fig)
            plt.close(fig)
            writer.append_data(frame)
    return mp4_path



def run_baseline_experiment(
    cfg: BaselineConfig,
    cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = weight_l1_cost(1.0),
    max_rounds_after_convergence: int = 0,
    store_snapshots: bool = True,
    add_expected_metrics: bool = False
):
    rounds_after_convergence = 0
    rng = np.random.default_rng(cfg.seed)
    snapshots: List[RoundSnapshot] = []

    # Sample users and labels
    x0 = rng.uniform(0.0, 1.0, size=cfg.n_users)
    y = deterministic_step_label(x0, tau=cfg.tau)

    # sns.scatterplot(x=x0, y=y, hue=y)
    # plt.title("Users at time=0")
    # plt.show()

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

    def _t(action: SupplierAction) -> float:
        if action.kind == "reject_all":
            return float("nan")
        return float(action.threshold)

    prev_actions = None

    for r in range(cfg.max_rounds):
        # Sequential best responses
        for j in range(cfg.n_suppliers):
            actions[j] = best_response_for_supplier(
                supplier_idx=j,
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
            history["threshold"][i].append(_t(actions[i]))
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
    if add_expected_metrics:
        _, chosen = simulate_round(x0, y, actions, cfg, cost_fn, rng, mode='prospective')
        for i in range(cfg.n_suppliers):
            history["profit"][i].append(compute_expected_profit(y, chosen, i, cfg.alpha, cfg.beta))
            history["mshare"][i].append(compute_expected_market_share_true_positives(y, chosen, i))
    return (history, snapshots) if store_snapshots else history


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cfg = BaselineConfig(
        n_users=100,
        n_suppliers=2,
        tau=0.7,
        alpha=1.0,
        beta=1.0,
        grid_size=101,
        max_rounds=10,
        seed=42,
    )
    add_expected_metrics = True
    max_rounds_after_convergence = 5
    hist, snaps = run_baseline_experiment(cfg, weight_l1_cost(10),
                                          max_rounds_after_convergence=max_rounds_after_convergence,
                                        add_expected_metrics=add_expected_metrics)
    print("Rounds:", len(hist["threshold"][0]))
    print("Final thresholds:", *[hist["threshold"][i][-1] for i in range(cfg.n_suppliers)])
    if add_expected_metrics:
        print("Final expected profits:", *[hist["profit"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final actual profits:", *[hist["profit"][i][-2] for i in range(cfg.n_suppliers)])
        print("Final expected mshares:", *[hist["mshare"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final actual mshares:", *[hist["mshare"][i][-2] for i in range(cfg.n_suppliers)])
    else:
        print("Final profits:", *[hist["profit"][i][-1] for i in range(cfg.n_suppliers)])
        print("Final mshares:", *[hist["mshare"][i][-1] for i in range(cfg.n_suppliers)])
    print("Final social burden (TP):", hist["social_burden"][-1])

    plot_baseline_dashboard(hist, w_expectation=add_expected_metrics)
    # gif_path = snapshots_to_gif(snaps, "baseline.gif", fps=2)
    gif_path = snapshots_to_mp4(snaps, "baseline.mp4", fps=6)

