"""Broker price figures (illustrates D5: the heavy-tailed volatile broker).

These are model-illustration figures, not results: they visualize an INPUT
assumption of the model (the three broker price processes, and in particular
the volatile low-cost broker's heavy-tailed, mean-reverting shock from
docs/DECISIONS.md D5), so an examiner can see the price dynamics the summary
statistics describe. Nothing here is a simulation outcome and nothing is tuned.

It produces, into results/plots/:
- fig_broker_price_archetypes.png: the three broker prices over one year, to
  show why the volatile broker is the only one with non-trivial dynamics
  (regulated is a slow seasonal cosine, green is flat, volatile fluctuates).
- fig_volatile_price_candles.png: a three-panel view of the volatile broker:
  (top) daily open-high-low-close candlesticks over a representative window with
  the floor, cap, and baseline marked; (bottom-left) the distribution of the
  hourly price innovations against a Gaussian of equal standard deviation, on a
  log count axis so the heavy tails are visible; (bottom-right) a normal
  quantile-quantile plot of the innovations, whose S-shape is the signature of
  heavy (Student-t) tails.

Prices come from the real broker classes (the same code the simulation runs).
The volatile broker's innovations are captured exactly, from the real code path,
by wrapping its random generator to record each standard_t draw (no pricing
logic is duplicated here). Deterministic given the config seed.

Run from the repository root:
    python scripts/plot_broker_prices.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Mesa 3.5.1 still supports seed= (the model-driver API) but nudges toward rng=;
# functionally identical per Mesa's own message (matches the other scripts).
warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.model import MicrogridModel

HORIZON_HOURS = 8760
HOURS_PER_DAY = 24
CANDLE_WINDOW_DAYS = 120  # a fixed, representative window for legible candles

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"

# Okabe-Ito colorblind-safe hues, assigned in a fixed order (identity, not rank).
COLOR_REGULATED = "#0072B2"  # blue
COLOR_VOLATILE = "#D55E00"  # vermillion (the broker of interest)
COLOR_GREEN = "#009E73"  # bluish green
INK = "#222222"
GRID = "#DDDDDD"
GUIDE = "#999999"


class _RecordingGenerator:
    """Wraps a numpy Generator and records every standard_t draw, so the real
    VolatileLowCostBroker.quote() code path can be run unchanged while the exact
    innovations it consumes are captured. Any other attribute access passes
    straight through to the wrapped generator."""

    def __init__(self, generator: np.random.Generator, sink: list) -> None:
        self._generator = generator
        self._sink = sink

    def standard_t(self, df):
        value = self._generator.standard_t(df)
        self._sink.append(value)
        return value

    def __getattr__(self, name):
        return getattr(self._generator, name)


def _broker_params(config: dict) -> dict:
    return {broker["id"]: broker for broker in config["brokers"]}


def realize_prices(config: dict):
    """Realize the three brokers' hourly prices for one year plus the volatile
    broker's exact hourly innovations (EUR/kWh).

    The brokers are taken from a real MicrogridModel, so the volatile broker
    carries exactly the model-seeded random generator the simulation would use
    (its seed is drawn from the model RNG at construction, see
    environment/model._build_volatile_low_cost). Right after construction the
    broker's generator is at draw zero (quote() is never called during model
    build), so driving quote() here reproduces the model's canonical price path
    for the config seed. The generator is wrapped first so each standard_t draw
    is captured without duplicating any pricing logic. This makes the figure's
    realized statistics match the values reported for the shipped config."""
    vol_p = _broker_params(config)["volatile_low_cost"]

    model = MicrogridModel(config)
    regulated = model.brokers["regulated_utility"]
    green = model.brokers["premium_green"]
    volatile = model.brokers["volatile_low_cost"]

    raw_draws: list = []
    volatile._rng = _RecordingGenerator(volatile._rng, raw_draws)

    price_reg = np.empty(HORIZON_HOURS)
    price_grn = np.empty(HORIZON_HOURS)
    price_vol = np.empty(HORIZON_HOURS)
    for hour in range(HORIZON_HOURS):
        price_reg[hour] = regulated.quote(hour)
        price_grn[hour] = green.quote(hour)
        price_vol[hour] = volatile.quote(hour)

    # innovation (EUR/kWh) = standard_t draw * shock_scale, one per hour, exact.
    innovations = np.asarray(raw_draws) * vol_p["shock_scale_eur_per_kwh"]

    stats_block = {
        "floor": vol_p["price_floor_eur_per_kwh"],
        "cap": vol_p["baseline_eur_per_kwh"] * vol_p["price_cap_multiplier"],
        "baseline": vol_p["baseline_eur_per_kwh"],
        "df": vol_p["shock_degrees_of_freedom"],
        "shock_scale": vol_p["shock_scale_eur_per_kwh"],
        "phi": vol_p["mean_reversion_phi"],
        "price_mean": float(price_vol.mean()),
        "price_std": float(price_vol.std(ddof=0)),
        "price_min": float(price_vol.min()),
        "price_max": float(price_vol.max()),
        "floor_hit_pct": 100.0 * float(np.mean(price_vol <= vol_p["price_floor_eur_per_kwh"] + 1e-12)),
        "cap_hit_pct": 100.0 * float(np.mean(price_vol >= vol_p["baseline_eur_per_kwh"] * vol_p["price_cap_multiplier"] - 1e-12)),
    }
    return price_reg, price_grn, price_vol, innovations, stats_block


def _daily_ohlc(hourly: np.ndarray) -> np.ndarray:
    """Reshape an hourly series into daily [open, high, low, close] rows."""
    n_days = len(hourly) // HOURS_PER_DAY
    day_grid = hourly[: n_days * HOURS_PER_DAY].reshape(n_days, HOURS_PER_DAY)
    ohlc = np.column_stack(
        [day_grid[:, 0], day_grid.max(axis=1), day_grid.min(axis=1), day_grid[:, -1]]
    )
    return ohlc


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GUIDE)
    ax.tick_params(colors=INK, labelsize=9)


def plot_archetypes(price_reg, price_grn, price_vol, out_path: Path) -> None:
    days = np.arange(HORIZON_HOURS) / HOURS_PER_DAY
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _style_axis(ax)
    # volatile drawn first and thinnest so the flat lines read on top of it
    ax.plot(days, price_vol, color=COLOR_VOLATILE, linewidth=0.5, alpha=0.9, label="Volatile Low-Cost")
    ax.plot(days, price_reg, color=COLOR_REGULATED, linewidth=1.8, label="Regulated Utility")
    ax.plot(days, price_grn, color=COLOR_GREEN, linewidth=1.8, label="Premium Green")
    ax.set_xlim(0, 365)
    ax.set_xlabel("Day of year", color=INK)
    ax.set_ylabel("Quoted price (EUR/kWh)", color=INK)
    ax.set_title(
        "Broker price archetypes over one year: only the volatile broker fluctuates",
        color=INK,
        fontsize=12,
    )
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    # The volatile series is drawn very thin (0.5) so the flat lines read on top
    # of it; give its legend swatch a visible width so identity is clear.
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        handle.set_linewidth(2.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_candles(ax, ohlc: np.ndarray, stats_block: dict) -> None:
    _style_axis(ax)
    up = ohlc[:, 3] >= ohlc[:, 0]  # close >= open
    x = np.arange(len(ohlc))
    # wicks: daily high-low
    ax.vlines(x, ohlc[:, 2], ohlc[:, 1], color=INK, linewidth=0.7, zorder=2)
    # bodies: open-close. Hollow = up day, filled = down day (direction by fill,
    # not by color, so the encoding survives grayscale and color-vision deficiency).
    body_low = np.minimum(ohlc[:, 0], ohlc[:, 3])
    body_high = np.maximum(ohlc[:, 0], ohlc[:, 3])
    height = np.maximum(body_high - body_low, 1e-4)
    for i in range(len(ohlc)):
        face = "white" if up[i] else COLOR_VOLATILE
        ax.add_patch(
            plt.Rectangle(
                (x[i] - 0.32, body_low[i]),
                0.64,
                height[i],
                facecolor=face,
                edgecolor=COLOR_VOLATILE,
                linewidth=0.7,
                zorder=3,
            )
        )
    for level, label in (
        (stats_block["cap"], "cap"),
        (stats_block["baseline"], "baseline"),
        (stats_block["floor"], "floor"),
    ):
        ax.axhline(level, color=GUIDE, linestyle="--", linewidth=0.8, zorder=1)
        ax.text(len(ohlc) - 1, level, f" {label}", va="center", ha="left", color=GUIDE, fontsize=8)
    ax.set_xlim(-1, len(ohlc) + 6)
    ax.set_xlabel(f"Day (first {len(ohlc)} days, representative window)", color=INK)
    ax.set_ylabel("Price (EUR/kWh)", color=INK)
    ax.set_title(
        "Volatile broker daily price (open-high-low-close); hollow = up day, filled = down day",
        color=INK,
        fontsize=11,
    )


def _plot_innovation_hist(ax, innovations: np.ndarray, stats_block: dict) -> None:
    _style_axis(ax)
    sigma = float(innovations.std(ddof=0))
    lim = float(np.quantile(np.abs(innovations), 0.999))
    bins = np.linspace(-lim, lim, 61)
    ax.hist(innovations, bins=bins, color=COLOR_VOLATILE, alpha=0.55, log=True, label="Innovations (Student-t)")
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    gauss_counts = len(innovations) * width * stats.norm.pdf(centers, loc=0.0, scale=sigma)
    ax.plot(centers, gauss_counts, color=INK, linestyle="--", linewidth=1.5, label="Gaussian, equal std")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Hourly price innovation (EUR/kWh)", color=INK)
    ax.set_ylabel("Count (log scale)", color=INK)
    ax.set_title("Innovations have heavier tails than a Gaussian", color=INK, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    excess_kurtosis = float(stats.kurtosis(innovations, fisher=True))
    ax.text(
        0.02,
        0.96,
        f"df = {stats_block['df']}\nsample excess kurtosis = {excess_kurtosis:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=INK,
        fontsize=8,
    )


def _plot_qq(ax, innovations: np.ndarray) -> None:
    _style_axis(ax)
    theoretical, ordered = stats.probplot(innovations, dist="norm", fit=False)
    ax.scatter(theoretical, ordered, s=5, color=COLOR_VOLATILE, alpha=0.5, zorder=2)
    sigma = float(innovations.std(ddof=0))
    line_x = np.array([theoretical.min(), theoretical.max()])
    ax.plot(line_x, sigma * line_x, color=INK, linestyle="--", linewidth=1.3, zorder=3, label="Gaussian reference")
    ax.set_xlabel("Standard normal quantiles", color=INK)
    ax.set_ylabel("Innovation quantiles (EUR/kWh)", color=INK)
    ax.set_title("Normal Q-Q: tails bend away from the line", color=INK, fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def plot_volatile_detail(price_vol, innovations, stats_block, out_path: Path) -> None:
    window_hours = CANDLE_WINDOW_DAYS * HOURS_PER_DAY
    ohlc = _daily_ohlc(price_vol[:window_hours])

    fig = plt.figure(figsize=(11, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22)
    ax_candle = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_qq = fig.add_subplot(gs[1, 1])

    _plot_candles(ax_candle, ohlc, stats_block)
    _plot_innovation_hist(ax_hist, innovations, stats_block)
    _plot_qq(ax_qq, innovations)

    caption = (
        f"Volatile Low-Cost broker (D5): baseline {stats_block['baseline']:.2f}, AR(1) phi "
        f"{stats_block['phi']:.2f}, Student-t(df={stats_block['df']}) innovations scaled by "
        f"{stats_block['shock_scale']:.3f}, clipped to [{stats_block['floor']:.2f}, "
        f"{stats_block['cap']:.2f}] EUR/kWh. Realized over one year: mean "
        f"{stats_block['price_mean']:.4f}, std {stats_block['price_std']:.4f}, min "
        f"{stats_block['price_min']:.3f}, max {stats_block['price_max']:.3f}; floor hit "
        f"{stats_block['floor_hit_pct']:.2f} percent of hours, cap hit "
        f"{stats_block['cap_hit_pct']:.3f} percent."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8, color=INK, wrap=True)
    fig.suptitle("Volatile broker price: heavy-tailed, mean-reverting, floor-and-cap bounded", fontsize=13, color=INK)
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.08, right=0.95, hspace=0.42, wspace=0.22)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    config = load_config(str(CONFIG_PATH))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    price_reg, price_grn, price_vol, innovations, stats_block = realize_prices(config)

    print(
        "[prices] volatile realized: mean=%.4f std=%.4f min=%.3f max=%.3f floor_hit=%.2f%% cap_hit=%.3f%%"
        % (
            stats_block["price_mean"],
            stats_block["price_std"],
            stats_block["price_min"],
            stats_block["price_max"],
            stats_block["floor_hit_pct"],
            stats_block["cap_hit_pct"],
        )
    )

    archetypes_path = PLOTS_DIR / "fig_broker_price_archetypes.png"
    detail_path = PLOTS_DIR / "fig_volatile_price_candles.png"
    plot_archetypes(price_reg, price_grn, price_vol, archetypes_path)
    plot_volatile_detail(price_vol, innovations, stats_block, detail_path)
    print(f"[prices] wrote {archetypes_path}")
    print(f"[prices] wrote {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
