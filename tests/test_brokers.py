import numpy as np
import pytest

from microgrid_sim.brokers.base import Broker
from microgrid_sim.brokers.premium_green import PremiumGreenBroker
from microgrid_sim.brokers.regulated_utility import RegulatedUtilityBroker
from microgrid_sim.brokers.volatile_low_cost import VolatileLowCostBroker


def test_broker_rejects_greenness_out_of_range():
    with pytest.raises(ValueError):
        Broker("b1", "Broker One", "generic", greenness=1.5)


def test_broker_price_volatility_zero_with_fewer_than_two_samples():
    broker = Broker("b1", "Broker One", "generic", greenness=0.5)
    assert broker.price_volatility() == 0.0
    broker._record_price(0.2)
    assert broker.price_volatility() == 0.0


def test_broker_price_volatility_positive_with_varying_prices():
    broker = Broker("b1", "Broker One", "generic", greenness=0.5)
    for price in (0.10, 0.30, 0.10, 0.30):
        broker._record_price(price)
    assert broker.price_volatility() > 0.0


def test_broker_record_sale_tracks_served_energy_and_revenue():
    broker = Broker("b1", "Broker One", "generic", greenness=0.5)
    broker.record_sale(price_eur_per_kwh=0.20, energy_kwh=10.0)
    broker.record_sale(price_eur_per_kwh=0.20, energy_kwh=-4.0)  # net export credit
    assert broker.cumulative_energy_served_kwh == pytest.approx(10.0)  # export not counted as load
    assert broker.cumulative_net_energy_kwh == pytest.approx(6.0)  # signed: 10.0 + -4.0 (F1)
    assert broker.cumulative_revenue_eur == pytest.approx(0.20 * 10.0 + 0.20 * -4.0)


def _make_regulated(seasonal_amplitude=0.02):
    return RegulatedUtilityBroker(
        "regulated_utility",
        "Regulated Utility",
        greenness=0.3,
        base_eur_per_kwh=0.19,
        seasonal_amplitude_eur_per_kwh=seasonal_amplitude,
        seasonal_peak_day=15,
    )


def test_regulated_utility_price_stays_within_seasonal_band():
    broker = _make_regulated()
    prices = [broker.quote(hour) for hour in range(24 * 365)]
    assert min(prices) >= 0.19 - 0.02 - 1e-9
    assert max(prices) <= 0.19 + 0.02 + 1e-9


def test_regulated_utility_is_flat_within_a_single_day():
    broker = _make_regulated()
    prices = [broker.quote(hour) for hour in range(24)]
    assert max(prices) - min(prices) < 1e-9


def test_regulated_utility_peaks_near_configured_seasonal_peak_day():
    broker = _make_regulated()
    price_at_peak = broker.quote(24 * 15)
    broker2 = _make_regulated()
    price_at_midyear = broker2.quote(24 * (15 + 182))
    assert price_at_peak > price_at_midyear


def test_regulated_utility_is_deterministic_and_has_no_intrinsic_randomness():
    broker_a = _make_regulated()
    broker_b = _make_regulated()
    prices_a = [broker_a.quote(h) for h in range(200)]
    prices_b = [broker_b.quote(h) for h in range(200)]
    assert prices_a == prices_b


def _make_volatile(seed=1):
    rng = np.random.default_rng(seed)
    return VolatileLowCostBroker(
        "volatile_low_cost",
        "Volatile Low-Cost",
        greenness=0.1,
        baseline_eur_per_kwh=0.12,
        mean_reversion_phi=0.85,
        shock_scale_eur_per_kwh=0.03,
        shock_degrees_of_freedom=3,
        price_floor_eur_per_kwh=0.03,
        price_cap_multiplier=4.0,
        rng=rng,
    )


def test_volatile_broker_cheaper_on_average_than_regulated():
    regulated = _make_regulated()
    volatile = _make_volatile()
    regulated_prices = [regulated.quote(h) for h in range(24 * 365)]
    volatile_prices = [volatile.quote(h) for h in range(24 * 365)]
    assert np.mean(volatile_prices) < np.mean(regulated_prices)


def test_volatile_broker_is_spikier_than_regulated():
    regulated = _make_regulated()
    volatile = _make_volatile()
    regulated_prices = [regulated.quote(h) for h in range(24 * 365)]
    volatile_prices = [volatile.quote(h) for h in range(24 * 365)]
    assert np.std(volatile_prices) > np.std(regulated_prices)


def test_volatile_broker_respects_floor_and_cap():
    volatile = _make_volatile()
    prices = [volatile.quote(h) for h in range(24 * 365)]
    assert min(prices) >= 0.03 - 1e-9
    assert max(prices) <= 0.12 * 4.0 + 1e-9


def test_volatile_broker_same_seed_reproducible():
    volatile_a = _make_volatile(seed=42)
    volatile_b = _make_volatile(seed=42)
    prices_a = [volatile_a.quote(h) for h in range(100)]
    prices_b = [volatile_b.quote(h) for h in range(100)]
    assert prices_a == prices_b


def test_volatile_broker_different_seed_differs():
    volatile_a = _make_volatile(seed=1)
    volatile_b = _make_volatile(seed=2)
    prices_a = [volatile_a.quote(h) for h in range(100)]
    prices_b = [volatile_b.quote(h) for h in range(100)]
    assert prices_a != prices_b


def test_premium_green_is_flat_markup_over_regulated_base():
    broker = PremiumGreenBroker(
        "premium_green",
        "Premium Green",
        greenness=0.95,
        regulated_base_eur_per_kwh=0.19,
        markup_eur_per_kwh=0.04,
    )
    prices = [broker.quote(h) for h in range(24 * 30)]
    assert all(price == pytest.approx(0.23) for price in prices)


def test_premium_green_has_high_greenness():
    broker = PremiumGreenBroker(
        "premium_green", "Premium Green", greenness=0.95,
        regulated_base_eur_per_kwh=0.19, markup_eur_per_kwh=0.04,
    )
    assert broker.greenness >= 0.9


class _StubStandardTRng:
    """Duck-typed stand-in for np.random.Generator: only implements
    standard_t(df), returning a scripted sequence of raw (pre-shock-scale)
    values so a test can force an exact, deterministic innovation each call
    without depending on a real Generator's internals."""

    def __init__(self, values):
        self._values = list(values)

    def standard_t(self, df):
        if self._values:
            return self._values.pop(0)
        return 0.0


def test_volatile_broker_anti_windup_resets_latent_state_after_deep_shock():
    """Bundled regression test (Phase 2 re-review coverage gap, carried into
    Phase 3): guards the F4 anti-windup fix in VolatileLowCostBroker.quote.

    A single very large negative shock drives the raw (pre-clip) price far
    below the floor. Without the anti-windup line (which resets the latent
    AR(1) state to the CLIPPED price's implied value), the state would carry
    the full pre-clip excursion forward, decaying only geometrically (by
    mean_reversion_phi per hour) while the visible price stays pinned flat at
    the floor for many consecutive hours -- an artifact of the clip, not the
    intended heavy-tailed spiky dynamic (see docs/DECISIONS.md's F4 note).

    With the fix in place, the state resets to (floor - baseline) after the
    deep shock, so once the shock innovations return to 0 the visible price
    recovers above the floor within a hour or two, not ~14 hours later.
    """
    stub_rng = _StubStandardTRng([-50.0] + [0.0] * 50)
    broker = VolatileLowCostBroker(
        "volatile_low_cost",
        "Volatile Low-Cost",
        greenness=0.1,
        baseline_eur_per_kwh=0.12,
        mean_reversion_phi=0.85,
        shock_scale_eur_per_kwh=0.015,
        shock_degrees_of_freedom=3,
        price_floor_eur_per_kwh=0.03,
        price_cap_multiplier=4.0,
        rng=stub_rng,
    )

    prices = [broker.quote(h) for h in range(30)]

    assert prices[0] == pytest.approx(0.03)  # the deep negative shock clips to the floor

    consecutive_floor_hours_after_shock = 0
    for price in prices[1:]:
        if price <= 0.03 + 1e-9:
            consecutive_floor_hours_after_shock += 1
        else:
            break

    # With anti-windup this recovers essentially immediately (0 further floor
    # hours at this calibration); without it (state carries the pre-clip
    # -0.75 excursion, decaying by 0.85 per hour with zero further
    # innovations) it takes about 14 more hours to climb back above the
    # floor -- so this bound fails if the anti-windup reset line is removed.
    assert consecutive_floor_hours_after_shock <= 2
