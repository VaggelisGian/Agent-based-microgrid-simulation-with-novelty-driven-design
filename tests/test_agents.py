import mesa
import numpy as np
import pytest

from microgrid_sim.agents.consumer import Consumer
from microgrid_sim.agents.prosumer import Prosumer
from microgrid_sim.brokers.base import Broker


class StubBroker(Broker):
    """Test double with a fixed price and directly settable volatility."""

    def __init__(self, broker_id, greenness=0.5, price=0.20, volatility=0.0):
        super().__init__(broker_id, broker_id, "stub", greenness)
        self._price = price
        self._volatility = volatility

    def quote(self, hour, context=None):
        self._record_price(self._price)
        return self._price

    def price_volatility(self):
        return self._volatility


class HarnessModel(mesa.Model):
    """Minimal real mesa.Model host so Consumer/Prosumer (mesa.Agent subclasses)
    can be constructed and stepped without needing the full MicrogridModel."""

    def __init__(self, seed=1, horizon=200):
        super().__init__(seed=seed)
        self.current_hour = 0
        self.demand_profile = np.ones(horizon)
        self.solar_profile = np.zeros(horizon)
        self.current_prices = {}
        self.brokers = {}
        self.switching_enabled = True

    def step(self):
        pass


def _consumer_kwargs(**overrides):
    kwargs = dict(
        profile="price_sensitive",
        demand_scale=1.0,
        price_tolerance_eur_per_kwh=0.25,
        volatility_tolerance_eur_per_kwh=0.05,
        greenness_threshold=0.5,
        switching_penalty_eur_per_kwh=0.01,
        sustained_breach_hours=3,
        initial_broker=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_consumer_billing_is_demand_times_price():
    model = HarnessModel()
    model.demand_profile[0] = 2.0
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    consumer = Consumer(model, **_consumer_kwargs(demand_scale=1.5, initial_broker=broker))

    consumer.step()

    assert consumer.last_demand_kwh == pytest.approx(3.0)
    assert consumer.last_net_import_kwh == pytest.approx(3.0)
    assert consumer.last_billed_eur == pytest.approx(0.6)
    assert broker.cumulative_energy_served_kwh == pytest.approx(3.0)
    assert broker.cumulative_revenue_eur == pytest.approx(0.6)


def test_consumer_no_switch_before_tolerance_breach_sustained():
    model = HarnessModel()
    cheap = StubBroker("cheap", price=0.05)
    expensive = StubBroker("expensive", price=0.30)  # breaches tolerance 0.25 every hour
    model.brokers = {"cheap": cheap, "expensive": expensive}
    model.current_prices = {"cheap": 0.05, "expensive": 0.30}
    consumer = Consumer(model, **_consumer_kwargs(sustained_breach_hours=3, initial_broker=expensive))

    consumer.step()
    assert consumer.broker is expensive
    consumer.step()
    assert consumer.broker is expensive
    consumer.step()
    assert consumer.broker is cheap  # third consecutive breach hour triggers reconsideration
    assert consumer.switch_count == 1


def test_consumer_never_breaches_never_switches():
    model = HarnessModel()
    cheap = StubBroker("cheap", price=0.05)
    acceptable = StubBroker("acceptable", price=0.20)  # below tolerance 0.25, never breaches
    model.brokers = {"cheap": cheap, "acceptable": acceptable}
    model.current_prices = {"cheap": 0.05, "acceptable": 0.20}
    consumer = Consumer(model, **_consumer_kwargs(sustained_breach_hours=2, initial_broker=acceptable))

    for _ in range(20):
        consumer.step()

    assert consumer.broker is acceptable
    assert consumer.switch_count == 0


def test_switching_disabled_never_switches_even_under_breach():
    model = HarnessModel()
    model.switching_enabled = False
    cheap = StubBroker("cheap", price=0.05)
    expensive = StubBroker("expensive", price=0.30)
    model.brokers = {"cheap": cheap, "expensive": expensive}
    model.current_prices = {"cheap": 0.05, "expensive": 0.30}
    consumer = Consumer(model, **_consumer_kwargs(sustained_breach_hours=1, initial_broker=expensive))

    for _ in range(10):
        consumer.step()

    assert consumer.broker is expensive
    assert consumer.switch_count == 0


def test_switching_penalty_blocks_marginal_price_improvement():
    model = HarnessModel()
    current = StubBroker("current", price=0.30)  # breaches tolerance 0.25
    marginal = StubBroker("marginal", price=0.295)  # only 0.005 cheaper, penalty is 0.01
    model.brokers = {"current": current, "marginal": marginal}
    model.current_prices = {"current": 0.30, "marginal": 0.295}
    consumer = Consumer(
        model,
        **_consumer_kwargs(sustained_breach_hours=1, switching_penalty_eur_per_kwh=0.01, initial_broker=current),
    )

    consumer.step()

    assert consumer.broker is current
    assert consumer.switch_count == 0


def test_stability_oriented_flees_volatile_current_broker_even_if_pricier_alternative():
    model = HarnessModel()
    volatile_current = StubBroker("volatile", price=0.10, volatility=0.06)  # exceeds tolerance 0.05
    stable_pricier = StubBroker("stable", price=0.22, volatility=0.01)
    model.brokers = {"volatile": volatile_current, "stable": stable_pricier}
    model.current_prices = {"volatile": 0.10, "stable": 0.22}
    consumer = Consumer(
        model,
        **_consumer_kwargs(
            profile="stability_oriented",
            sustained_breach_hours=1,
            volatility_tolerance_eur_per_kwh=0.05,
            switching_penalty_eur_per_kwh=0.01,
            initial_broker=volatile_current,
        ),
    )

    consumer.step()

    assert consumer.broker is stable_pricier


def test_green_preferring_filters_by_greenness_threshold_then_minimizes_price():
    model = HarnessModel()
    dirty_cheap = StubBroker("dirty", price=0.05, volatility=0.0)
    dirty_cheap.greenness = 0.2  # below threshold 0.5
    green_expensive = StubBroker("green_expensive", price=0.25, volatility=0.0)
    green_expensive.greenness = 0.9
    green_cheaper = StubBroker("green_cheaper", price=0.21, volatility=0.0)
    green_cheaper.greenness = 0.6
    model.brokers = {"dirty": dirty_cheap, "green_expensive": green_expensive, "green_cheaper": green_cheaper}
    model.current_prices = {"dirty": 0.05, "green_expensive": 0.25, "green_cheaper": 0.21}
    consumer = Consumer(
        model,
        **_consumer_kwargs(
            profile="green_preferring",
            sustained_breach_hours=1,
            greenness_threshold=0.5,
            switching_penalty_eur_per_kwh=0.01,
            initial_broker=dirty_cheap,
        ),
    )

    consumer.step()

    assert consumer.broker is green_cheaper


def _prosumer_kwargs(**overrides):
    kwargs = dict(
        profile="price_sensitive",
        demand_scale=1.0,
        price_tolerance_eur_per_kwh=0.25,
        volatility_tolerance_eur_per_kwh=0.05,
        greenness_threshold=0.5,
        switching_penalty_eur_per_kwh=0.01,
        sustained_breach_hours=1000,
        initial_broker=None,
        pv_capacity_kwp=1.0,
        battery_capacity_kwh=5.0,
        reserve_fraction=0.2,
        evening_reserve_hour=18,
    )
    kwargs.update(overrides)
    return kwargs


def test_prosumer_charges_battery_from_surplus_and_zero_import():
    model = HarnessModel()
    model.demand_profile[0] = 1.0
    model.solar_profile[0] = 2.0
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    prosumer = Prosumer(model, **_prosumer_kwargs(battery_capacity_kwh=5.0, reserve_fraction=0.2, initial_broker=broker))

    prosumer.step()

    assert prosumer.battery_soc_kwh == pytest.approx(2.0)  # reserve start 1.0 + 1.0 kWh charged
    assert prosumer.last_net_import_kwh == pytest.approx(0.0)
    assert prosumer.last_billed_eur == pytest.approx(0.0)


def test_prosumer_exports_surplus_beyond_battery_headroom():
    model = HarnessModel()
    model.demand_profile[0] = 1.0
    model.solar_profile[0] = 3.0
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    prosumer = Prosumer(
        model, **_prosumer_kwargs(pv_capacity_kwp=1.0, battery_capacity_kwh=2.0, reserve_fraction=0.2, initial_broker=broker)
    )

    prosumer.step()

    assert prosumer.battery_soc_kwh == pytest.approx(2.0)  # full
    assert prosumer.last_net_import_kwh == pytest.approx(-0.4)  # unstored surplus exported (net-metering credit)
    assert prosumer.last_billed_eur == pytest.approx(-0.4 * 0.20)


def test_prosumer_holds_reserve_before_evening_hour():
    model = HarnessModel()
    model.current_hour = 10  # before evening_reserve_hour=18
    model.demand_profile[10] = 3.0
    model.solar_profile[10] = 0.0
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    prosumer = Prosumer(
        model, **_prosumer_kwargs(battery_capacity_kwh=5.0, reserve_fraction=0.3, evening_reserve_hour=18, initial_broker=broker)
    )
    prosumer.battery_soc_kwh = 4.0

    prosumer.step()

    assert prosumer.battery_soc_kwh == pytest.approx(1.5)  # stopped exactly at reserve level (0.3*5.0)
    assert prosumer.last_net_import_kwh == pytest.approx(0.5)  # 3.0 deficit - 2.5 dischargeable


def test_prosumer_releases_reserve_after_evening_hour():
    model = HarnessModel()
    model.current_hour = 20  # at/after evening_reserve_hour=18
    model.demand_profile[20] = 3.0
    model.solar_profile[20] = 0.0
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    prosumer = Prosumer(
        model, **_prosumer_kwargs(battery_capacity_kwh=5.0, reserve_fraction=0.3, evening_reserve_hour=18, initial_broker=broker)
    )
    prosumer.battery_soc_kwh = 4.0

    prosumer.step()

    assert prosumer.battery_soc_kwh == pytest.approx(1.0)
    assert prosumer.last_net_import_kwh == pytest.approx(0.0)


def test_prosumer_self_sufficiency_accumulators_track_demand_and_grid_import():
    model = HarnessModel(horizon=3)
    model.demand_profile[:] = [2.0, 2.0, 2.0]
    model.solar_profile[:] = [0.0, 0.0, 0.0]
    broker = StubBroker("b1", price=0.20)
    model.brokers = {"b1": broker}
    model.current_prices = {"b1": 0.20}
    prosumer = Prosumer(
        model, **_prosumer_kwargs(battery_capacity_kwh=1.0, reserve_fraction=0.0, evening_reserve_hour=0, initial_broker=broker)
    )
    prosumer.battery_soc_kwh = 1.0

    for hour in range(3):
        model.current_hour = hour
        prosumer.step()

    assert prosumer.total_demand_kwh == pytest.approx(6.0)
    assert prosumer.total_grid_import_kwh == pytest.approx(5.0)  # 1 kWh of the 6 came from the battery
