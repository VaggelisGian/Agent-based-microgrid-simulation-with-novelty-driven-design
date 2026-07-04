import numpy as np
import pytest

from microgrid_sim.data.loaders import (
    DataLoadError,
    generate_synthetic_demand,
    generate_synthetic_solar,
    load_profiles,
    tile_to_horizon,
    validate_series,
)


def test_generate_synthetic_solar_is_deterministic():
    a = generate_synthetic_solar(hours=8760)
    b = generate_synthetic_solar(hours=8760)
    assert np.array_equal(a, b)


def test_generate_synthetic_solar_no_negative_values():
    solar = generate_synthetic_solar(hours=8760)
    assert np.all(solar >= 0.0)


def test_generate_synthetic_solar_zero_at_deep_night():
    solar = generate_synthetic_solar(hours=8760)
    hour_of_day = np.arange(8760) % 24
    deep_night = np.isin(hour_of_day, [1, 2, 3])
    assert np.all(solar[deep_night] < 1e-9)


def test_generate_synthetic_solar_has_daytime_generation():
    solar = generate_synthetic_solar(hours=8760)
    assert solar.max() > 0.1


def test_generate_synthetic_demand_is_deterministic():
    a = generate_synthetic_demand(hours=8760)
    b = generate_synthetic_demand(hours=8760)
    assert np.array_equal(a, b)


def test_generate_synthetic_demand_no_negative_or_nan():
    demand = generate_synthetic_demand(hours=8760)
    assert np.all(demand >= 0.0)
    assert not np.any(np.isnan(demand))


def test_generate_synthetic_demand_has_morning_and_evening_peaks():
    demand = generate_synthetic_demand(hours=24 * 30)
    hour_of_day = np.arange(24 * 30) % 24
    by_hour_mean = np.array([demand[hour_of_day == h].mean() for h in range(24)])
    night_mean = by_hour_mean[[2, 3, 4]].mean()
    assert by_hour_mean[8] > night_mean
    assert by_hour_mean[20] > night_mean


def test_tile_to_horizon_truncates_longer_series():
    series = np.arange(100, dtype=float)
    result = tile_to_horizon(series, 10)
    assert len(result) == 10
    assert np.array_equal(result, series[:10])


def test_tile_to_horizon_tiles_shorter_series():
    series = np.array([1.0, 2.0, 3.0])
    result = tile_to_horizon(series, 7)
    assert len(result) == 7
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))


def test_tile_to_horizon_rejects_empty_series():
    with pytest.raises(DataLoadError):
        tile_to_horizon(np.array([]), 10)


def test_validate_series_rejects_nan():
    series = np.array([1.0, np.nan, 3.0])
    with pytest.raises(DataLoadError):
        validate_series(series, name="test_series")


def test_validate_series_rejects_negative_when_disallowed():
    series = np.array([1.0, -0.5, 3.0])
    with pytest.raises(DataLoadError):
        validate_series(series, name="test_series", allow_negative=False)


def test_validate_series_accepts_clean_series():
    series = np.array([1.0, 0.0, 3.0])
    validate_series(series, name="test_series")


def test_load_profiles_returns_requested_horizon_length(tmp_path):
    config = {
        "data": {
            "solar_profile_path": str(tmp_path / "solar.csv"),
            "demand_profile_path": str(tmp_path / "demand.csv"),
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    profiles = load_profiles(config, horizon_hours=336, allow_network_fetch=False)
    assert len(profiles.solar_kw_per_kwp) == 336
    assert len(profiles.demand_kwh_reference) == 336
    assert profiles.demand_source == "generated_sample"
    # allow_network_fetch=False with no cache present guarantees this can only
    # ever be "generated_sample" (F8: was a loose membership check that would
    # also have passed if the code path could never actually reach pvgis_fetch).
    assert profiles.solar_source == "generated_sample"


def test_load_profiles_warns_on_synthetic_fallback_cache_build(tmp_path):
    # F5 fix: building a fresh cache from a synthetic fallback (no existing
    # cache, no real fetch) must be loud, not silent.
    config = {
        "data": {
            "solar_profile_path": str(tmp_path / "solar.csv"),
            "demand_profile_path": str(tmp_path / "demand.csv"),
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    with pytest.warns(RuntimeWarning, match="synthetic fallback"):
        load_profiles(config, horizon_hours=48, allow_network_fetch=False)


def test_load_profiles_no_warning_when_cache_already_exists(tmp_path):
    config = {
        "data": {
            "solar_profile_path": str(tmp_path / "solar.csv"),
            "demand_profile_path": str(tmp_path / "demand.csv"),
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    load_profiles(config, horizon_hours=48, allow_network_fetch=False)  # builds and caches
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        load_profiles(config, horizon_hours=48, allow_network_fetch=False)  # loads from cache, no warning


def test_load_profiles_caches_to_disk_and_reload_matches(tmp_path):
    config = {
        "data": {
            "solar_profile_path": str(tmp_path / "solar.csv"),
            "demand_profile_path": str(tmp_path / "demand.csv"),
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    first = load_profiles(config, horizon_hours=200, allow_network_fetch=False)
    assert (tmp_path / "solar.csv").exists()
    assert (tmp_path / "demand.csv").exists()

    second = load_profiles(config, horizon_hours=200, allow_network_fetch=False)
    assert np.allclose(first.solar_kw_per_kwp, second.solar_kw_per_kwp)
    assert np.allclose(first.demand_kwh_reference, second.demand_kwh_reference)
    assert second.solar_source == "generated_sample"


def test_load_profiles_raises_on_horizon_mismatch_after_bad_cache(tmp_path):
    solar_path = tmp_path / "solar.csv"
    solar_path.write_text("# source: generated_sample\nhour,solar_kw_per_kwp\n0,not_a_number\n")
    demand_path = tmp_path / "demand.csv"
    demand_path.write_text("# source: generated_sample\nhour,demand_kwh_reference\n0,1.0\n")
    config = {
        "data": {
            "solar_profile_path": str(solar_path),
            "demand_profile_path": str(demand_path),
        }
    }
    with pytest.raises(DataLoadError):
        load_profiles(config, horizon_hours=10, allow_network_fetch=False)
