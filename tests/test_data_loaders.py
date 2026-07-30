import numpy as np
import pandas as pd
import pytest

from microgrid_sim.data.loaders import (
    DataLoadError,
    OpsdFetchError,
    _align_to_utc_midnight_multiple_of_24,
    _extract_opsd_hourly_consumption,
    _validate_opsd_alignment,
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


# ---------------------------------------------------------------------------
# data.demand_series config switch (opt-in real OPSD household-load series;
# see docs/DECISIONS.md's "Data provenance" section and
# scripts/fetch_opsd_demand.py). This is a distinct config key from
# MicrogridModel.demand_source (the provenance marker read back from a
# cache file), deliberately named differently so the two are never confused.
# ---------------------------------------------------------------------------


def _opsd_style_cache_config(tmp_path, demand_series=None):
    config = {
        "data": {
            "solar_profile_path": str(tmp_path / "solar.csv"),
            "demand_profile_path": str(tmp_path / "demand.csv"),
            "demand_profile_path_opsd": str(tmp_path / "demand_opsd.csv"),
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    if demand_series is not None:
        config["data"]["demand_series"] = demand_series
    return config


def _write_fake_opsd_cache(path, hours=8760, seed=99):
    rng = np.random.default_rng(seed)
    series = rng.uniform(0.1, 1.0, size=hours)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="") as handle:
        handle.write("# source: opsd_household_fetch\n")
        handle.write("hour,demand_kwh_reference\n")
        for i, value in enumerate(series):
            handle.write(f"{i},{value}\n")
    return series


def test_demand_series_absent_key_matches_explicit_synthetic_default(tmp_path):
    # Default is unchanged: absent data.demand_series behaves exactly like an
    # explicit "synthetic" (both read the same synthetic cache path).
    config_absent = _opsd_style_cache_config(tmp_path)
    config_explicit = _opsd_style_cache_config(tmp_path, demand_series="synthetic")

    profiles_absent = load_profiles(config_absent, horizon_hours=336, allow_network_fetch=False)
    profiles_explicit = load_profiles(config_explicit, horizon_hours=336, allow_network_fetch=False)

    assert profiles_absent.demand_source == "generated_sample"
    assert profiles_explicit.demand_source == "generated_sample"
    # allclose, not array_equal: profiles_absent's array is the freshly built
    # in-memory series, while profiles_explicit's is read back from the CSV
    # cache profiles_absent just wrote, which round-trips through text
    # (matches test_load_profiles_caches_to_disk_and_reload_matches above).
    assert np.allclose(profiles_absent.demand_kwh_reference, profiles_explicit.demand_kwh_reference)


def test_demand_series_opsd_selects_the_opsd_cache_file(tmp_path):
    opsd_path = tmp_path / "demand_opsd.csv"
    fake_series = _write_fake_opsd_cache(opsd_path)
    config = _opsd_style_cache_config(tmp_path, demand_series="opsd")

    profiles = load_profiles(config, horizon_hours=336, allow_network_fetch=False)

    assert profiles.demand_source == "opsd_household_fetch"
    assert np.allclose(profiles.demand_kwh_reference, fake_series[:336])
    # Confirms the synthetic path was never touched: no synthetic demand
    # cache file was created for this config.
    assert not (tmp_path / "demand.csv").exists()


def test_demand_series_unknown_value_raises_clear_error(tmp_path):
    config = _opsd_style_cache_config(tmp_path, demand_series="bogus_value")
    with pytest.raises(DataLoadError, match="demand_series"):
        load_profiles(config, horizon_hours=336, allow_network_fetch=False)


def test_demand_series_opsd_missing_cache_and_no_network_raises_not_fabricates(tmp_path):
    # No cache file at demand_profile_path_opsd, and allow_network_fetch is
    # False: this must raise, never silently build a synthetic substitute
    # and call it "opsd".
    config = _opsd_style_cache_config(tmp_path, demand_series="opsd")
    with pytest.raises(DataLoadError, match="opsd"):
        load_profiles(config, horizon_hours=336, allow_network_fetch=False)


def test_demand_series_opsd_series_passes_same_validation_as_synthetic(tmp_path):
    opsd_path = tmp_path / "demand_opsd.csv"
    _write_fake_opsd_cache(opsd_path, hours=100)  # shorter than the requested horizon, forces tiling
    config = _opsd_style_cache_config(tmp_path, demand_series="opsd")

    profiles = load_profiles(config, horizon_hours=250, allow_network_fetch=False)

    assert len(profiles.demand_kwh_reference) == 250
    assert np.all(np.isfinite(profiles.demand_kwh_reference))
    assert np.all(profiles.demand_kwh_reference >= 0.0)
    # Tiling wrapped the 100-hour series to fill 250 hours (matches tile_to_horizon).
    assert np.allclose(profiles.demand_kwh_reference[100:200], profiles.demand_kwh_reference[0:100])


def test_shipped_opsd_cache_file_loads_and_validates_through_existing_loader():
    # The real, committed data/samples/residential_demand_opsd_hourly.csv,
    # loaded exactly the way MicrogridModel loads it (allow_network_fetch is
    # always False there): must be present, load cleanly, carry the real
    # fetch's source marker, and pass the same validation as the synthetic
    # series (finite, non-negative, tiles/truncates to the requested horizon).
    config = {
        "data": {
            "solar_profile_path": "data/samples/solar_thessaloniki_hourly.csv",
            "demand_profile_path": "data/samples/residential_demand_thessaloniki_hourly.csv",
            "demand_series": "opsd",
            "solar_lat": 40.64,
            "solar_lon": 22.95,
        }
    }
    profiles = load_profiles(config, horizon_hours=8760, allow_network_fetch=False)

    assert profiles.demand_source == "opsd_household_fetch"
    assert len(profiles.demand_kwh_reference) == 8760
    assert np.all(np.isfinite(profiles.demand_kwh_reference))
    assert np.all(profiles.demand_kwh_reference >= 0.0)


# ---------------------------------------------------------------------------
# UTC-midnight alignment (review-found bug: the OPSD series was previously
# cached starting at whatever timestamp its raw coverage window began, not
# at 00:00 UTC, phase-shifting it by hours against the solar series' and the
# synthetic series' shared index-0-is-midnight convention).
# ---------------------------------------------------------------------------


def test_align_to_utc_midnight_derives_offset_from_timestamps_not_assumed():
    # 5 leading hours (utc hour 19, 20, 21, 22, 23) before the first
    # midnight, then 30 hours of aligned data (one full day plus 6 hours),
    # so both the leading trim (5) and the trailing multiple-of-24 trim (6)
    # are exercised in one case.
    timestamps = pd.date_range("2020-01-01T19:00:00Z", periods=35, freq="h").astype(str).to_numpy()
    consumption = np.arange(1.0, 36.0)  # 1.0 .. 35.0, strictly increasing so misalignment is obvious

    aligned, aligned_ts, offset = _align_to_utc_midnight_multiple_of_24(consumption, timestamps, min_hours=24)

    assert offset == 5  # derived: 5 rows (19,20,21,22,23) before the first hour==0
    assert aligned.size == 24  # 30 remaining hours trimmed down to the nearest whole day
    assert pd.Timestamp(str(aligned_ts[0])).hour == 0
    assert np.array_equal(aligned, consumption[5:29])  # exactly the derived window, nothing guessed


def test_align_to_utc_midnight_raises_if_no_midnight_in_window():
    timestamps = pd.date_range("2020-01-01T05:00:00Z", periods=10, freq="h").astype(str).to_numpy()
    consumption = np.arange(10.0)
    with pytest.raises(OpsdFetchError, match="UTC-midnight"):
        _align_to_utc_midnight_multiple_of_24(consumption, timestamps, min_hours=1)


def test_align_to_utc_midnight_raises_if_too_short_after_trim():
    timestamps = pd.date_range("2020-01-01T23:00:00Z", periods=25, freq="h").astype(str).to_numpy()
    consumption = np.arange(25.0)
    # After dropping the 1 leading hour, 24 hours remain, which is below a
    # min_hours requirement of 48.
    with pytest.raises(OpsdFetchError):
        _align_to_utc_midnight_multiple_of_24(consumption, timestamps, min_hours=48)


def test_validate_opsd_alignment_rejects_non_midnight_start():
    timestamps = pd.date_range("2020-01-01T01:00:00Z", periods=24, freq="h").astype(str).to_numpy()
    with pytest.raises(OpsdFetchError, match="midnight"):
        _validate_opsd_alignment(np.arange(24.0), timestamps)


def test_validate_opsd_alignment_rejects_length_not_multiple_of_24():
    timestamps = pd.date_range("2020-01-01T00:00:00Z", periods=25, freq="h").astype(str).to_numpy()
    with pytest.raises(OpsdFetchError, match="multiple of 24"):
        _validate_opsd_alignment(np.arange(25.0), timestamps)


def test_validate_opsd_alignment_accepts_well_formed_series():
    timestamps = pd.date_range("2020-01-01T00:00:00Z", periods=48, freq="h").astype(str).to_numpy()
    _validate_opsd_alignment(np.arange(48.0), timestamps)  # must not raise


def _mean_by_hour_of_day(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, comment="#")
    column = [c for c in df.columns if c != "hour"][0]
    values = df[column].to_numpy(dtype=float)
    hour_of_day = np.arange(values.size) % 24
    return np.array([values[hour_of_day == h].mean() for h in range(24)])


def test_shipped_opsd_cache_length_is_multiple_of_24_and_starts_at_midnight():
    df = pd.read_csv("data/samples/residential_demand_opsd_hourly.csv", comment="#")
    assert len(df) % 24 == 0
    with open("data/samples/residential_demand_opsd_hourly.csv", "r", encoding="ascii") as handle:
        handle.readline()  # "# source: ..." marker
        handle.readline()  # header
        first_data_line = handle.readline()
    # hour 0 is present as the very first data row (index 0 == 00:00 UTC by
    # construction; the file itself does not carry timestamps, only the
    # 0-based "hour" index that MicrogridModel indexes by).
    assert first_data_line.startswith("0,")


def test_shipped_opsd_cache_evening_peak_and_overnight_trough_align_with_synthetic_and_solar():
    # History of two bugs this test guards against, in the order they were
    # found (both in review, before either shipped series was used in a
    # run):
    #
    # 1. Before _align_to_utc_midnight_multiple_of_24 existed, the OPSD
    #    series was not aligned to a UTC-midnight boundary at all: it peaked
    #    at index 5 (early morning) and troughed at index 14
    #    (mid-afternoon), roughly 12-14 hours out of phase with the
    #    synthetic demand series and with solar's own hour-of-day
    #    convention. The evening_band/overnight_band checks below were
    #    written to catch a regression of that.
    #
    # 2. A second, one-hour bug in _extract_opsd_hourly_consumption (fixed
    #    later: interval consumption values were tagged with the LATER
    #    row's timestamp instead of the interval-START timestamp OPSD's own
    #    datapackage.json documents) shifted every value one hour later than
    #    it belonged, moving the peak from 16 to 17 and the trough from 1 to
    #    2. Both of those still land inside the 6-hour-wide bands below, so
    #    this test passed straight through that bug without catching it.
    #
    # The exact-hour assertions below close that gap: they pin the peak and
    # trough to the specific hours the corrected pipeline produces, so a
    # one-hour (or few-hour) phase regression fails here instead of passing.
    #
    # What this test CAN catch: any drift of the shipped cache's phase away
    # from the pinned hours, including a one-hour shift like bug 2 above.
    # What it CANNOT catch: it checks the shipped, already-built cache file
    # for consistency/regression; it does not independently re-derive ground
    # truth from the raw OPSD file (which is deleted after every fetch and
    # is not available at test time), so it cannot distinguish "correct" from
    # "wrong in a way that was baked into the pinned expectation itself".
    # test_extract_opsd_hourly_consumption_labels_intervals_by_interval_start_timestamp
    # below closes that remaining gap: it checks the labelling convention
    # directly, against a small synthetic counter with known timestamps, with
    # no dependency on the shipped cache at all.
    opsd = _mean_by_hour_of_day("data/samples/residential_demand_opsd_hourly.csv")
    synthetic = _mean_by_hour_of_day("data/samples/residential_demand_thessaloniki_hourly.csv")
    solar = _mean_by_hour_of_day("data/samples/solar_thessaloniki_hourly.csv")

    opsd_peak_hour = int(np.argmax(opsd))
    opsd_trough_hour = int(np.argmin(opsd))
    synthetic_peak_hour = int(np.argmax(synthetic))

    df = pd.read_csv("data/samples/residential_demand_opsd_hourly.csv", comment="#")
    assert len(df) % 24 == 0, "shipped OPSD cache length must be a whole multiple of 24 hours"

    # Exact-hour pins, not a band: re-derive with _mean_by_hour_of_day above
    # if data/samples/residential_demand_opsd_hourly.csv is ever legitimately
    # refreshed; do not loosen this to make a real phase regression pass.
    assert opsd_peak_hour == 16, f"OPSD peak hour {opsd_peak_hour} != 16 (pinned); check for a phase regression"
    assert opsd_trough_hour == 1, f"OPSD trough hour {opsd_trough_hour} != 1 (pinned); check for a phase regression"

    evening_band = set(range(16, 23))  # 16:00-22:00, a generous plausible evening-peak window
    overnight_band = set(range(0, 6))  # 00:00-05:00, a generous plausible overnight-trough window

    assert opsd_peak_hour in evening_band, f"OPSD peak hour {opsd_peak_hour} is not in the evening band {sorted(evening_band)}"
    assert opsd_trough_hour in overnight_band, (
        f"OPSD trough hour {opsd_trough_hour} is not in the overnight band {sorted(overnight_band)}"
    )
    # Both series' peaks should fall within a few hours of each other, not on
    # opposite sides of the clock (the original pre-alignment bug put them
    # about 12 hours apart: synthetic peak 20, buggy OPSD peak 5).
    hour_gap = min(abs(opsd_peak_hour - synthetic_peak_hour), 24 - abs(opsd_peak_hour - synthetic_peak_hour))
    assert hour_gap <= 6, f"OPSD peak hour {opsd_peak_hour} and synthetic peak hour {synthetic_peak_hour} are {hour_gap}h apart"

    # Solar's deep-night zero-generation hours (see loaders._DEEP_NIGHT_HOURS
    # and _validate_solar_zero_at_night) should be inside OPSD's low-demand
    # overnight hours, not its peak hours, confirming both series share the
    # same index-0-is-00:00-UTC clock.
    solar_zero_hours = {h for h in range(24) if solar[h] < 1e-9}
    assert opsd_peak_hour not in solar_zero_hours


def test_extract_opsd_hourly_consumption_labels_intervals_by_interval_start_timestamp(tmp_path):
    """Direct unit test of the interval-labelling convention documented in
    _extract_opsd_hourly_consumption: OPSD's datapackage.json defines
    utc_timestamp as the START of the hour a row covers, so the consumption
    value obtained by differencing row i and row i+1 (energy used DURING
    that hour) must be tagged with row i's timestamp (the interval start),
    never row i+1's (the interval end). Tagging with the later row's
    timestamp is exactly the bug this test exists to catch: it silently
    shifted every OPSD demand value one hour later than it belonged, and it
    was found in review before the series was ever used in a run (see
    docs/DECISIONS.md, "Data provenance").

    Uses a small synthetic cumulative counter with known, hand-picked
    timestamps (no network access, no dependency on the real OPSD file) and
    a deliberately small chunk_rows so one of the expected diffs straddles a
    chunk boundary, exercising the prev_ts carry across chunks, the exact
    mechanism the original bug lived in.
    """
    timestamps = pd.date_range("2020-01-01T00:00:00Z", periods=10, freq="h")
    counter = 100.0 + np.arange(10, dtype=float)  # cumulative meter reading, +1.0 kWh every hour
    raw_path = tmp_path / "fake_opsd_raw.csv"
    pd.DataFrame({"utc_timestamp": timestamps.astype(str), "meter": counter}).to_csv(raw_path, index=False)

    # chunk_rows=3 puts chunk boundaries after rows 2, 5, and 8 (0-based), so
    # the diff between row 2 (last row of the first chunk) and row 3 (first
    # row of the second chunk) can only be labelled correctly if prev_ts
    # survives the chunk boundary rather than resetting with each chunk.
    consumption, out_timestamps = _extract_opsd_hourly_consumption(
        raw_path, ["meter"], start_idx=0, end_idx=9, chunk_rows=3
    )

    assert consumption.size == 9  # 10 rows -> 9 row-to-row diffs
    assert np.allclose(consumption, 1.0)

    # Every returned timestamp must be the START of its interval: for the
    # diff between row i and row i+1, that is row i's timestamp.
    expected_start_timestamps = timestamps[:-1]
    actual_start_timestamps = pd.to_datetime(out_timestamps, utc=True)
    assert list(actual_start_timestamps) == list(expected_start_timestamps), (
        "each returned timestamp must be the interval START (the earlier row), not the interval "
        "END (the later row); a mismatch here means the one-hour mislabelling bug is back"
    )

    # The chunk-boundary diff specifically (row 2 -> row 3): must be tagged
    # with row 2's timestamp (02:00), not row 3's (03:00).
    boundary_diff_index = 2
    assert pd.Timestamp(str(out_timestamps[boundary_diff_index])) == timestamps[2]
