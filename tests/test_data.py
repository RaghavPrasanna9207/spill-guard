import numpy as np
import pandas as pd
import pytest

from spillguard import config as cfg
from spillguard import data

if not (data.RAW / "olist_orders_dataset.csv").exists():
    pytest.skip("Olist CSVs not in data/raw", allow_module_level=True)

orders = data.load_orders()


def test_tiers_are_terciles_and_states_filtered():
    assert set(orders.state) <= set(cfg.DEST_STATES)
    shares = orders.tier.value_counts(normalize=True)
    assert set(shares.index) == {0, 1, 2}
    assert shares.between(0.30, 0.37).all()


def test_counts_lose_no_orders():
    start = orders.ts.min().floor("h")
    hours = int((orders.ts.max() - start) / pd.Timedelta(hours=1)) + 1
    assert data.counts(orders, start, hours).sum() == len(orders)


def test_peak_day_scales_to_target():
    assert data.actual_demand(orders)[:24].sum() == pytest.approx(cfg.PEAK_DAY_ORDERS)


def test_forecast_days_match_weekday_averages():
    # Independent check with pandas: mean Fri/Sat/Sun totals over the profile weeks.
    start = pd.Timestamp(cfg.WINDOW_START)
    hist = orders[(orders.ts >= start - pd.Timedelta(weeks=cfg.PROFILE_WEEKS)) & (orders.ts < start)]
    per_day = hist.groupby(hist.ts.dt.normalize()).size()
    expected = [per_day[per_day.index.day_of_week == (start.day_of_week + d) % 7].mean() for d in range(3)]
    fc = data.forecast_demand(orders).sum((1, 2)).reshape(3, 24).sum(1)
    np.testing.assert_allclose(fc, np.array(expected) * data.scale_factor(orders))


def test_surges_apply_to_their_own_days():
    base = data.forecast_demand(orders)
    fc = data.forecast_demand(orders, peak_surge=3, weekend_surge=2, buffer=1.1)
    np.testing.assert_allclose(fc[:24], base[:24] * 3.3)
    np.testing.assert_allclose(fc[24:], base[24:] * 2.2)
