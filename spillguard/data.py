"""Olist orders -> hourly demand arrays D[hour, state, tier] for the 72h Black Friday window.

Two series are kept apart on purpose: the planning forecast (what a planner knew on
23 Nov 2017) and the realised demand. The gap between them is a project result.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from spillguard import config as cfg

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
S, V = len(cfg.DEST_STATES), len(cfg.TIER_NAMES)


def load_orders(raw_dir=RAW):
    """One row per SP-seller order to a destination state: ts, state, value (2026 R$), tier (0-2)."""
    read = lambda name: pd.read_csv(Path(raw_dir) / f"olist_{name}_dataset.csv")
    items = read("order_items").merge(read("sellers"), on="seller_id")
    value = items[items.seller_state == cfg.FC_STATE].groupby("order_id").price.sum().rename("value")
    o = read("orders").merge(read("customers"), on="customer_id").merge(value, on="order_id")
    o = o[~o.order_status.isin(cfg.EXCLUDED_STATUSES) & o.customer_state.isin(cfg.DEST_STATES)]
    cuts = o.value.quantile([1 / 3, 2 / 3]).to_numpy()
    return pd.DataFrame({
        "ts": pd.to_datetime(o.order_purchase_timestamp).to_numpy(),
        "state": o.customer_state.to_numpy(),
        "value": o.value.to_numpy() * cfg.IPCA_2017_TO_2026,
        "tier": np.searchsorted(cuts, o.value, side="right"),
    })


def counts(orders, start, hours):
    """Order counts as an array [hour, state, tier] for `hours` hours from `start`."""
    h = ((orders.ts - pd.Timestamp(start)) // pd.Timedelta(hours=1)).to_numpy()
    keep = (h >= 0) & (h < hours)
    s = orders.state.map({st: k for k, st in enumerate(cfg.DEST_STATES)}).to_numpy()
    out = np.zeros((hours, S, V))
    np.add.at(out, (h[keep], s[keep], orders.tier.to_numpy()[keep]), 1)
    return out


def scale_factor(orders):
    """Multiplier from Olist volume to FC volume: the Black Friday day becomes PEAK_DAY_ORDERS."""
    return cfg.PEAK_DAY_ORDERS / counts(orders, cfg.WINDOW_START, 24).sum()


def actual_demand(orders):
    """Realised Black Friday weekend demand at FC volume, as an expected hourly rate.

    The window holds ~1,300 Olist orders (~14 an hour) scaled ~27x, and 92% of hour x state x
    tier cells have 0-2 orders, so raw scaled counts are far lumpier than a 20k-a-day FC.
    Kept: each day's total, the hourly shape smoothed over ACTUAL_SMOOTH_H hours, and the
    window's state/tier mix. sim.py adds Poisson noise at FC scale.
    """
    raw = counts(orders, cfg.WINDOW_START, cfg.HOURS)
    hourly = raw.sum((1, 2))
    k = cfg.ACTUAL_SMOOTH_H
    smooth = np.convolve(np.pad(hourly, k // 2, mode="edge"), np.ones(k) / k, "valid")
    day = np.arange(cfg.HOURS) // 24
    smooth *= (np.bincount(day, hourly) / np.bincount(day, smooth))[day]  # keep each day's total
    mix = raw.sum(0) / raw.sum()
    return smooth[:, None, None] * mix * scale_factor(orders)


def forecast_demand(orders, peak_surge=1.0, weekend_surge=1.0, buffer=1.0):
    """Planning forecast from the PROFILE_WEEKS before Black Friday.

    weekday total x hour-of-day share x state/tier mix x surge x buffer, scaled to FC volume.
    Day 0 (Black Friday) uses peak_surge; the two weekend days use weekend_surge.
    """
    weeks = cfg.PROFILE_WEEKS
    hist_start = pd.Timestamp(cfg.WINDOW_START) - pd.Timedelta(weeks=weeks)  # same weekday
    hist = counts(orders, hist_start, weeks * 168)
    by_hour = hist.sum((1, 2)).reshape(weeks, 7, 24)  # day 0 = the window's weekday
    day_total = by_hour.sum(2).mean(0)
    hour_share = by_hour.sum((0, 1)) / by_hour.sum()
    t = np.arange(cfg.HOURS)
    surge = np.where(t < 24, peak_surge, weekend_surge)
    base = day_total[t // 24] * hour_share[t % 24] * surge
    mix = hist.sum(0) / hist.sum()
    return base[:, None, None] * mix * buffer * scale_factor(orders)


def tier_values(orders):
    """Mean order value (2026 R$) per value tier."""
    return orders.groupby("tier").value.mean().reindex(range(V)).to_numpy()


def hindsight_surges(orders):
    """(Black Friday, weekend) surges that make the forecast's day totals match realised demand.

    Known only after the fact: Olist has no orders in November 2016, so nothing in the data
    forecasts the 2017 spike.
    """
    days = cfg.HOURS // 24
    act, fc = (x.sum((1, 2)).reshape(days, 24).sum(1)
               for x in (actual_demand(orders), forecast_demand(orders)))
    return act[0] / fc[0], act[1:].sum() / fc[1:].sum()


def wape(forecast, actual):
    """Weighted absolute percentage error of hourly totals."""
    f, a = forecast.sum((1, 2)), actual.sum((1, 2))
    return np.abs(f - a).sum() / a.sum()


if __name__ == "__main__":
    o = load_orders()
    act, fc = actual_demand(o), forecast_demand(o)
    act_day, fc_day = (x.sum((1, 2)).reshape(3, 24).sum(1) for x in (act, fc))
    peak, weekend = hindsight_surges(o)
    print(f"orders kept: {len(o)}  scale x{scale_factor(o):.1f}")
    print("actual per day:", act_day.round(), " forecast per day (surge 1):", fc_day.round())
    print(f"hindsight surges: peak {peak:.2f}, weekend {weekend:.2f}")
    print(f"hourly WAPE at surge 1: {wape(fc, act):.0%}, "
          f"at hindsight surges: {wape(forecast_demand(o, peak, weekend), act):.0%}")
