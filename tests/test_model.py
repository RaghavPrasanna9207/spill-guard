import time

import numpy as np
import pytest

from spillguard import config as cfg
from spillguard import data, model

# Hand-sized case: 1 state, 1 tier, 100 orders arriving in hour 0, one permanent picker
# (50/wave). Trucks at hours 3 and 27. Wave 1 is the earliest wave and catches the 03:00
# truck (1 + 1h wave + 1h loading = 3); wave 2 onward catches 27:00, i.e. 24h late.
DEMAND = np.zeros((4, 1, 1))
DEMAND[0, 0, 0] = 100
DEPS = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
VALUES = [100.0]  # R$ per order -> 10 + 100 x 0.05 = R$15 per day late


def solve(**kw):
    return model.solve(DEMAND, VALUES, base=1, deps=DEPS, **kw)


def test_no_temps_spills_the_excess():
    plan = solve(max_temps=0)
    assert plan.spilled.sum() == pytest.approx(50)
    assert plan.spill_cost == pytest.approx(50 * 24 * 15 / 24)


def test_stops_hiring_when_next_temp_costs_more_than_it_saves():
    # Temp 1 (+40/wave, ~R$190) saves 40 x R$15 = R$600. Temp 2 would save only the last
    # 10 orders (R$150) for ~R$190, so the optimum is one temp and 10 spilled.
    plan = solve()
    assert plan.temps.sum() == 1
    assert plan.spilled.sum() == pytest.approx(10)


def test_high_penalty_clears_all_spill():
    # At 2x penalty the last 10 orders cost R$300, more than a second temp.
    plan = solve(penalty_scale=2)
    assert plan.temps.sum() == 2
    assert plan.spilled.sum() == pytest.approx(0)


def test_zero_penalty_hires_nobody():
    assert solve(penalty_scale=0).temps.sum() == 0


def test_floor_cap_limits_hiring():
    # The 2x penalty wants 2 temps, but a 2-picker floor leaves room for 1 temp, and with
    # crowding at the permanent headcount that temp picks CROWDED_RATE: 50 - 25 spill.
    plan = solve(penalty_scale=2, floor_cap=2, crowding=1)
    assert plan.temps.sum() == 1
    assert plan.spilled.sum() == pytest.approx(50 - cfg.CROWDED_RATE)


def test_crowded_temps_pick_less():
    # With crowding at the permanent headcount, the one allowed temp adds CROWDED_RATE
    # (25/h) instead of TEMP_RATE (40/h): 50 - 25 = 25 spill instead of 10.
    plan = solve(max_temps=1, crowding=1)
    assert plan.spilled.sum() == pytest.approx(50 - cfg.CROWDED_RATE)


def test_lp_with_milp_headcount_reproduces_milp():
    milp = solve(max_temps=1)
    lp = solve(fixed_temps=milp.temps)
    assert lp.spill_cost == pytest.approx(milp.spill_cost)


def test_frontier_trades_labour_for_spill_linearly_on_hand_case():
    # Each fractional temp in wave 1 saves 40 orders x R$15, more than its ~R$190, so the LP
    # optimum hires 1.25 temps and clears all 50 excess orders; spill falls linearly to it.
    plans = model.frontier(DEMAND, VALUES, base=1, deps=DEPS, points=6)
    caps = [p.temps.sum() for p in plans]
    assert caps == pytest.approx([0, 0.25, 0.5, 0.75, 1, 1.25])
    for cap, p in zip(caps, plans):
        assert p.spill_cost == pytest.approx(15 * (50 - 40 * cap))
        assert p.labour_cost == pytest.approx(model.shift_cost(1) * cap)  # 01:00 start is cheapest


def test_packing_caps_picks_and_hiring():
    # Packing 60/wave: one temp would raise picking from 50 to 90 but only 10 more orders get
    # packed (R$150 saved for ~R$190), so the optimum hires nobody and 50 spill.
    plan = solve(pack_cap=60)
    assert plan.temps.sum() == 0
    assert plan.spilled.sum() == pytest.approx(50)
    assert solve(pack_cap=60, fixed_temps=[0, 1, 0, 0]).spilled.sum() == pytest.approx(40)


def test_extra_temps_save_nothing_when_packing_binds():
    # One temp already lifts picking to 90 = packing capacity: a second one adds nothing.
    roi = model.marginal_roi(DEMAND, VALUES, 1, [0, 1, 0, 0], deps=DEPS, pack_cap=90)
    assert list(roi.spill_avoided) == pytest.approx([0, 0, 0, 0])


needs_olist = pytest.mark.skipif(not (data.RAW / "olist_orders_dataset.csv").exists(),
                                 reason="Olist CSVs not in data/raw")


def black_friday():
    orders = data.load_orders()
    base = model.base_headcount(data.forecast_demand(orders))
    demand = data.forecast_demand(orders, peak_surge=7.5, weekend_surge=3.3)  # ~hindsight surges
    return demand, data.tier_values(orders), base


def test_marginal_roi_on_hand_case():
    # One temp covering wave 1 (start 0 or 1) gets 40 more orders onto the 03:00 truck:
    # 40 x R$15 = R$600. Later starts miss that truck, and everything waiting already
    # catches the next one, so they save nothing.
    roi = model.marginal_roi(DEMAND, VALUES, 1, np.zeros(4), deps=DEPS)
    assert list(roi.spill_avoided) == pytest.approx([600, 600, 0, 0])
    assert list(roi.orders_saved) == pytest.approx([40, 40, 0, 0])
    assert roi.net[1] == pytest.approx(600 - model.shift_cost(1))


def test_marginal_roi_skips_hours_with_a_full_floor():
    # A 2-picker floor with one temp already on shift: every start overlaps a full wave.
    roi = model.marginal_roi(DEMAND, VALUES, 1, [0, 1, 0, 0], deps=DEPS, floor_cap=2, crowding=1)
    assert roi.net.isna().all()


@needs_olist
def test_marginal_roi_matches_frontier_slope():
    # Best extra temp's net saving ~ the drop in the LP bound from one more allowed temp.
    demand, vals, base = black_friday()
    plan = model.solve(demand, vals, base, max_temps=50)
    roi = model.marginal_roi(demand, vals, base, plan.temps)
    slope = plan.lower_bound - model.solve(demand, vals, base, max_temps=51).lower_bound
    assert roi.net.max() == pytest.approx(slope, rel=0.05)
    assert roi.net.isna().any()  # Black Friday's peak fills the floor


@needs_olist
def test_frontier_on_black_friday():
    demand, vals, base = black_friday()
    start = time.perf_counter()
    plan = model.solve(demand, vals, base)
    assert time.perf_counter() - start < 10
    plans = model.frontier(demand, vals, base)
    spill = [p.spill_cost for p in plans]
    labour = [p.labour_cost for p in plans]
    assert all(b <= a + 1e-6 for a, b in zip(spill, spill[1:]))
    assert all(b >= a - 1e-6 for a, b in zip(labour, labour[1:]))
    assert spill[0] == pytest.approx(model.solve(demand, vals, base, max_temps=0).spill_cost)
    assert plans[-1].total_cost == pytest.approx(plan.lower_bound)


def test_truck_schedule():
    sp = model.departures("SP", 24)
    assert len(sp[sp < 24]) == 2  # 150 km, under the 2-truck threshold
    ba = model.departures("BA", 24)
    assert len(ba[ba < 24]) == 1  # Salvador is ~2,000 road km away
    assert model.road_km("BA") > 1500


def test_night_shifts_cost_more():
    day = model.shift_cost(8)    # 08:00-16:00, no night hours
    night = model.shift_cost(22)  # 22:00-06:00, 7 night hours
    assert night == pytest.approx(day * (1 + cfg.NIGHT_PREMIUM * 7 / 8))
