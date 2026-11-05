import numpy as np
import pytest

from spillguard import config as cfg
from spillguard import model

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
