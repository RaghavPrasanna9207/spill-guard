import numpy as np
import pytest

from spillguard import data, model, sim

VALUES = [100.0]  # R$15 per order per day late, as in test_model


def run(demand, deps, policy, temps=None, **kw):
    temps = np.zeros(len(demand)) if temps is None else temps
    return sim.run(demand, demand, VALUES, 1, temps, policy, noise=False, deps=deps, **kw)


def test_spill_accounting_matches_lp():
    # 100 orders in hour 0, one permanent picker (50/h) plus one temp from hour 1 (40/h):
    # wave 1 picks 90 for the 03:00 truck, the last 10 wait 24h for the next one.
    demand = np.zeros((4, 1, 1))
    demand[0, 0, 0] = 100
    deps = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
    temps = np.array([0, 1, 0, 0])
    lp = model.solve(demand, VALUES, 1, deps=deps, fixed_temps=temps, overtime_daily=0)
    for policy in sim.POLICIES:
        plan = run(demand, deps, policy, temps, overtime_daily=0)
        assert plan.spilled.sum() == pytest.approx(10)
        assert plan.spill_cost == pytest.approx(lp.spill_cost) == pytest.approx(150)


def test_packing_caps_simulated_picks():
    demand = np.zeros((4, 1, 1))
    demand[0, 0, 0] = 100
    deps = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
    temps = np.array([0, 1, 0, 0])  # picking 90 in wave 1, packing only 60
    lp = model.solve(demand, VALUES, 1, deps=deps, fixed_temps=temps, pack_cap=60,
                     overtime_daily=0)
    for policy in sim.POLICIES:
        plan = run(demand, deps, policy, temps, pack_cap=60, overtime_daily=0)
        assert plan.spilled.sum() == pytest.approx(40)
        assert plan.spill_cost == pytest.approx(lp.spill_cost)


def test_sim_uses_planned_overtime():
    # The plan covers the 50 excess orders with one overtime hour; every policy gets it.
    demand = np.zeros((4, 1, 1))
    demand[0, 0, 0] = 100
    deps = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
    plan = model.solve(demand, VALUES, 1, deps=deps, max_temps=0)
    assert plan.overtime.sum() == pytest.approx(1)
    for policy in sim.POLICIES:
        out = run(demand, deps, policy, plan.temps, overtime=plan.overtime)
        assert out.spilled.sum() == pytest.approx(0)
        assert out.labour_cost == pytest.approx(plan.labour_cost)


def test_replan_keeps_a_picker_late_when_demand_beats_the_forecast():
    # The day-before plan expected 50 orders (no temps, no overtime); 100 arrive. Seeing the
    # backlog, the re-plan keeps the permanent picker one extra hour for the other 50.
    actual = np.zeros((4, 1, 1))
    actual[0, 0, 0] = 100
    forecast = actual / 2
    deps = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
    args = (forecast, actual, VALUES, 1, np.zeros(4))
    late = sim.run(*args, "replan", noise=False, deps=deps)
    assert late.spilled.sum() == pytest.approx(0)
    assert late.overtime.sum() == pytest.approx(1)
    assert late.labour_cost == pytest.approx(model.overtime_cost(1))
    fixed = sim.run(*args, "replan", noise=False, deps=deps, same_day_overtime=False)
    assert fixed.spilled.sum() == pytest.approx(50)
    assert sim.run(*args, "earliest_truck", noise=False, deps=deps).spilled.sum() == pytest.approx(50)


def test_persistent_replan_matches_rebuilding_the_lp(monkeypatch):
    # Dense forecast, so every arrival hour has a row. Demand runs 30% under it, so the real
    # backlog differs from what the LP was built with: a re-plan that never updated its
    # arrival rows costs R$391.55 here, not R$376.80. (Over the forecast, capacity saturates
    # and every re-plan ties.)
    forecast = np.random.default_rng(0).uniform(100, 200, (6, 1, 1))
    deps = [np.array([3.0, 27.0, 51.0, 75.0, 99.0, 123.0])]
    temps = np.array([0, 1, 0, 0, 0, 1])  # one night start, within the 1.5 night cap
    args = (forecast, 0.7 * forecast, VALUES, 1, temps, "replan")
    persistent = sim.run(*args, noise=False, deps=deps, replan_every=1)
    init = model.Replanner.__init__

    def rebuild_every_call(self, *a, **k):
        init(self, *a, **k)
        self.lp.arrive = {}  # no arrival rows: every call takes the rebuild path

    monkeypatch.setattr(model.Replanner, "__init__", rebuild_every_call)
    rebuilt = sim.run(*args, noise=False, deps=deps, replan_every=1)
    assert persistent.spill_cost + persistent.labour_cost == pytest.approx(
        rebuilt.spill_cost + rebuilt.labour_cost)


def test_earliest_truck_beats_fifo_when_trucks_differ():
    # 50 orders each for state A (truck 03:00) and B (truck 10:00) arrive together; one picker.
    # FIFO splits wave 1 evenly, so 25 A orders miss 03:00. Picking A first spills nothing.
    demand = np.zeros((4, 2, 1))
    demand[0] = 50
    deps = [np.array([3.0, 27, 51, 75, 99, 123]), np.array([10.0, 34, 58, 82, 106, 130])]
    assert run(demand, deps, "fifo").spilled.sum() == pytest.approx(25)
    assert run(demand, deps, "earliest_truck").spilled.sum() == pytest.approx(0)
    assert run(demand, deps, "replan").spilled.sum() == pytest.approx(0)


@pytest.mark.skipif(not (data.RAW / "olist_orders_dataset.csv").exists(),
                    reason="Olist CSVs not in data/raw")
def test_replan_reproduces_lp_on_black_friday():
    orders = data.load_orders()
    base = model.base_headcount(data.forecast_demand(orders))
    actual, vals = data.actual_demand(orders), data.tier_values(orders)
    plan = model.solve(actual, vals, base)
    # Noise off and a perfect forecast: re-planning must land on the day-before plan's cost.
    replan = sim.run(actual, actual, vals, base, plan.temps, "replan", noise=False,
                     overtime=plan.overtime)
    assert replan.spill_cost == pytest.approx(plan.spill_cost, rel=0.01)
    # With noise, every policy sees the same draws for a seed.
    a, b = (sim.run(actual, actual, vals, base, plan.temps, p, seed=3)
            for p in ("fifo", "earliest_truck"))
    assert a.demand_total == b.demand_total
