"""Day-before staffing MILP; with temps fixed it becomes the LP used for re-plans and what-ifs.

Time is in hours from the window start. Wave w runs [w, w+1); an order arriving during hour a
can be picked from wave a+1 on, and catches the first truck leaving at or after
w + 1 + LOADING_BUFFER_H. Its target truck is the one its earliest wave would catch.

Orders are pooled into one queue per (state, tier, target truck): once they share a target,
their arrival hour no longer matters. Pickers drain queues wave by wave; delay is priced per
hour between the truck an order actually catches and its target.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from spillguard import config as cfg

# ponytail: orders still queued LOOKAHEAD waves after their last arrival go to a backlog at
# a flat delay; raise it if backlog shows up in results.
LOOKAHEAD = 36
BACKLOG_DELAY_H = LOOKAHEAD + 24
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4}  # CLT art. 73 night period 22:00-05:00


def road_km(state):
    if state == cfg.FC_STATE:
        return cfg.INTRASTATE_KM
    (la1, lo1), (la2, lo2) = (np.radians(cfg.CAPITALS[s]) for s in (cfg.FC_STATE, state))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a)) * cfg.CIRCUITY


def departures(state, hours):
    """Daily departure times for a state, repeated to cover [0, hours]."""
    km = road_km(state)
    first = (cfg.HUB_ARRIVAL_HOUR - km / cfg.TRUCK_KMH) % 24
    daily = [first, (first + 12) % 24] if km < cfg.TWO_TRUCKS_UNDER_KM else [first]
    return np.sort([d + 24 * k for k in range(math.ceil(hours / 24) + 1) for d in daily])


def shift_cost(start):
    """R$ cost of one temp shift starting at window hour `start`, with the night premium."""
    night = sum((start + k) % 24 in NIGHT_HOURS for k in range(cfg.SHIFT_HOURS))
    day_rate = cfg.TEMP_MONTHLY_WAGE / cfg.WORKDAYS_PER_MONTH * cfg.EMPLOYER_COST_MULT
    return day_rate * (1 + cfg.NIGHT_PREMIUM * night / cfg.SHIFT_HOURS)


def delay_cost_per_hour(tier_values, penalty_scale=1.0):
    """R$ per order per hour late, by tier: support cost + value x cancellation chance."""
    per_day = cfg.SUPPORT_COST_PER_DAY_LATE + np.asarray(tier_values) * cfg.CANCEL_PROB_PER_DAY_LATE
    return penalty_scale * per_day / 24


def base_headcount(normal_demand):
    """Permanent pickers on the floor every hour: enough for a normal day's busiest hour."""
    return math.ceil(normal_demand.sum((1, 2)).max() / cfg.PERM_RATE)


@dataclass
class Plan:
    temps: np.ndarray      # temps starting at each hour
    labour_cost: float     # R$, temps only (permanent staff are a sunk cost)
    spill_cost: float      # R$
    spilled: np.ndarray    # [state, tier] orders that missed their target truck
    backlog: float         # orders still queued at the end of the lookahead
    demand_total: float
    lower_bound: float = math.nan  # LP-relaxation cost; no whole-number plan can beat it

    @property
    def gap(self):
        """How far above the proven lower bound this plan's cost could be (fraction)."""
        return self.total_cost / self.lower_bound - 1

    @property
    def total_cost(self):
        return self.labour_cost + self.spill_cost

    @property
    def spill_rate(self):
        return self.spilled.sum() / self.demand_total


def resolve_limits(base, floor_cap=None, crowding=None, pack_cap=None, agency_daily=None,
                   agency_night=None):
    """Floor and supply limits; unset ones default to config multiples of `base`.

    floor_cap: most pickers (permanent + temps) on the floor in any hour
        (default FLOOR_CAP_MULT x base).
    crowding: above this many on the floor, each extra temp adds CROWDED_RATE, not TEMP_RATE
        (default CROWDING_MULT x base).
    pack_cap: most orders packed per wave, so most picked (default PACK_STATIONS_MULT x base
        stations x PACK_RATE).
    agency_daily: most temp shifts the agency fills per window day (default
        AGENCY_DAILY_MULT x base).
    agency_night: most of those starting in night hours, 22:00-04:00 (default
        AGENCY_NIGHT_MULT x base).
    """
    floor_cap = cfg.FLOOR_CAP_MULT * base if floor_cap is None else floor_cap
    crowding = cfg.CROWDING_MULT * base if crowding is None else crowding
    if not base <= crowding <= floor_cap:
        raise ValueError("need base headcount <= crowding threshold <= floor cap")
    if pack_cap is None:
        pack_cap = cfg.PACK_STATIONS_MULT * base * cfg.PACK_RATE
    agency_daily = cfg.AGENCY_DAILY_MULT * base if agency_daily is None else agency_daily
    agency_night = cfg.AGENCY_NIGHT_MULT * base if agency_night is None else agency_night
    return SimpleNamespace(floor_cap=floor_cap, crowding=crowding, pack_cap=pack_cap,
                           agency_daily=agency_daily, agency_night=agency_night)


def solve(demand, tier_values, base, deps=None, max_temps=None, fixed_temps=None,
          penalty_scale=1.0, **limits):
    """Choose temps by start hour and wave assignment; with fixed_temps, only wave assignment.

    demand: [hour, state, tier] orders arriving each hour.
    deps: departure-time arrays per state (defaults to the configured truck schedule).
    max_temps: cap on total temp shifts.
    limits: floor_cap, crowding, pack_cap (see resolve_limits); the other functions here and
        sim.run take the same keywords.

    Exact branch-and-bound took 20-60+ s per day even on this pooled model (CBC, SCIP,
    HiGHS), so whole-number temps come from the LP relaxation rounded up, then a clean-up
    that removes one temp at a time while total cost falls. `gap` reports the distance from
    the LP lower bound. One LP is built; temps are pinned by bounds, so re-solves are warm.
    """
    lp = _WaveLP(demand, tier_values, base, deps, penalty_scale, max_temps, limits)
    if fixed_temps is not None:
        lp.cost(np.asarray(fixed_temps, float))
        return lp.plan()
    frac = lp.relax()
    bound = lp.solver.Objective().Value()
    temps = np.ceil(frac - 1e-6)
    if max_temps is not None:  # trim the most-rounded hours first
        for t in np.argsort(frac - temps):
            if temps.sum() <= max_temps:
                break
            temps[t] = max(temps[t] - 1, 0)
    # Rounding up can also ask the agency for more than it supplies; trim the most-rounded
    # start hour in each over-cap group. Terminates for the same reason as the floor below.
    for hours, cap in lp.supply:
        while temps[hours].sum() > cap + 1e-9:
            temps[hours[np.argmax((temps - frac)[hours])]] -= 1
    # Rounding up can overfill the floor; trim the most-rounded start hour covering each
    # overfull wave. Terminates: rounded-down temps never exceed the LP's, which fit.
    while (over := np.flatnonzero(on_shift(temps, lp.W) > lp.lim.floor_cap - base + 1e-9)).size:
        w = over[0]
        window = np.arange(max(0, w - cfg.SHIFT_HOURS + 1), min(w, len(temps) - 1) + 1)
        temps[window[np.argmax((temps - frac)[window])]] -= 1
    # Clean-up: drop a temp, or add one where the floor and agency allow, while total cost
    # falls. Adds win back capacity the trims above removed (agency caps took the gap from
    # 1.5% to 4.7% without them). Cost is convex in each hour's temps, so an hour's reduced
    # cost rc bounds the move: adding can only help if rc < 0, removing only if rc > 0. Only
    # those moves are tried, most promising first.
    best = lp.cost(temps)
    while True:  # ponytail: single add/remove moves; try swaps if `gap` gets large
        rc = np.array([v.reduced_cost() for v in lp.h])
        moves = [(t, -1) for t in np.flatnonzero((temps > 0) & (rc > 1e-6))]
        moves += [(t, 1) for t in np.flatnonzero(rc < -1e-6)]
        for t, step in sorted(moves, key=lambda m: m[1] * rc[m[0]]):
            if step == 1 and not lp.can_add(temps, t):
                continue
            temps[t] += step
            c = lp.cost(temps)
            if c < best - 1e-6:
                best = c
                break  # the LP now holds this plan, so its reduced costs are current
            temps[t] -= step
        else:
            break
    lp.cost(temps)
    plan = lp.plan()
    plan.lower_bound = bound
    return plan


def frontier(demand, tier_values, base, points=15, deps=None, penalty_scale=1.0, **limits):
    """Plans along the labour-cost vs spill-cost curve, as the cap on total temp shifts rises
    from 0 to the unconstrained optimum (past it, extra temps cost more than they save).

    Each point is the LP relaxation (fractional temps) of one model re-solved per cap, about
    0.1 s a point; solve() per cap took ~1.3 s a point on 72h. Total cost can only fall as the
    cap rises. Whole-number plans from solve() sit within their `gap` (~1-2%) of this curve.
    """
    lp = _WaveLP(demand, tier_values, base, deps, penalty_scale, None, limits)
    plans = []
    for cap in np.linspace(0, lp.relax().sum(), points):
        lp.limit.SetUb(cap)
        lp.relax()
        plan = lp.plan()
        plan.lower_bound = plan.total_cost
        plans.append(plan)
    return plans


def wave_picks(demand, tier_values, base, temps, deps=None, penalty_scale=1.0, **limits):
    """Orders [wave, state, tier] the LP picks in each wave with temps fixed (rolling re-plan)."""
    lp = _WaveLP(demand, tier_values, base, deps, penalty_scale, None, limits)
    lp.cost(temps)
    out = np.zeros((lp.W,) + demand.shape[1:])
    for s, v, w, _, x in lp.picks:
        out[w, s, v] += x.solution_value()
    return out


def marginal_roi(demand, tier_values, base, temps, deps=None, penalty_scale=1.0, **limits):
    """What one more temp starting at each hour would cost and save, on top of `temps`.

    Exact: re-solves the fixed-headcount LP per start hour (re-running solve() would add its
    ~1-2% heuristic noise, about one temp's worth). Rows are NaN where the extra temp could
    not work: the floor is already full during part of that shift, or the agency has no more
    temps for that day or night.
    """
    lp = _WaveLP(demand, tier_values, base, deps, penalty_scale, None, limits)
    temps = np.asarray(temps, float)
    lp.cost(temps)
    before = lp.plan()
    H = len(temps)
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(H, name="start"),
                       columns=["shift_cost", "spill_avoided", "orders_saved"])
    for t in range(H):
        if not lp.can_add(temps, t):
            continue
        temps[t] += 1
        lp.cost(temps)
        after = lp.plan()
        temps[t] -= 1
        out.loc[t] = [shift_cost(t), before.spill_cost - after.spill_cost,
                      before.spilled.sum() - after.spilled.sum()]
    out["net"] = out.spill_avoided - out.shift_cost
    return out


class _WaveLP:
    """Wave-assignment LP with temps as variables; pin them via bounds to cost a plan."""

    def __init__(self, demand, tier_values, base, deps, penalty_scale, max_temps, limits):
        lim = self.lim = resolve_limits(base, **limits)
        H, S, V = demand.shape
        W = self.W = H + LOOKAHEAD
        if deps is None:
            deps = [departures(s, W + BACKLOG_DELAY_H + 48) for s in cfg.DEST_STATES]
        rate = delay_cost_per_hour(tier_values, penalty_scale)
        self.demand, self.rate, self.base = demand, rate, base
        solver = self.solver = pywraplp.Solver.CreateSolver("GLOP")
        inf = solver.infinity()
        h = self.h = [solver.NumVar(0, inf, f"h{t}") for t in range(H)]
        objective = solver.Objective()
        self.picks, self.leftover = _add_flows(
            solver, objective, demand, deps, rate, base, h, W, lim)
        for t in range(H):
            objective.SetCoefficient(h[t], shift_cost(t))
        limit = self.limit = solver.Constraint(0, inf if max_temps is None else max_temps)
        for t in range(H):
            limit.SetCoefficient(h[t], 1)
        # Agency supply: (start hours, most temp shifts the agency can fill) per window day,
        # all starts and night starts.
        self.supply = []
        for day in range(0, H, 24):
            hours = np.arange(day, min(day + 24, H))
            night = hours[np.isin(hours % 24, list(NIGHT_HOURS))]
            self.supply += [(hours, lim.agency_daily), (night, lim.agency_night)]
        for hours, cap in self.supply:
            c = solver.Constraint(0, cap)
            for t in hours:
                c.SetCoefficient(h[t], 1)
        objective.SetMinimization()

    def can_add(self, temps, t):
        """Whether one more temp can start at hour t: within the temp cap, room on the floor for
        the whole shift, and agency supply left for that day and night."""
        if temps.sum() + 1 > self.limit.ub() + 1e-9:
            return False
        on = on_shift(np.pad(temps, (0, self.W - len(temps))), self.W)
        if (on[t:t + cfg.SHIFT_HOURS] >= self.lim.floor_cap - self.base - 1e-9).any():
            return False
        return not any(t in hours and temps[hours].sum() + 1 > cap + 1e-9
                       for hours, cap in self.supply)

    def _solve(self):
        if self.solver.Solve() != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("solver did not reach an optimal solution")
        return self.solver.Objective().Value()

    def relax(self):
        """Solve with fractional temps allowed; returns them."""
        for v in self.h:
            v.SetBounds(0, self.solver.infinity())
        self._solve()
        return np.array([v.solution_value() for v in self.h])

    def cost(self, temps):
        """Total cost (R$) with temps pinned to `temps`."""
        for v, n in zip(self.h, temps):
            v.SetBounds(n, n)
        return self._solve()

    def plan(self):
        S, V = self.demand.shape[1:]
        spilled = np.zeros((S, V))
        spill_cost = 0.0
        for s, v, _, delay, x in self.picks:
            if delay > 0:
                spilled[s, v] += x.solution_value()
                spill_cost += x.solution_value() * delay * self.rate[v]
        for s, v, q in self.leftover:
            spilled[s, v] += q.solution_value()
            spill_cost += q.solution_value() * BACKLOG_DELAY_H * self.rate[v]
        temps = np.array([v.solution_value() for v in self.h])  # whole when pinned by cost()
        return Plan(
            temps=temps,
            labour_cost=sum(shift_cost(t) * n for t, n in enumerate(temps)),
            spill_cost=spill_cost,
            spilled=spilled,
            backlog=sum(q.solution_value() for _, _, q in self.leftover),
            demand_total=float(self.demand.sum()),
        )


def on_shift(temps, W):
    """Temps on the floor in each of W waves (each shift covers SHIFT_HOURS waves)."""
    return np.convolve(temps, np.ones(cfg.SHIFT_HOURS))[:W]


def _add_flows(solver, objective, demand, deps, rate, base, h, W, lim):
    """Capacity rows per wave and one picking queue per (state, tier, target truck)."""
    H, S, V = demand.shape
    inf = solver.infinity()
    # Capacity per wave: permanent pickers plus temps on shift, split into "calm" temps (up to
    # the crowding threshold, TEMP_RATE each) and "crowded" temps (CROWDED_RATE each, up to
    # the floor cap). Calm slots give more capacity for the same cost, so the LP fills them
    # first; that makes capacity concave in headcount without integer variables.
    cap = []
    for w in range(W):
        calm = solver.NumVar(0, lim.crowding - base, "")
        crowded = solver.NumVar(0, lim.floor_cap - lim.crowding, "")
        split = solver.Constraint(0, 0)  # calm + crowded = temps on shift
        split.SetCoefficient(calm, 1)
        split.SetCoefficient(crowded, 1)
        for t in range(max(0, w - cfg.SHIFT_HOURS + 1), min(w, H - 1) + 1):
            split.SetCoefficient(h[t], -1)
        c = solver.Constraint(-inf, cfg.PERM_RATE * base)
        c.SetCoefficient(calm, -cfg.TEMP_RATE)
        c.SetCoefficient(crowded, -cfg.CROWDED_RATE)
        cap.append(c)
    # Packing: pickers cannot send more down to the pack stations than they can pack in a wave.
    # ponytail: pack stations are fixed staff, not planned or costed; add packer shifts if
    # packing binds often.
    pack = [solver.Constraint(0, lim.pack_cap) for _ in range(W)]

    picks = []    # (s, v, wave, delay_h, var)
    leftover = []  # (s, v, var) queue left at the end
    for s in range(S):
        # truck[w]: index of the departure caught by wave w
        truck = np.searchsorted(deps[s], np.arange(W) + 1 + cfg.LOADING_BUFFER_H)
        target = truck[1:H + 1]  # target truck of orders arriving in each hour
        for i in np.unique(target):
            hours = np.flatnonzero(target == i)  # contiguous arrival hours sharing this target
            for v in range(V):
                if not demand[hours, s, v].any():
                    continue
                queue = None  # orders waiting before this wave
                for w in range(hours[0] + 1, min(hours[-1] + 1 + LOOKAHEAD, W)):
                    arriving = demand[w - 1, s, v] if w - 1 in hours else 0.0
                    x = solver.NumVar(0, inf, "")
                    q = solver.NumVar(0, inf, "")
                    # queue after wave w = queue before + arrivals - picks
                    balance = solver.Constraint(arriving, arriving)
                    balance.SetCoefficient(q, 1)
                    balance.SetCoefficient(x, 1)
                    if queue is not None:
                        balance.SetCoefficient(queue, -1)
                    cap[w].SetCoefficient(x, 1)
                    pack[w].SetCoefficient(x, 1)
                    delay = deps[s][truck[w]] - deps[s][i]
                    objective.SetCoefficient(x, delay * rate[v])
                    picks.append((s, v, w, delay, x))
                    queue = q
                objective.SetCoefficient(queue, BACKLOG_DELAY_H * rate[v])
                leftover.append((s, v, queue))
    return picks, leftover
