"""Day-before staffing MILP; with temps fixed it becomes the LP used for re-plans and what-ifs.

Time is in hours from the window start. Wave w runs [w, w+1); an order arriving during hour a
can be picked from wave a+1 on, and catches the first truck leaving at or after
w + 1 + LOADING_BUFFER_H. Delay is priced per hour behind the truck its earliest wave
would have caught.
"""
import math
from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp

from spillguard import config as cfg

# ponytail: orders not picked within LOOKAHEAD waves go to a backlog at a flat delay;
# raise it if backlog shows up in results.
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


def caught(deps, wave):
    """Departure time caught by orders picked in `wave`."""
    return deps[np.searchsorted(deps, wave + 1 + cfg.LOADING_BUFFER_H)]


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
    spilled: np.ndarray    # [hour, state, tier] orders that missed their earliest truck
    backlog: float         # orders not picked within LOOKAHEAD waves
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


def solve(demand, tier_values, base, deps=None, max_temps=None, fixed_temps=None,
          penalty_scale=1.0):
    """Choose temps by start hour and wave assignment; with fixed_temps, only wave assignment.

    demand: [hour, state, tier] orders arriving each hour.
    deps: departure-time arrays per state (defaults to the configured truck schedule).
    max_temps: cap on total temp shifts, used for the frontier sweep.

    The staffing MILP is solved by LP relaxation + rounding up, then an LP re-solve with the
    rounded temps fixed. Exact branch-and-bound took 25-60 s for one day (CBC, SCIP, HiGHS);
    this takes well under a second and reports its distance from the LP bound in `gap`.
    """
    args = (demand, tier_values, base, deps, penalty_scale)
    if fixed_temps is not None:
        return _lp(*args, temps=np.asarray(fixed_temps, float))
    relaxed = _lp(*args, max_temps=max_temps)
    frac = relaxed.temps
    # Two roundings, keep the cheaper plan. Rounding each hour up never loses capacity but
    # overshoots by up to one person per start hour. Rounding the running total keeps the
    # total, and every 8h window, within one person of the LP.
    up = np.ceil(frac - 1e-6)
    if max_temps is not None:  # trim the most-rounded hours first
        for t in np.argsort(frac - up):
            if up.sum() <= max_temps:
                break
            up[t] = max(up[t] - 1, 0)
    running = np.diff(np.ceil(np.cumsum(frac) - 1e-6), prepend=0)
    # ponytail: no local search after rounding; add one if `gap` gets large.
    plan = min((_lp(*args, temps=t) for t in (up, running)), key=lambda p: p.total_cost)
    plan.lower_bound = relaxed.total_cost
    return plan


def _lp(demand, tier_values, base, deps, penalty_scale, temps=None, max_temps=None):
    """One LP: wave assignment with temps fixed, or with temps as continuous variables."""
    H, S, V = demand.shape
    W = H + LOOKAHEAD
    if deps is None:
        deps = [departures(s, W + BACKLOG_DELAY_H + 48) for s in cfg.DEST_STATES]
    rate = delay_cost_per_hour(tier_values, penalty_scale)

    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()
    fixed = temps is not None
    h = list(temps) if fixed else [solver.NumVar(0, inf, f"h{t}") for t in range(H)]
    objective = solver.Objective()

    # Capacity per wave: permanent + temps whose 8h shift covers it.
    cap = []
    for w in range(W):
        on_shift = range(max(0, w - cfg.SHIFT_HOURS + 1), min(w, H - 1) + 1)
        ub = cfg.PERM_RATE * base + (cfg.TEMP_RATE * sum(h[t] for t in on_shift) if fixed else 0)
        c = solver.Constraint(-inf, ub)
        if not fixed:
            for t in on_shift:
                c.SetCoefficient(h[t], -cfg.TEMP_RATE)
        cap.append(c)

    flows = []  # (a, s, v, delay_h, var)
    backlog = {}
    for a, s, v in zip(*np.nonzero(demand)):
        ideal = caught(deps[s], a + 1)
        row = solver.Constraint(demand[a, s, v], demand[a, s, v])
        for w in range(a + 1, min(a + 1 + LOOKAHEAD, W)):
            x = solver.NumVar(0, inf, "")
            delay = caught(deps[s], w) - ideal
            row.SetCoefficient(x, 1)
            cap[w].SetCoefficient(x, 1)
            objective.SetCoefficient(x, delay * rate[v])
            flows.append((a, s, v, delay, x))
        b = solver.NumVar(0, inf, "")
        row.SetCoefficient(b, 1)
        objective.SetCoefficient(b, BACKLOG_DELAY_H * rate[v])
        backlog[a, s, v] = b

    if not fixed:
        for t in range(H):
            objective.SetCoefficient(h[t], shift_cost(t))
        if max_temps is not None:
            limit = solver.Constraint(0, max_temps)
            for t in range(H):
                limit.SetCoefficient(h[t], 1)
    objective.SetMinimization()

    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("solver did not reach an optimal solution")

    temps = np.array([t if fixed else t.solution_value() for t in h])
    spilled = np.zeros_like(demand, dtype=float)
    spill_cost = 0.0
    for a, s, v, delay, x in flows:
        if delay > 0:
            spilled[a, s, v] += x.solution_value()
            spill_cost += x.solution_value() * delay * rate[v]
    for (a, s, v), b in backlog.items():
        spilled[a, s, v] += b.solution_value()
        spill_cost += b.solution_value() * BACKLOG_DELAY_H * rate[v]
    return Plan(
        temps=temps,
        labour_cost=sum(shift_cost(t) * n for t, n in enumerate(temps)),
        spill_cost=spill_cost,
        spilled=spilled,
        backlog=sum(b.solution_value() for b in backlog.values()),
        demand_total=float(demand.sum()),
    )
