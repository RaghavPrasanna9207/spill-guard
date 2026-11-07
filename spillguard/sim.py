"""Wave-level simulation of the window at fixed headcount, comparing picking policies.

Each hourly wave: orders that arrived up to the previous hour wait as counts per
(arrival hour, state, tier); the floor's capacity is drawn with absent pickers and pick-speed
noise; the policy chooses what to pick; delay is costed as in model.py (hours between the
truck an order catches and its target truck). All noise is drawn up front from the seed, so
every policy sees the same demand and staffing for a given seed.
"""
import numpy as np

from spillguard import config as cfg
from spillguard import model

POLICIES = ("fifo", "earliest_truck", "replan")


def run(forecast, actual, tier_values, base, temps, policy, noise=True, seed=0, replan_every=2,
        penalty_scale=1.0, deps=None, overtime=None, same_day_overtime=True, **limits):
    """Simulate one realisation; returns a model.Plan of what actually happened.

    forecast: [hour, state, tier] demand the re-plan expects for hours not yet seen.
    actual: [hour, state, tier] mean realised demand (Poisson-sampled when noise is on).
    temps: temp shifts by start hour, fixed for every policy.
    replan_every: hours between re-plan solves. ponytail: solving now dominates (the LP is
        built once per run, 1.6x faster than rebuilding), so hourly runs take about twice as
        long as the 2-hour default, which costs ~1% more spill. Upgrade: cache results per
        scenario in the app, or OR-Tools MathOpt for incremental GLOP solves.
    deps: departure-time arrays per state (defaults to the configured truck schedule).
    overtime: planned permanent overtime picker-hours per wave (Plan.overtime). FIFO and
        earliest-truck keep it; not hit by absence, since those pickers are already on shift.
    same_day_overtime: the re-plan re-chooses overtime from the real backlog for every wave not
        yet worked, within the same caps (hours already worked count toward the day's cap).
        Labour cost counts the overtime actually worked.
    limits: floor_cap, crowding, pack_cap, as for model.solve.
    """
    H, S, V = actual.shape
    W = H + model.LOOKAHEAD
    temps = np.asarray(temps, float)
    ot = np.zeros(W)
    if overtime is not None:
        ot[:min(W, len(overtime))] = overtime[:W]
    lim = model.resolve_limits(base, **limits)
    rate = model.delay_cost_per_hour(tier_values, penalty_scale)
    if deps is None:
        deps = [model.departures(s, W + model.BACKLOG_DELAY_H + 48) for s in cfg.DEST_STATES]
    # dep_at[w, s]: departure time of the truck wave w catches for state s
    dep_at = np.stack(
        [d[np.searchsorted(d, np.arange(W + 1) + 1 + cfg.LOADING_BUFFER_H)] for d in deps], 1)

    rng = np.random.default_rng(seed)
    on = model.on_shift(np.pad(temps, (0, W - H)), W).round().astype(int)
    if noise:
        # ponytail: absence drawn per hour, so it never lasts a whole shift; draw per shift
        # if the spread of results looks too narrow.
        arrivals = rng.poisson(actual).astype(float)
        perm = rng.binomial(base, 1 - cfg.ABSENT_PROB, W)
        temp = rng.binomial(on, 1 - cfg.ABSENT_PROB)
        speed = np.maximum(rng.normal(1, cfg.PICK_SPEED_SD, W), 0)
    else:
        arrivals, perm, temp, speed = actual.astype(float), np.full(W, base), on, np.ones(W)

    if policy == "replan":
        replanner = model.Replanner(forecast, tier_values, base, temps, deps=deps,
                                    penalty_scale=penalty_scale, **limits)
    hours = np.arange(H)
    rem = np.zeros((H, S, V))  # orders waiting, by arrival hour
    spilled = np.zeros((S, V))
    cost = 0.0
    for w in range(1, W):
        if w <= H:
            rem[w - 1] += arrivals[w - 1]
        elif rem.sum() < 1e-9:
            break  # all orders picked and no more arriving
        earliest_truck = dep_at[w][None, :] * 1000 + hours[:, None]  # then oldest first
        if policy == "replan" and (w - 1) % replan_every == 0:
            free = ot.copy()
            if same_day_overtime:
                free[w:] = np.nan
            planned, chosen = replanner(rem.sum(0), w, free)
            ot[w:] = chosen[w:W]
        # Overtime pickers take floor slots like temps and add the experience gap, as in the LP.
        slots = temp[w] + ot[w]
        calm = min(slots, lim.crowding - base)
        picking = speed[w] * (cfg.PERM_RATE * perm[w] + cfg.TEMP_RATE * calm
                              + cfg.CROWDED_RATE * (slots - calm)
                              + (cfg.PERM_RATE - cfg.TEMP_RATE) * ot[w])
        cap = min(picking, lim.pack_cap)
        if policy == "fifo":
            picked = _take(rem, cap, np.broadcast_to(hours[:, None], (H, S)).astype(float))
        elif policy == "earliest_truck":
            picked = _take(rem, cap, earliest_truck)
        elif policy == "replan":
            picked = _oldest_first(rem, planned[w])
            if picked.sum() > cap:  # fewer pickers turned up than the LP assumed
                picked *= cap / picked.sum()
            picked += _take(rem - picked, cap - picked.sum(), earliest_truck)
        else:
            raise ValueError(f"unknown policy {policy!r}")
        rem -= picked
        delay = dep_at[w][None, :] - dep_at[hours + 1]  # [arrival hour, state]
        spilled += (picked * (delay > 0)[..., None]).sum(0)
        cost += (picked * delay[..., None] * rate).sum()
    spilled += rem.sum(0)
    cost += (rem.sum((0, 1)) * rate).sum() * model.BACKLOG_DELAY_H
    return model.Plan(
        temps=temps,
        overtime=ot,
        labour_cost=sum(model.shift_cost(t) * n for t, n in enumerate(temps))
        + sum(model.overtime_cost(w) * n for w, n in enumerate(ot)),
        spill_cost=cost,
        spilled=spilled,
        backlog=rem.sum(),
        demand_total=arrivals.sum(),
    )


def compare(forecast, actual, tier_values, base, temps, runs=10, **kw):
    """{policy: {kpi: (mean, 95% CI half-width)}} for spill rate and spill cost over `runs` seeds."""
    out = {}
    for p in POLICIES:
        plans = [run(forecast, actual, tier_values, base, temps, p, seed=i, **kw)
                 for i in range(runs)]
        out[p] = {k: _mean_ci([getattr(x, k) for x in plans]) for k in ("spill_rate", "spill_cost")}
    return out


def _mean_ci(xs):
    # ponytail: normal 1.96, about 15% narrower than Student-t at 10 runs, 4% at 30.
    xs = np.asarray(xs)
    return xs.mean(), 1.96 * xs.std(ddof=1) / np.sqrt(len(xs))


def _take(rem, cap, key):
    """Up to `cap` orders from rem [arrival, state, tier], lowest key first; equal keys share
    pro rata."""
    total = rem.sum(2)
    keys, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(key.shape)
    group = np.bincount(inv.ravel(), total.ravel(), len(keys))
    taken = np.clip(cap - (np.cumsum(group) - group), 0, group)
    frac = np.divide(taken, group, out=np.zeros_like(group), where=group > 0)
    return rem * frac[inv][..., None]


def _oldest_first(rem, amount):
    """Take amount [state, tier] from rem, oldest arrivals first."""
    return np.clip(amount - (rem.cumsum(0) - rem), 0, rem)
