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
        penalty_scale=1.0, floor_cap=None, crowding=None, pack_cap=None, deps=None):
    """Simulate one realisation; returns a model.Plan of what actually happened.

    forecast: [hour, state, tier] demand the re-plan expects for hours not yet seen.
    actual: [hour, state, tier] mean realised demand (Poisson-sampled when noise is on).
    temps: temp shifts by start hour, fixed for every policy.
    replan_every: hours between re-plan solves. ponytail: each solve rebuilds the LP (~0.15 s,
        70% of it building), so hourly runs take ~12 s on 72h; every 2 h costs ~1% more
        spill. Upgrade: build the LP once and update the backlog right-hand sides.
    deps: departure-time arrays per state (defaults to the configured truck schedule).
    """
    H, S, V = actual.shape
    W = H + model.LOOKAHEAD
    temps = np.asarray(temps, float)
    crowding = cfg.CROWDING_MULT * base if crowding is None else crowding
    if pack_cap is None:
        pack_cap = cfg.PACK_STATIONS_MULT * base * cfg.PACK_RATE
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
    calm = np.minimum(temp, crowding - base)
    picking = speed * (cfg.PERM_RATE * perm + cfg.TEMP_RATE * calm
                       + cfg.CROWDED_RATE * (temp - calm))
    capacity = np.minimum(picking, pack_cap)

    hours = np.arange(H)
    rem = np.zeros((H, S, V))  # orders waiting, by arrival hour
    spilled = np.zeros((S, V))
    cost = 0.0
    for w in range(1, W):
        if w <= H:
            rem[w - 1] += arrivals[w - 1]
        elif rem.sum() < 1e-9:
            break  # all orders picked and no more arriving
        cap = capacity[w]
        earliest_truck = dep_at[w][None, :] * 1000 + hours[:, None]  # then oldest first
        if policy == "fifo":
            picked = _take(rem, cap, np.broadcast_to(hours[:, None], (H, S)).astype(float))
        elif policy == "earliest_truck":
            picked = _take(rem, cap, earliest_truck)
        elif policy == "replan":
            if (w - 1) % replan_every == 0:
                planned = _replan(rem, forecast, w, tier_values, base, temps, deps,
                                  penalty_scale, floor_cap, crowding, pack_cap)
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
        labour_cost=sum(model.shift_cost(t) * n for t, n in enumerate(temps)),
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


def _replan(rem, forecast, w, tier_values, base, temps, deps, penalty_scale, floor_cap, crowding,
            pack_cap):
    """LP picks per wave from wave w on, with the real backlog and the forecast for later hours.

    The backlog enters as orders that arrived in hour w-1. That is exact for decisions: any
    waiting order whose target truck is still reachable shares that hour's target, and delay
    already locked in on the rest is a constant.
    """
    H = len(forecast)
    n = max(H, w)
    demand = np.zeros((n,) + forecast.shape[1:])
    demand[w:H] = forecast[w:H]
    demand[w - 1] = rem.sum(0)
    t = np.zeros(n)
    t[:H] = temps
    return model.wave_picks(demand, tier_values, base, t, deps=deps, penalty_scale=penalty_scale,
                            floor_cap=floor_cap, crowding=crowding, pack_cap=pack_cap)


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
