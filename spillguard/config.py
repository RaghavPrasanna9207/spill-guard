"""Every assumption in SpillGuard. Sources are cited inline; ASSUMPTION marks a value with no source."""

# --- Scope -------------------------------------------------------------------
FC_STATE = "SP"                  # one FC, shipping only Sao Paulo-state sellers' orders
WINDOW_START = "2017-11-24"      # Black Friday 2017 (Fri), start of the 72h window
HOURS = 72
PEAK_DAY_ORDERS = 20_000         # realised Black Friday day is scaled to this volume
PROFILE_WEEKS = 4                # weeks before Black Friday used for the planning profile
# Top 8 destination states of SP-seller orders (86% of volume); others are dropped.
DEST_STATES = ["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "ES"]
TIER_NAMES = ["low", "mid", "high"]  # order-value terciles, cut points computed in data.py
EXCLUDED_STATUSES = ["canceled", "unavailable"]  # never reached the picking floor
# 2017 order values -> 2026 R$, to match 2026 wages. Compounded monthly IPCA, Dec 2017 to
# Sep 2026, Banco Central SGS series 433 (api.bcb.gov.br/dados/serie/bcdata.sgs.433).
IPCA_2017_TO_2026 = 1.5722

# --- Trucks ------------------------------------------------------------------
# Capital coordinates (decimal degrees) from IBGE Anuario Estatistico 2024, tabela 1.1.1.2.
CAPITALS = {
    "SP": (-23.5477, -46.6376),  # Sao Paulo (FC location)
    "RJ": (-22.9111, -43.2057),  # Rio de Janeiro
    "MG": (-19.9237, -43.9363),  # Belo Horizonte
    "RS": (-30.0282, -51.2286),  # Porto Alegre
    "PR": (-25.4133, -49.2679),  # Curitiba
    "BA": (-12.9742, -38.5124),  # Salvador
    "SC": (-27.5975, -48.5497),  # Florianopolis
    "ES": (-20.3212, -40.3393),  # Vitoria
}
# Road km / straight-line km: median over the same 8 capitals' routes to Brasilia in that
# IBGE table (road km from DNIT). Ratios range 1.15 (BH) to 1.45 (Salvador).
CIRCUITY = 1.26
INTRASTATE_KM = 150              # ASSUMPTION: average haul for SP-state deliveries
TRUCK_KMH = 60                   # ASSUMPTION: average line-haul speed including stops
HUB_ARRIVAL_HOUR = 6             # ASSUMPTION: trucks reach destination hubs by 06:00
TWO_TRUCKS_UNDER_KM = 500        # states closer than this get 2 departures/day
LOADING_BUFFER_H = 1             # a wave must finish 1h before its truck departs

# --- Labour ------------------------------------------------------------------
# Single-item manual picking: 40-60 orders/h; new workers 35-50 lines/h
# (Pallite, "KPIs for eaches picking"). Olist averages ~1.14 items/order.
PERM_RATE = 50                   # orders/h, experienced permanent picker
TEMP_RATE = 40                   # orders/h, new temp picker
SHIFT_HOURS = 8
# Floor limits. Aisle space, carts and scanners cap how many pickers can work at once, and
# crowded aisles make each extra picker add less (picker blocking).
# Both scale with the permanent headcount, so they move with FC volume. At 20k orders on the
# peak day (4 permanent pickers) this is a 16-picker floor that congests above 12.
FLOOR_CAP_MULT = 4               # ASSUMPTION: floor holds 4x the permanent headcount
CROWDING_MULT = 3                # ASSUMPTION: aisles congest above 3x (75% of the floor)
CROWDED_RATE = 25                # ASSUMPTION: orders/h added by each picker above the threshold
# Agency supply. Temps come from an agency that can only fill so many shifts a day, and
# fewer at night. Scaled with permanent headcount like the floor cap: 30/day, 6 at night.
AGENCY_DAILY_MULT = 7.5          # ASSUMPTION: temp shift starts per day the agency can fill
AGENCY_NIGHT_MULT = 1.5          # ASSUMPTION: of those, starts in night hours (22:00-04:00)
# Packing. Manual single-item packing runs 40-80 orders per packer-hour (Racklify pack-rate
# benchmarks, vendor figures); Olist orders are nearly all single-item.
PACK_RATE = 60                   # orders/h per staffed pack station
PACK_STATIONS_MULT = 2.5         # ASSUMPTION: pack stations per permanent picker (10 at 4 pickers)
# ponytail: capacity counted in orders/h, not items; ~1.14 items/order makes the gap small.
TEMP_MONTHLY_WAGE = 2110         # R$, SP logistics temp job ads 2026 (R$2,000-2,120)
WORKDAYS_PER_MONTH = 22
EMPLOYER_COST_MULT = 1.8         # ASSUMPTION: payroll charges + agency fee on top of wage
NIGHT_PREMIUM = 0.20             # CLT art. 73: +20% for hours worked 22:00-05:00
# ponytail: ignores the shorter 52.5-min legal night hour.

# Simulation noise (sim.py)
PICK_SPEED_SD = 0.10             # ASSUMPTION: hour-to-hour spread of the floor's pick rate
ABSENT_PROB = 0.05               # ASSUMPTION: chance a rostered picker is missing in a given hour

# --- Spill penalty -----------------------------------------------------------
SUPPORT_COST_PER_DAY_LATE = 10.0  # ASSUMPTION: R$ per order per day late (contacts, vouchers)
CANCEL_PROB_PER_DAY_LATE = 0.05   # ASSUMPTION: chance a late order is cancelled, per day late

# --- LLM ---------------------------------------------------------------------
LLM_DAILY_CALL_CAP = 200
