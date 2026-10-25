"""Model specifications shared by all scripts (single source of truth).

Keeping the formulas here avoids copy-pasting them into every script,
which is how silent differences between scripts creep in.
"""

FORMULA_BASE = (
    "ln_mv ~ goals_p90 + assists_p90 + minutes + age + age_sq + Winger"
    " + int_A + int_youth + ln_fees_cum + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_argentina + d_brasileirao"
)

# Advanced metrics: rows with a missing value here are dropped from the Extended sample
ADVANCED = ["npxg_p90", "xa_p90", "shots_p90", "dribbles_p90",
            "keypasses_p90", "aerial_p90", "finishing"]

FORMULA_EXT = (
    "ln_mv ~ assists_p90 + minutes + age + age_sq + Winger + int_A + int_youth"
    " + ln_fees_cum + npxg_p90 + xa_p90 + shots_p90 + dribbles_p90"
    " + keypasses_p90 + aerial_p90 + finishing + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_brasileirao"
)

LEAGUES = ["d_premier", "d_laliga", "d_seriea", "d_bundesliga", "d_ligue1",
           "d_eredivisie", "d_championship", "d_segunda", "d_argentina", "d_brasileirao"]