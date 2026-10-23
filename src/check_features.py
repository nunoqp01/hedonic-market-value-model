from load_data import load_data
from features import add_features

df = add_features(load_data())
print(df.shape)
print(df[["ln_mv", "age_sq", "ln_fees_cum", "Winger", "player_id"]].head(3))
# The purpose of this file is to run tests on the code (kind of like running a do-file in stata to make sure everything works).