from load_data import load_data
from features import add_features

df = add_features(load_data())
print(df.shape)
print(df[["ln_mv", "age_sq", "ln_fees_cum", "Winger", "player_id"]].head(3))
import sys
sys.path.append("src")
from load_data import load_data
from features import add_features
df = add_features(load_data())
df[["s2122", "s2223", "s2324", "s2425"]].sum()