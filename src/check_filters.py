from load_data import load_data
from features import add_features
from filters import apply_filters

df = add_features(load_data())
print(df.shape)
print(df[["tm_mv", "age", "minutes"]].isna().sum())
print(apply_filters(df).shape)