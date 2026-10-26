"""Build a SQLite database from the project's player-season table."""
import sqlite3
from pathlib import Path

from load_data import load_data      # copia estes imports do oot_flags.py
from features import add_features

DB_PATH = Path("data/processed/football.db")


def main():
    df = add_features(load_data())
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    # to_sql writes the DataFrame as a table; replace = rebuild each run
    df.to_sql("player_seasons", con, if_exists="replace", index=False)
    n = con.execute("SELECT COUNT(*) FROM player_seasons").fetchone()[0]
    con.close()
    print(f"Table player_seasons written to {DB_PATH}: {n} rows")


if __name__ == "__main__":
    main()

