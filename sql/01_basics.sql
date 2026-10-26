SELECT season_clean, COUNT(*) AS num_player_seasons
FROM player_seasons
GROUP BY season_clean
ORDER BY season_clean;