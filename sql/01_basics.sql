SELECT season_clean, COUNT(*) AS num_player_seasons
FROM player_seasons
GROUP BY season_clean
ORDER BY season_clean;

--Exercise 1.2: Count the number of players per league
SELECT league, COUNT (*) AS num_player_seasons
FROM player_seasons
GROUP BY league
ORDER BY num_player_seasons DESC;

--Exercise 1.3: Count the 10 players with the biggest tm_mv
SELECT name, club, season_clean, tm_mv
FROM player_seasons
ORDER BY tm_mv DESC
LIMIT 10;

--Exercise 1.4: Count the 10 players with the biggest tm_mv without repetition
SELECT name, MAX(tm_mv  ) AS max_tm_mv
FROM player_seasons
GROUP BY name
ORDER BY max_tm_mv DESC
LIMIT 10;

-- Exercise 1.5: Calculate the average tm_mv per season_clean
SELECT season_clean, round(AVG(tm_mv) / 1000000, 2) AS avg_tm_mv
FROM player_seasons
GROUP BY season_clean
ORDER BY season_clean;

-- Exercise 2.1: Who are the 2024/25 Liga Portugal Forwards with over 1.800 minutes?
SELECT name, club, ROUND(age, 1) AS age, minutes, npxg_p90
FROM player_seasons
WHERE season_clean = '2024/2025' AND league = 'Liga Portugal' AND minutes > 1800
ORDER BY npxg_p90 DESC;

-- Exercise 2.2: How many missing values are there in the player_seasons table?
SELECT league, COUNT(*) AS total_rows, COUNT(npxg_p90) AS non_null_npxg_p90, COUNT(*) - COUNT(npxg_p90) AS missing_npxg_p90, ROUND(((COUNT(*) - COUNT(npxg_p90)) * 100.0) / COUNT(*), 1) AS missing_percentage
FROM player_seasons
GROUP BY league;

-- Exercise 3.1: Who are the leagues with more than 300 lines in the player_seasons table? (Show the league and number of lines in order)
SELECT league, COUNT(*) AS num_player_seasons
FROM player_seasons
GROUP BY league
HAVING COUNT(*) > 300
ORDER BY num_player_seasons DESC;

-- Exercise 3.2: In 2024/25, which are the clubes with at least 3 players with more than 1.500 minutes?
SELECT club, league, COUNT(*) AS num_players
FROM player_seasons
WHERE season_clean = '2024/2025' AND minutes > 1500
GROUP BY club
HAVING COUNT(*) >= 3
ORDER BY num_players DESC;

-- Exercise 3.3: Classify each league as "Big 5" or "Other", and show, for each league, the number of lines and the average tm_mv.
SELECT 
    CASE 
        WHEN league IN ('Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1') THEN 'Big 5'
        ELSE 'Other'
    END AS league_classification,
    COUNT(*) AS num_player_seasons,
    ROUND(AVG(tm_mv) / 1000000, 2) AS avg_tm_mv
    FROM player_seasons
    GROUP BY league_classification;