import pandas as pd
import duckdb


df = pd.read_csv("ClassicHit.csv")

artist_info = duckdb.sql("""
    SELECT artist, MIN(year) AS artist_start_year, MAX(year) AS artist_end_year, MAX(year)-MIN(year) AS artist_career_length FROM df GROUP BY artist
    """).df()

full_df = duckdb.sql("""
    SELECT *, (year-artist_start_year) AS years_since_debut FROM df LEFT JOIN artist_info ON df.artist = artist_info.artist""").df()

full_df.to_csv("ClassicHit2.csv", index=False)

