import pandas as pd
import polars as pl
import sqldf


df = pd.read_csv("ClassicHit.csv")

artist_info = sqldf.run("""
    SELECT 
            artist, 
            MIN(year) AS artist_start_year, 
            MAX(year) AS artist_end_year, 
            MAX(year)-MIN(year) AS artist_career_length 

    FROM df GROUP BY artist""")

full_df = sqldf.run("""
    SELECT 
        *, 
        (year-artist_start_year) AS years_since_debut 
    FROM df 
        LEFT JOIN artist_info 
            ON df.artist = artist_info.artist""")

if full_df is not None:
    full_df.to_csv("ClassicHit2.csv", index=False)

