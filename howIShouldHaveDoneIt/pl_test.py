import pandas as pd
import polars as pl

df = pl.scan_csv("ClassicHit.csv")
print(df)

artist_info = pl.sql("""
    SELECT 
            Artist, 
            MIN(Year) AS artist_start_year, 
            MAX(Year) AS artist_end_year, 
            MAX(Year)-MIN(Year) AS artist_career_length 

    FROM df GROUP BY Artist""")
pl.sql(

    """
    SELECT 
        *, 
        (Year-artist_start_year) AS years_since_debut 
    FROM df 
        LEFT JOIN artist_info 
            ON df.Artist = artist_info.Artist""").collect().write_csv("ClassicHitPl.csv")

