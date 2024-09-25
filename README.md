# Summary
## The Data
I found [this](href=https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023?resource=download>this) dataset on Kaggle the other day, and as a musician, I couldn't resist taking a closer look. In this analysis I will be taking an informed look at the cultural nature of the different genres of music. dataset consisting of some parsed musical features of 15000 tracks of various genres on Spotify. I will be looking at some of this data, and inserting some more personal understanding of Music culture from the standpoint of a musician.
## The Cultural Roots of Music Genre
In a world constantly looking for the "next thing", the time any given piece of music will remain popular seems so very short. But might there be some particular bands or styles which garner the appreciation of listeners over longer periods of time? Well, lets take a look at the data. This dashboard was designed to answer the following question: What time periods are most listened to across different styles?
### The Results
![Dashboard1.png](Dashboard1.png)
### Interesting Notes
I would like to start with the left chart, because it is really a lot more insightful and interesting than the right. The left chart is a series of bar charts which plot the popularity of the average song from any given decade of music, in any given Genre. An important thing to keep in mind is that these are the popularity indicators related to _todays_ listeners. These are not the relative measures of popularity _in_ the decades, but rather the relative measures of popularity _of_ the decades.

With that in mind, we can now see a lot of the common cultural qualities between genres displayed in this chart. One of the things that I immediately noticed is that the two more musician-oriented styles of Jazz and Blues have relatively even popularity throughout the decades, with Jazz peaking in popularity with its oldest songs. I hypothesize that one of the biggest reasons for this is that both Jazz and Blues music sub-cultures are deeply rooted in traditional "standards", or famous songs. Lots of these standards are collected in books such as the Great American Songbook or The Real Book, and have been re-recorded by numerous famous musicians. Cover playing is more well respected in Jazz (and Blues) than most other styles.

In comparison, other styles which are more oriented to the average listener are not so rooted in the past, and
find a largely positive trend such that newer songs are more likely to be popular than older songs. Such is evidently the case (according to the data) for R&B, Rap, and Country Music. It may be suprising to find a style with such an old history as Country music in this category, but if you take the time to consider it, country may be an older _style_ of playing, but it is not culturally rooted in the playing and re-recording of old classics like Jazz and Blues are.
## How much we prefer "their old stuff"
A well known joke in some music circles is that if someone asks for your opinion on a band you have never heard of, all you need to do is say you "like their old stuff better", and you will almost always get positive response. I thought this would be an interesting thing to test with Python so I wrote a script to do some math. 

Note: I am using a python library called "sqldf" which performs SQL Queries on Pandas DataFrames. It is not exactly "fast" \* , but it is a great choice for ad-hoc one-time operations. I am also starting to learn another tool called DuckDb which looks promising for easily and (somewhat) quickly performing similar computations in the future.
### The Script
```
import pandas as pd
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

```

Using the output .csv file from this script, I generate a graph in Tableau which represents the average popularity of an artist over time (grouped by genre).
### The Results
![Dashboard2](Dashboard2.png)
### Interesting Notes
According to this graph, it is in fact the case that listeners tend to prefer the earlier music of the musicians they listen to. But now is a good time to keep in mind that this data is incomplete. On Kaggle, the data is listed as "15,150 classic hits", meaning that this dataset is not representative of all music. These are hits, and that may be skewing our data.
## Is it Pop? - A Classification Task
Next I made an attempt at a genre classification model. The goal was to make a model to classify what genre any given song is using only the data available upon release. I tried a couple different models which all had low degrees of success for classification (0.3 validation acc), so I decided to simplify the problem and design a model to tell if a particular song is classified as Pop. First I prepared the data for training by making sure all features were normalized. Then I tested several different Classification models (Decision Tree, KNN, MLP, Log Reg). Out of all these models, the two best performing models were the K-Nearest-Neighbors and Decision Tree, with KNN accuracy averaging around 84% and Decision Tree averaging around 75% (both on validation data with 1 being most consistent. (This was not a I wanted to ensemble the two, but whenever I tried, the accuracy was lower than if I just used the KNN. I was thinking about how to improve the Decision Tree, when I realized that the Decision Tree model was overfitting *a lot* (I don't know why I didn't realize this earlier, it was hitting ~98% on training data). So I got to hyperparameter tuning. The only parameter I was really interested in was _max_tree_depth_, as limiting tree depth is a universally-good way to reduce overfitting in a Decision Tree. Through a testing script (shown in tree\_and\_ensemble\_tests.py), I found that I could consistently get Decision Tree Validation accuracy into the low 80s by lowering the _max_tree_depth_ to somewhere between 5-10 (I chose 8). After this improvement, my ensemble was doing almost 90% on validation data, and I was confident in my solution.
One last thing I did at this point is I optimized the K value and weighting for KNN. It tended to be most accurate either with k either between 5-7 or at 1 with 1 being most consistent. The differences between any of these was rarely more than 1%, so I didnt sweat it too much. I did set the knn to be weighted by distance just as a fallback in case I wanted to experiment with an increased K later on since distance weighting performed better in tests for higher Ks (1-3%).
### Final Model and Results
In my final test of the model (shown in final\_model.py), I trained my Decision Tree and a KNN Ensemble on both training and validation data and came up with a testing score of 88% Accuracy.
## Conclusion
This dataset is quite rich, and here I only really scratched the surface dealing with time and Genre. There is a lot of potential left in this dataset, particularly in the area of popularity prediction.
### What I learned, and what I would have done differently
One thing which was very significant about this experience is that it taught me to work slower, and really think through the possilibities. I had to keep in mind time scale and data availability, and . It was also during the course of this project that I discovered the idea of using SQL inside of Python not only for querying a database (which is the normal application), but for performing common transformations on dataframes. I also learned the importance of improving _before_ testing/validating. So much time can be saved by planning for contingencies and possible improvements before instead of after the tests have already been performed, and it also prevents the emotional/personal side from coming through and trying to justify bad practices.

<br><br>

\*_In order to execute the sql, sqldf converts the pd.DataFrame to an in-memory sqlite3 database before executing the query on the database and then converting it back to a DataFrame. A fun project for the future might be a sql-to-pandas converter which utilizes an input string of some simplified version of sql to generate the standard pandas syntax (also potential for a nice AI tool here). This would be performance-wise essentially on equal ground with pandas, and it would also be easy to integrate into many existing pandas workflows. Or I might also just start using polars now.._
