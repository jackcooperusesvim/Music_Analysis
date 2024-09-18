import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DecTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LinearRegression
from icecream import ic
import matplotlib.pyplot as plt

data = pd.read_csv("ClassicHit2.csv")
genres = ['Alt. Rock', 'Blues', 'Country', 'Disco', 'EDM', 'Folk', 'Funk',
       'Gospel', 'Jazz', 'Metal', 'Pop', 'Punk', 'R&B', 'Rap', 'Reggae',
       'Rock', 'SKA', 'Today', 'World']

def encode_genres(In: pd.Series):
    return genres.index(In.Genre)

ic(data.columns)
data['genre_int'] = data.apply(encode_genres, axis=1)
ic(data.head())
train, test = train_test_split(data, test_size = 0.1)
train, val = train_test_split(train, test_size = (1/9))


first_model_x_columns: list[str] = ["Year","Duration","Time_Signature","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]

first_model_y_columns: list[str] = ["genre_int"]

second_model_x_columns: list[str] =  first_model_x_columns+first_model_y_columns

second_model_y_columns: list[str] = ["Popularity"]

mod_1_x: pd.DataFrame = train[first_model_x_columns]
mod_1_y: pd.DataFrame = train[first_model_y_columns]

def solve_with_model(model):
    mod_1 = DecTree()
    ax, fig = plt.subplots()
    mod_1.fit(mod_1_x,mod_1_y)
    train[f'mod_1_predict({first_model_y_columns[0]})'] = mod_1.predict(X = mod_1_x)



    matrix = ConfusionMatrixDisplay.from_predictions(y_true=mod_1_y,y_pred=mod_1.predict(X = mod_1_x),display_labels = genres)
    acc = accuracy_score(y_true=mod_1_y,y_pred=mod_1.predict(X = mod_1_x))
    ic(acc)


plt.show()
#--------------------------------------------------------
# Model 2 (Popularity)
#--------------------------------------------------------

#
# mod_2_x: pd.DataFrame  = train[second_model_x_columns]
# mod_2_y: pd.DataFrame  = train[second_model_y_columns]
#
# mod_2 = LinearRegression()
# mod_2.fit(mod_2_x,mod_2_y)
#
# train[f'mod_2_predict({second_model_y_columns[0]})'] = mod_2.predict(X = mod_2_x)
# mod_2.fit(mod_2_x,mod_2_y)
