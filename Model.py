import pandas as pd
from sklearn import tree, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from icecream import ic
import matplotlib.pyplot as plt
from typing import Any
from sklearn.utils import resample
from numpy import average
import numpy as np
print('imports are over')

random_state = 42
data_section = pd.read_csv("ClassicHit2.csv")
genres = ['Alt. Rock', 'Blues', 'Country', 'Disco', 'EDM', 'Folk', 'Funk',
       'Gospel', 'Jazz', 'Metal', 'Pop', 'Punk', 'R&B', 'Rap', 'Reggae',
       'Rock', 'SKA', 'Today', 'World']



#-------------------------------------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------------------------------------
def encode_genres(In: pd.Series):
    return genres.index(In.Genre)

def encode_pop(In: pd.Series):
    match In.Genre:
        case 'Pop':
            return 1
        case _ :
            return 0
#--------------------------------------------------------
# Genre Prediction Model (K-Means + Decision Tree)
#--------------------------------------------------------

ic(data_section.columns)
data_section['pop_int'] = data_section.apply(encode_pop, axis=1)
train, val = train_test_split(data_section, test_size = 0.2)
test, val = train_test_split(val, test_size = 0.5)
pop_train = train[train['Genre']=="Pop"]


ic(len(train))
ic(len(val))
ic(len(test))

first_model_x_columns: list[str] = ["Year","Duration","Time_Signature","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]

first_model_y_columns: list[str] = ["pop_int"]

second_model_x_columns: list[str] =  first_model_x_columns+first_model_y_columns

second_model_y_columns: list[str] = ["Popularity"]

train_x = train[["Year","Duration","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]]
train_y = train["pop_int"]
val_x = val[["Year","Duration","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]]
val_y = val["pop_int"]
test_x = test[["Year","Duration","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]]
test_y = test["pop_int"]


data_tup = ((train_x,train_y),
            (val_x,val_y),
            # (test_x,test_y)
            )
knn_model = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_model.fit(pd.concat([train_x,val_x]), pd.concat([train_y,val_y]))

knn_acc = accuracy_score(y_true=test_y,y_pred=knn_model.predict(test_x))
conf_mat = confusion_matrix(y_true=test_y,y_pred=knn_model.predict(test_x))

print(knn_acc)
print(conf_mat)
