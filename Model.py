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

random_state = 234
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
acc_list = []
for i in range(0,10,2):
    n_neighbors = i+1
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_x,train_y)

    print(f'number of neighbors: {n_neighbors}')
    data_names = ('training','validation','testing')
    counter = 0
    training = True
    for x,y in data_tup:
        print(data_names[counter])
        counter+=1
        knn_acc = accuracy_score(y_true=y,y_pred=knn_model.predict(x))
        if not training:
            acc_list.append(knn_acc)

        print(f'\taccuracy  k-means: {knn_acc}')
        print(confusion_matrix(y_true=y,y_pred = knn_model.predict(x)))
        training = not training
fig, ax = plt.subplots()
ax.plot(range(1,len(acc_list)+1),acc_list)
plt.show()

