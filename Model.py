import pandas as pd
from sklearn import tree, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import  ConfusionMatrixDisplay, accuracy_score
from icecream import ic
import matplotlib.pyplot as plt
from typing import Any
from sklearn.utils import resample
from numpy import average
import numpy as np
from seaborn import lineplot
print('imports are over')

random_state = 42
data_section = pd.read_csv("ClassicHit2.csv")
genres = ['Alt. Rock', 'Blues', 'Country', 'Disco', 'EDM', 'Folk', 'Funk',
       'Gospel', 'Jazz', 'Metal', 'Pop', 'Punk', 'R&B', 'Rap', 'Reggae',
       'Rock', 'SKA', 'Today', 'World']



#-------------------------------------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------------------------------------
def regress_to_class(model, in_x):
    return [num.round() for num in model.predict(in_x)]

def vote_prediction(km, tree, in_x):
    predictions = pd.DataFrame()
    predictions['km'] = km.predict(in_x)
    predictions['tree'] = tree.predict(in_x)
    return predictions.apply(vote,axis=1)

def vote(pred: pd.Series):
    return round(average(pred))

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

tree_acc_list: list[float]= []
knn_acc_list: list[float]= []
combo_acc_list: list[float]= []

data_tup = ((train_x,train_y),
            (val_x,val_y),
            # (test_x,test_y)
            )
iters = 2
max_i = 12
for i in range(max_i):
    tree_acc_list.append(0.0)
    knn_acc_list.append(0.0)
    combo_acc_list.append(0.0)
    print(i)
    for j in range(iters):
        max_depth = i+1
        tree_model = tree.DecisionTreeRegressor(max_depth=max_depth)
        knn_model = neighbors.KNeighborsRegressor()

        tree_model.fit(train_x,train_y)
        knn_model.fit(train_x,train_y)

        data_tup = ((train_x,train_y),(val_x,val_y),
                    # (test_x,test_y)
                    )
        data_names = ('training','validation',
                      # 'testing'
                      )
        train = True
        for x,y in data_tup:

            tree_acc = accuracy_score(y_true=y,y_pred=regress_to_class(tree_model,x))
            knn_acc = accuracy_score(y_true=y,y_pred=regress_to_class(knn_model,x))
            combo_acc = accuracy_score(y_true=y,y_pred=vote_prediction(knn_model,tree_model,x))
            if not train:
                tree_acc_list[i] += tree_acc
                knn_acc_list[i] += knn_acc
                combo_acc_list[i] += knn_acc

            # print(f'accuracy  k-means: {knn_acc}')
            # print(f'accuracy tree: {tree_acc}')
            # print(f'accuracy combo: {combo_acc}')
            # ConfusionMatrixDisplay.from_predictions(y_true=y,y_pred = regress_to_class(knn_model,x))
            # ConfusionMatrixDisplay.from_predictions(y_true=y,y_pred = regress_to_class(tree_model,x))
            # ConfusionMatrixDisplay.from_predictions(y_true=y,y_pred = vote_prediction(knn_model,tree_model,x))
            # plt.show()
            train = False
    tree_acc_list[i] = tree_acc_list[i]/iters
    knn_acc_list[i] = knn_acc_list[i]/iters
    combo_acc_list[i] = combo_acc_list[i]/iters

dta = pd.DataFrame({'Max_Depth':[item+1 for item in list(range(max_i))], 
                    'Tree':tree_acc_list, 
                    'Knn': knn_acc_list, 
                    'Combo': combo_acc_list})
lineplot(x = 'Max_Depth', y = ['Tree', 'KNN', 'Combo'], data=dta)
plt.savefig('acc_list.png')
plt.show()
