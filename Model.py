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
print('imports are over')

random_state = 320
data = pd.read_csv("ClassicHit2.csv")
genres = ['Alt. Rock', 'Blues', 'Country', 'Disco', 'EDM', 'Folk', 'Funk',
       'Gospel', 'Jazz', 'Metal', 'Pop', 'Punk', 'R&B', 'Rap', 'Reggae',
       'Rock', 'SKA', 'Today', 'World']

def encode_genres(In: pd.Series):
    return genres.index(In.Genre)

def encode_pop(In: pd.Series):
    match In.Genre:
        case 'Pop':
            return 1
        case _ :
            return 0

ic(data.columns)
data['pop_int'] = data.apply(encode_pop, axis=1)
train, val = train_test_split(data, test_size = 0.2)
test, val = train_test_split(val, test_size = 0.5)
pop_train = train[train['Genre']=="Pop"]

balanced_train = pd.concat([resample(train,replace=False, n_samples=len(pop_train),random_state=random_state),
                   resample(pop_train, replace=False, n_samples=len(pop_train),random_state=random_state)])

ic(balanced_train['Genre'].describe())

ic(len(train))
ic(len(val))
ic(len(test))

#--------------------------------------------------------
# Model 1 (Genre)
#--------------------------------------------------------


first_model_x_columns: list[str] = ["Year","Duration","Time_Signature","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","years_since_debut"]

first_model_y_columns: list[str] = ["pop_int"]

second_model_x_columns: list[str] =  first_model_x_columns+first_model_y_columns

second_model_y_columns: list[str] = ["Popularity"]

mod_1_x: pd.DataFrame = train[first_model_x_columns]
mod_1_y: pd.DataFrame = train[first_model_y_columns]

bal_mod_1_x: pd.DataFrame = balanced_train[first_model_x_columns]
bal_mod_1_y: pd.DataFrame = balanced_train[first_model_y_columns]

models: list[tuple[Any, str,bool]] = [(tree.DecisionTreeRegressor(),"decision tree",False),
                                 (neighbors.KNeighborsRegressor(),"k-nearest neighbors",False),
                                 # (svm.SVC(),"support vector classifier")
                                 ]
ind = 0
matrices: list[np.ndarray] = []

def regress_to_class(model,x_data):
    return [num.round() for num in model.predict(X = x_data)]

for model, model_name, balanced in models:
    if balanced:
        model.fit(bal_mod_1_x,bal_mod_1_y)
    else:
        model.fit(mod_1_x,mod_1_y)

    acc = accuracy_score(y_true=mod_1_y,y_pred=regress_to_class(model,mod_1_x))
    ic(f'training_accuraccy {model_name}: {acc}')

    acc = accuracy_score(y_true=val[first_model_y_columns],y_pred=regress_to_class(model,val[first_model_x_columns]))
    ic(f'validation_accuracy {model_name}: {acc}')
    cmd = ConfusionMatrixDisplay.from_predictions(y_true=val[first_model_y_columns[0]],y_pred=np.array(regress_to_class(model,val[first_model_x_columns])))
    matrices.append(cmd.confusion_matrix)
    plt.show()

    models[ind] = (model,model_name,balanced)
    ind +=1

predictions = pd.DataFrame()
for model, model_name, _ in models:
    predictions[model_name] = model.predict(val[first_model_x_columns])

def vote(pred: pd.Series):
    return round(average(pred))


voted_pred = predictions.apply(vote,axis=1)
acc = accuracy_score(y_true=val[first_model_y_columns[0]],y_pred=voted_pred)
ConfusionMatrixDisplay.from_predictions(y_true=val[first_model_y_columns[0]],y_pred=voted_pred)
ic(f'validation_accuracy from vote: {acc}')
plt.show()
