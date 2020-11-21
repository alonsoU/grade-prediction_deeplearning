import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import pylab as pyl
from classfolder.tools import MarkTools
import classfolder.frameprocess as preprocess
from classfolder.layers import FactorizationMachine
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import os

logpath = pl.Path(os.curdir)
print(logpath)
root_logdir = os.path.join(logpath, "pCM_logs")
def get_run_logdir(model_name):
    """Entrega un path donde crear datos de la sesion de TensorFlow.
    Típicamente mediante callback a TensorBoard.
    """
    import time
    run_id = time.strftime(model_name+"_%Y-%m-%d_%H-%M-%S")
    return os.path.join(root_logdir, run_id)
def split_label(DataFrame, label='nota', label_toframe=True):
    """Separa dataframe de su designado target o label"""
    if label_toframe:
        data_label = DataFrame.loc[:,label].to_frame()
    else:
        data_label = DataFrame.loc[:,label]
    data = DataFrame.drop([label], axis=1)
    return data, data_label
def batching_dataset(train, target, batch_size=32, prefetch=1):
    """Recive train data y targey, lo trasforma en tf.data.Dataset,
    lo junta en tuplas(requerido por Model.fit(x) al pasar objeto Dataset),
    agrupa la secuencia en bachas y finalmente aplica prefetch eficiencia de
    lectura.
    batch_size: tamaño de la bacha.
    prefetch: cantidad de datos al que se le aplica prefetch.
    """
    train = tf.data.Dataset.from_tensor_slices(train)
    target = tf.data.Dataset.from_tensor_slices(target)
    data = tf.data.Dataset.zip((train, target))
    data = data.batch(batch_size).prefetch(prefetch)
    return data

path = pl.Path('data/notas_clean.csv')
col_label = ['asignatura', 'curso', 'año', 'rut', 'fecha', 'nota']
col_use=['asignatura','nota','rut','fecha','curso','año']
crude = pd.read_csv(path, delimiter=',', names=col_label)
df = pd.DataFrame(data=crude)

df_18 = df.loc[df['año'] == 2018, :]
df_19 = df.loc[df['año'] == 2019, :]

mt = MarkTools(df)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
def split_data(DataFrame, split=split, prop_column='curso'):
    """Separa indices de dataframe en train y test data, mediante objeto
    scikit-learn, para luego construir un generador pero retorna diferentes
    particiones del dataframe en cada interación.
    split: instancia de sk-learn dedicada a generar indices en dataframe
    prop_column: columna que se utiliza como mascara para la partición,
        correspondiendo la proporción de los elementos en esta.
    """
    for train_index, test_index in split.split(DataFrame, DataFrame[prop_column]):
        train_frame = DataFrame.iloc[train_index].copy()
        test_frame = DataFrame.iloc[test_index].copy()
        yield train_frame, test_frame

train_data, test_data = split_data(df_18).__next__()
train_data, validation_data = split_data(train_data).__next__()

test, test_label = split_label(test_data)
validation, validation_label = split_label(validation_data)
train, train_label = split_label(train_data)

tf_pipe = preprocess.LookUpFrame(['asignatura', 'curso', 'año', 'rut'])
tables, keys = tf_pipe.dtypes_toarray(df).sort_column('rut').make_tables()
indices, keys = tf_pipe.to_index(train)
sparse_tensors, keys = tf_pipe.to_onehot(train)
print(indices)

num_names, cat_names = ['nota'], ['asignatura','curso','rut']
date_names = ['fecha']
ignored = ['año']

num_pipe = Pipeline([('mark_tweak', preprocess.NumProcess()),
    ('normalize', MinMaxScaler())
    ])
date_pipe = Pipeline([('scaler_per_year', preprocess.DateProcess()),
    ])
"""
cat_pipe = Pipeline([('onehot', OneHotEncoder(categories=obj_cat, sparse=False))
])
transformer = [('cat', cat_pipe, cat_names),
    ('date', date_pipe, date_names)]

fullpipe = ColumnTransformer(transformer)
train_dummy = fullpipe.fit_transform(train)
train_label = num_pipe.fit_transform(train_label)
validation_dummy = fullpipe.fit_transform(validation)
validation_label = num_pipe.fit_transform(validation_label)
n, p = train_dummy.shape

train_data = batching_dataset(train_dummy, train_label)
validation_data = batching_dataset(validation_dummy, validation_label)
"""

regular_inputs = keras.layers.Input(shape=[8])
categories = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
