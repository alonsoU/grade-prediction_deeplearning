import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import pylab as pyl
from classfolder.tools import MarkTools
from classfolder.frameprocess import CatProcess, GradeProcess, PivotMark, DateProcess, DropSparce
from classfolder.layers import FactorizationMachine, MultiHeadAttention
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import os

curr = pl.Path(os.curdir)
root_logdir = os.path.join(curr, "pCM_logs")
def get_run_logdir(model_name):
    import time
    run_id = time.strftime(model_name+"_%Y-%m-%d_%H-%M-%S")
    return os.path.join(root_logdir, run_id)


path = pl.Path('data/notas_clean.csv')
col_label = ['asignatura', 'curso', 'año', 'rut', 'fecha', 'nota']
col_use=['asignatura','nota','rut','fecha','curso','año']
crude = pd.read_csv(path, delimiter=',', names=col_label)
df = pd.DataFrame(data=crude)
"""divide el dataset en los dos años disponble. Se utilizara el año 2018
primeramente para estudio de la maquina de factorización, y en posterior
embedding del factor_matrix. luego de esto, con el año 2019 se estudiara algun
modelo de deeplearning.
"""
df_18 = df.loc[df['año'] == 2018, :]
df_19 = df.loc[df['año'] == 2019, :]

df = df.astype(dtype={'curso':'object'}, copy=False)
"""Generando una matriz de categorical values, como 'categories:' en OneHotEncoder
para asi, contabilizar todas los posibles valores categroicos, independiente que
no esten en el sub-espacio de entrenamiento strat_train_set.
"""
cat_columns = df.select_dtypes(include=np.object)
t = MarkTools(df)
obj_cat = []
for column in cat_columns:
    aux = cat_columns[column].value_counts().index.tolist()
    obj_cat.append(np.array(aux))

obj_cat[-1] = np.sort(obj_cat[-1])#ordena ruts de mayor a menor
from sklearn.model_selection import StratifiedShuffleSplit
"""Separa el datset en train-80% y test-20% (test_size=0.2), solo una vez,
este porsentaje de seleccion se aplica a cada sub-muestra de cursos,
dado cat_subj, manteniendo asi su proporción.
"""
df_18 = df_18.astype(dtype={'curso':'object'}, copy=False)
#print(df_18['curso'].value_counts() / len(df_18))
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(df_18, df_18['curso']):
    strat_train_set = df_18.iloc[train_index].copy()
    strat_test_set = df_18.iloc[test_index].copy()
#print(strat_test_set['curso'].value_counts() / len(strat_test_set))
m_test_label = strat_test_set.loc[:,'nota'].to_frame()
m_test = strat_test_set.drop(['nota'],axis=1)

for train_index, validation_index in split.split(strat_train_set, strat_train_set['curso']):
    m_train_set = strat_train_set.iloc[train_index].copy()
    m_validation_set = strat_train_set.iloc[validation_index].copy()

m_train_label = m_train_set.loc[:,'nota'].to_frame()
m_train = m_train_set.drop(['nota'], axis=1)

m_validation_label = m_validation_set.loc[:,'nota'].to_frame()
m_validation = m_validation_set.drop(['nota'], axis=1)

"""
Creación de Pipelines para la transformación de la data, para su subsiguiente
analisis y modelamiento.
"""
num_names, cat_names = ['nota'], ['asignatura','curso','rut']
date_names = ['fecha']
ignored = ['año']

"""Pipeline de scikit-learn dedicada a procesar datos formato pandas o inverior
array values: (name, trasformer_instance)
"""
num_pipe = Pipeline([('mark_tweak', GradeProcess()),
    ('normalize', MinMaxScaler())
    ])
date_pipe = Pipeline([('scaler_per_year', DateProcess()),
    ])
cat_pipe = Pipeline([('onehot', OneHotEncoder(categories=obj_cat, sparse=False))
    ])
"""Fullpipe hace todas las transformaciones necesarias. Para cada columna.
array values: (name, pipeline, transformed_columns)
"""
transformer = [('cat', cat_pipe, cat_names),
    ('date', date_pipe, date_names)]
fullpipe = ColumnTransformer(transformer)
train_dummy = fullpipe.fit_transform(m_train)
train_label = num_pipe.fit_transform(m_train_label)
validation_dummy = fullpipe.fit_transform(m_validation)
validation_label = num_pipe.fit_transform(m_validation_label)

"""Se ingresa la secuencia de datos a data.Dataset para separarlo en batches de
36 vectores por forwardpass. Antes, ordena en tuplas (train, target) como
formato requerido por fit(), metodo de Model class. Luego de esto se separa en
batch y finalmente se aplica prefetch(1) para optimizar el flujo de datos al
modelo.
"""
n, p = train_dummy.shape
def batching_dataset(train, target, batch_size=32, prefetch=1):
    train = tf.data.Dataset.from_tensor_slices(train)
    target = tf.data.Dataset.from_tensor_slices(target)
    data = tf.data.Dataset.zip((train, target))
    data = data.batch(batch_size).prefetch(prefetch)
    return data

train_data = batching_dataset(train_dummy, train_label)
validation_data = batching_dataset(validation_dummy, validation_label)
"""
input = keras.layers.Input(shape=[p])
embedding = keras.layers.Embedding(p, 100)
mh_attention = MultiHeadAttention(4)([embedding, embedding])
dense = keras.layers.Dense(1, activation='relu')(mh_attention)
mh_model = keras.Model(inputs=embedding, outputs=dense)

mh_model.compile(loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=0.01),
    metric='accuracy')
mh_model.fit(train_data, epochs=25,
    validation_data=validation_data,
    callbacks=[tensorboard_callback,
    early_cb])
"""
pre_encode = keras.Sequential([
    keras.layers.Input(shape=[p]),
    FactorizationMachine(50, activation='relu', k=20),
    FactorizationMachine(50, activation='relu', k=10),
    keras.layers.Dropout(0.5),
    FactorizationMachine(10, activation='relu', k=10),
    keras.layers.Dropout(0.2),
    FactorizationMachine(1,activation='relu', k=10)
])
control = keras.Sequential([
    keras.layers.Input(shape=[p]),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])


run_logdir = get_run_logdir("MHA_tryouts")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_logdir,
    write_images=True,
    histogram_freq=1)
early_cb = keras.callbacks.EarlyStopping(patience=5,
    restore_best_weights=True)

control.compile(loss=tf.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(lr=0.01),)
control.summary()
control.fit(train_data, validation_data=validation_data,
    epochs=5,
    callbacks=[tensorboard_callback, early_cb])
"""
pre_encode.compile(loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=0.01),)

pre_encode.summary()

pre_encode.fit(train_data, epochs=25,
    validation_data=validation_data,
    callbacks=[tensorboard_callback,
    early_cb])
"""
test_dummy = fullpipe.fit_transform(m_test)
test_label = num_pipe.fit_transform(m_test_label)

#pre_encode.save('factor_machine_encoder.h5')
#control.evaluate(test_dummy, test_label)
#pre_encode.evaluate(test_dummy, test_label)
