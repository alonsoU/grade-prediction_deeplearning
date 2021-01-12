import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pathlib as pl
import os

def get_run_dir(model_path):
    # Creates all directories specified by String model_path parameter, if any already exists
    # create childs from there. Then creates a final child directory with the date as name.
    rootpath = pl.Path(os.curdir)
    import time
    model_path = pl.Path(model_path) / time.strftime("%Y-%m-%d_%H-%M-%S")
    for part in model_path.parts:
        rootpath /= part
        rootpath.mkdir(exist_ok=True)
    return rootpath

encoder = keras.models.load_model('./records/glorot/5bit-dim/noise_s08-p04/2021-01-12_16-17-13/subj_focus.h5')

path = pl.Path('data/archivos_mayor-cleanframe.csv')
with open(path) as file:
    # making sure all files are closed
    df = pd.read_csv(file).iloc[:, 1:]
    names = ['sede', 'ano', 'curso', 'semestre', 'asignatura',
             'profesor-rut']
    print(df)
    cats = df.drop(columns=["nota", "sede", "ano", "semestre"])
    # nested dict{names:{categories: _ }}
    nest = {}
for column in cats:
    try:
        layer = encoder.get_layer(name="".join([column, "_embed"]))
        nest[column] = {value:{i+1:list(layer(i+1))} for i, value in enumerate(sorted(cats[column].unique()))}
    except ValueError:
        pass
jsons = get_run_dir("jsons")
path_str = str(jsons)+"embeddings.json"
with open(path, 'w') as f:  # writing JSON object
    json.dump(nest, f)
