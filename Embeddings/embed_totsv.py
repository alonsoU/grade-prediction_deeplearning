import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pathlib as pl
import os
import io

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
    cats = df.drop(columns=["nota", "sede", "ano", "semestre", "curso"])
    #cats = {column: {value: i + 1 for i, value in enumerate(sorted(cats[column].unique()))} for column in cats}

path = get_run_dir("tsvs")

for column in cats:
    out_meta = io.open(str(path) + f"/{column}_meta.tsv", 'w', encoding='utf-8')
    out_vec = io.open(str(path) + f"/{column}_vecs.tsv", 'w', encoding='utf-8')
    unique_values = sorted(cats[column].unique())
    out_meta.write('\t'.join([column, 'nota media total']) + "\n")
    for i, value in enumerate(unique_values):
        layer = encoder.get_layer(name="".join([column, "_embeds"]))
        vec = layer(i)
        mask = df[column] == value
        more_meta = round(df['nota'].loc[mask].mean(), 2)
        values = [value, str(more_meta)]
        out_meta.write('\t'.join(values) + "\n")
        out_vec.write('\t'.join([str(x) for x in vec.numpy()]) + "\n")
    out_meta.close()
    out_vec.close()

