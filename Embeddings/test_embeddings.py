import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import pylab as pyl
from classfolder.tools import MarkTools
import classfolder.frameprocess as pre
import classfolder.layers as mylayers
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from tensorflow import keras
import tensorflow as tf
import os
import seaborn as sb
import h5py

path_basedata = pl.Path('data/archivos_mayor-cleanframe.csv')
basedata = pd.read_csv(path_basedata, delimiter=',', low_memory=False, index_col=0,)
tf_tables = pre.LookupFrame(basedata,
    ['asignatura', 'curso', 'año', 'semestre', 'alumno-rut', 'profesor-rut',],
    prop_bucket=0.01,
    )

path_newdata = pl.Path('data/notas_cleanframe.csv')
newdata = pd.read_csv(path_newdata, delimiter=',')

model_path = "records/clean_subjects/glorot/6bit-dim/noise_s08-p04/2020-09-14_13-11-04/noise_s08-p04.h5"
embeds = keras.models.load_model(model_path)
embeds.summary()
mask = basedata['alumno-rut'] == basedata['alumno-rut'].value_counts().index[0]
pp = basedata['nota'].loc[mask].mean()
# print(pp)
def get_run_dir(model_path):
    rootpath = pl.Path(os.curdir)
    import time
    model_path = pl.Path(model_path) / time.strftime("%Y-%m-%d_%H-%M-%S")
    for part in model_path.parts:
        rootpath /= part
        rootpath.mkdir(exist_ok=True)
    return rootpath
def check_foreign(categ, verbose=True):
    num = 0
    total = 0
    difer = []
    same = []
    outbound = []
    last_trueindex = tf.constant(basedata[categ].unique()[-1], dtype=tf.string)
    for i, student in enumerate(newdata[categ].unique()):
        student_tensor = tf.constant(student, dtype=tf.string)
        total += 1
        try:
            t = tf_tables[categ].lookup(student_tensor)
            if t > tf_tables[categ].lookup(last_trueindex):
                num += 1
                difer.append(student)
            else:
                same.append(student)
        except:
            if verbose:
                outbound.append(student)
                print('exception index: ', i)
        if i%1000 == 0 and i>=500:
            if verbose:
                print(f'{i}° checkpoint')
    difer = np.array(sorted(difer))
    same = np.array(sorted(same))
    outbound = np.array(sorted(outbound))
    if verbose:
        print('ultimo valor entrenado: ', last_trueindex)
        print(f'{categ} proporcion de valores diferidos: ', num/total)
        print("Valores comumes: \n", same)
        print("Valores diferidos: \n", difer)
        print("Valores fuera de rango: \n", outbound)
    return same, difer, outbound
s, d, o = check_foreign('alumno-rut', verbose=False)

vec = embeds.get_layer('alumno-rut')(tf.constant(1))

def new_metadata(frame):
    # frame()
    pass

# embedding_name = 'embed_subjects'
def embedding_tsv(name, model, statictables, frame, run_dir,
    embedding_name=None,
    verbose=True):
    if embedding_name is None: layer = model.get_layer(name)
    else: layer = model.get_layer(embedding_name)
    table = statictables[name]
    unique_values = frame[name].unique()
    if callable(run_dir):
        if verbose: print("Callable 'run_dir' input given")
        run_dir = run_dir('data/tsv_visuals/' + name)
    if not isinstance(run_dir, pl.Path):
        run_dir = pl.Path(run_dir)
    import io
    out_meta = io.open(run_dir / f'{name}_meta.tsv', 'w', encoding='utf-8')
    out_vec = io.open(run_dir / f'{name}_vecs.tsv', 'w', encoding='utf-8')
    if verbose: print(f'Files open at {run_dir}')
    for i, value in enumerate(unique_values):
        trace = tf.constant(value)
        num = table.lookup(trace)
        vec = layer(num)
        mask = frame[name] == value
        # if name=='alumno-rut' or name=='asignatura':
        if i == 0:
            out_meta.write('\t'.join([name, 'nota media total']))
        more_meta = round(frame['nota'].loc[mask].mean(), 2)
        values = [value, str(more_meta)]
        if name=='profesor-rut':
            if i == 0:
                out_meta.write('\t' + 'asignatura primaria' + '\n')
            more_meta = frame['asignatura'].loc[mask].value_counts().index[0]
            values.append(str(more_meta))
        else: out_meta.write('\n')
        out_meta.write('\t'.join(values) + "\n")
        out_vec.write('\t'.join([str(x) for x in vec.numpy()]) + "\n")
    out_meta.close()
    out_vec.close()
    if verbose: print('Files writen')

embedding_tsv(name='asignatura',
    # embedding_name='embed_subjects',
    model=embeds,
    statictables=tf_tables,
    frame=basedata,
    run_dir=get_run_dir
    )
