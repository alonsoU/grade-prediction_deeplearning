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
print('checkpoint1')
def force_dtype(value, dtype=int):
    try:
        value = dtype(value)
        return value
    except (ValueError, TypeError):
        if isinstance(value, str):
            values = value.split()
            if not values:
                return None
            else:
                allnum = all([v.isnumeric() for v in values])
                if allnum:
                    return values
        else:
            return None
def force(DataFrame, dtype=int, dropna=False, any=True):
    if not callable(dtype):
        raise TypeError("'dtype' must be callable")
    DataFrame = DataFrame.applymap(lambda x: force_dtype(x, dtype))
    if dropna:
        how = 'any' if any else 'all'
        DataFrame = DataFrame.dropna(how=how)
    return DataFrame
def mean_nonzeros(array, interval=[10,70]):
    if len(array) == 0:
        raise ValueError('Empty input array')
    i = 0
    sum = 0
    allow_types = (int, float)
    for v in array:
        if not isinstance(v, allow_types):
            raise ValueError('Allowed type inside array:', allow_types,
                'Input type:', type(v))
        elif int(v):
            if v > interval[1]:
                v = 70
            elif (v < interval[0]) & (v >= 1):
                v *= 10
            elif v < 1:
                continue
            sum+=v
            i += 1
    try:
        return round(sum/i, 1)
    except ZeroDivisionError:
        # print('Array is full of zeros')
        return None
def count_na(DataFrame, label):
    c = 0
    for value in DataFrame[label]:
        if pd.isna(value):
            c+=1
    print("NaN count:", c)

path = pl.Path("./data/archivos_mayor")
all_files = path.glob('**/*.txt')
all_files = list(all_files)
print('checkpoint2')
names = ["colegio", "sede", "año", "curso", "nose", "asignatura",
    "profesor-rut", "profesor-nombre", "semestre", "alumno-rut",
    "alumno-nombre"]
collection = []
for file in all_files:
    if "trash" in str(file.parent):
        continue
    sub_df = pd.read_table(file,
        sep='|',
        header=None,
        engine='python',
        skip_blank_lines=False,
        )
    collection.append(sub_df)
print('checkpoint3')
df = pd.concat(collection, axis=0)
mapper = dict(list(enumerate(names)))
df = df.rename(columns=mapper)
drop_categories = ['nose', 'colegio', 'profesor-nombre', 'alumno-nombre']
df = df.drop(drop_categories, axis=1)
len_cat = len(names) - len(drop_categories)
# corrige columnas numericas
df.iloc[:,len_cat:] = force(df.iloc[:,len_cat:].fillna(0))
# modifica columnas categoricas a dtype=string
df.iloc[:,:len_cat] = force(df.iloc[:,:len_cat], dtype=str).applymap(lambda x:
    x.lower().strip())
df = df.dropna()
df['semestre'] = df['semestre'].map(lambda x: x.split()[0]) # saca 'semestre'
df = df.convert_dtypes()
# Crea nueva columna 'notas promediando todos los valores numericos(nonzero)
# de una fila'
df['nota'] = df.iloc[:,len_cat:].apply(mean_nonzeros,
    axis=1, raw=True)
df = df.drop(df.select_dtypes(np.int64).columns.tolist(), axis=1)
print('checkpoint4')
df = df.dropna()
def check_sparcity(DataFrame, min_sample=10, dtypes='string', drop=True):
    for column in DataFrame.select_dtypes(dtypes):
        sub_mask = DataFrame[column].value_counts() >= min_sample
        mask = DataFrame[column].replace(sub_mask)
        DataFrame[column] = DataFrame[column].where(mask)
    if drop:
        DataFrame = DataFrame.dropna()
        DataFrame.reset_index(inplace=True, drop=True)
    return DataFrame
# En cada columna categorica, elimina valores que con conteo menos a 'min_sample'
print(df)
df = check_sparcity(df, dtypes='string')
# Estandariza cursos y los ordena naturalmente mediante string method
df['curso'] = df['curso'].apply(lambda x: ' '.join(x.split()[:2][::-1]))
print(df['asignatura'].unique().sort())
# df.to_csv('data/archivos_mayor-dirtyframe.csv')
#################################################################
"""                TODO LISTO PARA ANALIZAR                  """#
#################################################################

def check_for_errors(files_list,
    put_in_trash=True,
    method=pd.read_csv,
    sep='|',
    show_readed=False):
    # Detencta archivos corruptos en una lista 'files_list', mediante metodo
    # pandas de lectura
    # files_list tiene que ser lista de objetos WindowsPath
    import os
    from shutil import move
    import pathlib as pl
    for file in files_list:
        parent = file.parent
        if "trash" in str(parent):
            continue
        try:
            with open(file, 'r') as f:
                method(f,
                sep='|',
                engine='python'
                )
                if show_readed:
                    print(f"File {f.name} was successfully opened")
        except Exception as inst:
            trash = parent / f'trash-{parent.name}'
            if not trash.is_dir():
                trash.mkdir()
                print(f"{trash.name} was made")
            print(type(inst))
            print(inst.args)
            if put_in_trash:
                # os.path.join(trash, file.name)
                new_location = trash / file.name
                move(file, new_location)
                print(f"{new_location.name} was move to trash")
            continue
# check_for_errors(all_files, put_in_trash=True, show_readed=False)
