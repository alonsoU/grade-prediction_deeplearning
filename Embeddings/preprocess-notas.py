import pandas as pd
from classfolder.frameprocess import CatProcess, DropSparce
import pathlib as pl
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np

def explicit_text(string):
    newparts = []
    if '(' in string and ')' in string:
        i = string.index('(')
        j = string.index(')')
        parts = list(string)
        del parts[i:j+1]
        parts = ''.join(parts)
        return parts
    else:
        return string
def remove_symbols(string):
    symbols = ['?']#, '-']
    parts = list(string)
    for i, char in enumerate(parts):
        if char.isnumeric() or char in symbols:
            parts[i] = ''
    return ''.join(parts).strip()
def cut_redundant(string):
    redundant = ['comun', 'diferenciada']
    for r in redundant:
        if r in string:
            parts = string.split()
            try:
                i = parts.index(r)
                return ' '.join(parts[:i+1])
            except:
                return string
    return string

path = pl.Path('data/notas.csv')
col_label = ['asignatura', 'curso', 'año', 'alumno-rut', 'fecha', 'nota']
col_use=['asignatura','nota','rut','fecha','curso','año']
crude = pd.read_csv(path, names=col_label, delimiter='|')
df = pd.DataFrame(data=crude)

# Proceso de 'limpieza de Dataframe'(lowcase, quitar muestras deficientes,
# cursos a var. categorica).
cat_columns = df.select_dtypes(include=np.object).columns
df[cat_columns] = df[cat_columns].applymap(lambda x: x.lower().strip())

def check_sparcity(DataFrame, min_sample=10, dtypes='string'):
    for column in DataFrame.select_dtypes(dtypes):
        sub_mask = DataFrame[column].value_counts() >= min_sample
        mask = DataFrame[column].replace(sub_mask)
        DataFrame[column] = DataFrame[column].where(mask)
    DataFrame = DataFrame.dropna()
    DataFrame.reset_index(inplace=True, drop=True)
    return DataFrame
df = check_sparcity(df, dtypes=np.object)
df = df.convert_dtypes()

def rewrite(course_string):
    grade = {'primero': 1, 'segundo': 2, 'tercero': 3, 'cuarto': 4,
        'quinto': 5, 'sexto': 6, 'septimo': 7, 'octavo': 8}
    parts = course_string.split()
    level = parts[1][:-1] + 'a'
    number = str(grade[parts[0]])
    return " ".join([level, number+'-'+parts[2]])

df['curso'] = df['curso'].apply(lambda x: rewrite(x))
df['alumno-rut'] = df['alumno-rut'].apply(lambda x: '-'.join([x[:-1], x[-1]]))
df['asignatura'] = df['asignatura'].apply(lambda x:
    cut_redundant(remove_symbols(explicit_text(x))))

# df.to_csv('data/notas_dirtyframe.csv', index=False)
