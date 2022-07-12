import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
if __name__ == '__main__':
    from tools import MarkTools
else:
    from classfolder.tools import MarkTools
import time
import datetime
import tensorflow as tf

class LookupFrame():
    """Genera listado de valores tomados por las columnas del o los dtypes dados.
        # Dataframe: pandas.Dataframe
        # dtypes:
    """
    def __init__(self,
        DataFrame,
        keys=None,
        prop_bucket=0.05,
        sort=True,
        input_dtypes='string',
        dict_dtype=tf.int64):
        self.DataFrame = DataFrame
        self.keys = keys
        self.prop_bucket = prop_bucket
        self.sort = sort
        self.input_dtypes = input_dtypes
        self.dict_dtype = dict_dtype
        self.lengths = []
        self.num_buckets = []
        self.tables = []

        # """Genera listas de los valores en cada columna de DataFrame,
        # convenientemente ordenadas por número de ocurrencias.
        # """
        if self.keys is None:
            # Si no hay keys, se asignan las correspondientes al dtypes
            cat_columns = self.DataFrame.select_dtypes(include=self.input_dtypes)
            self.keys = cat_columns.columns.tolist()
        else:
            # Revisa inputs correctos y secciona el DataFrame correspondiente
            for i, key in enumerate(self.keys):
                if key not in DataFrame:
                     self.keys.pop(i)
            cat_columns = self.DataFrame.loc[:,self.keys]
        for column in cat_columns:
            if self.sort:
                values = pd.unique(cat_columns[column])
                sorted(values)
            else:
                values = cat_columns[column].value_counts().index.tolist()
                if column=='curso':
                    sorted(values)
            # Convierte el dtype correspondiente a la columna a tensorflow DType
            numeric = (int, float, np.int32, np.int64)
            if all(isinstance(x, numeric) for x in values):
                values = tf.constant(values, dtype=tf.int64)
            elif all(isinstance(x, (str)) for x in values):
                values = tf.constant(values, dtype=tf.string)
            else:
                # Solo soporta los types descritos arriba.
                raise TypeError(f'Types in {column} has to be'
                    f'{type(int).__name__}, {type(np.int64).__name__} or'
                    f'{type(str).__name__} entirely.')
            # Genera las tablas para cada columna. Cada una de estas tiene un
            # bucket para tratar variables no vistas en el DataFrame original,
            # y corresponde a una proporción de la misma.
            length = len(values)
            num_bucket = max(int(length * self.prop_bucket), 5) # 5 in bucket minimum
            # Es importrante para tf el dtype
            # Es conveniente empezar de 1, debido a que ciertas layers de
            # keras tienen por defecto el digito 0 como mask.
            """ATENTO A BUG"""
            indices = tf.range(start=1, limit=length+1, dtype=self.dict_dtype)
            table_init = tf.lookup.KeyValueTensorInitializer(values, indices)
            table = tf.lookup.StaticVocabularyTable(table_init, num_bucket)
            self.tables.append(table)
            self.num_buckets.append(num_bucket)
            final_length = length + num_bucket
            self.lengths.append(final_length)
    @tf.function
    def to_index(self, frame, name=None, same_shape=False):
        """Entrega los indices correspondientes a los valores en frame.
        **Actualmente soporta frames con las mismas columnas que
        self.DataFrame, Series, y DataFrames con columnas dentro de self.keys,
        pudiendo estar en diferente orden.
        """
        indices = []
        new_keys = []
        if not same_shape:
            if isinstance(frame, (list, np.ndarray)):
                if name is None:
                    raise ValueError('list type needs a name')
                i = self.keys.index(name)
                table = self.tables[i]
                keys_dtype =  table.key_dtype
                categories = tf.constant(frame,
                    dtype=keys_dtype)

                return table.lookup(categories)
            elif isinstance(frame, str):
                if name is None:
                    raise ValueError(f'{type(frame)} type needs a name')
                i = self.keys.index(name)
                table = self.tables[i]
                keys_dtype =  table.key_dtype
                categories = tf.constant(frame,
                    dtype=keys_dtype)

                return table.lookup(categories)
            elif isinstance(frame, pd.Series):
                if frame.name in self.keys:
                    new_keys.append(frame.name)
                    i = self.keys.index(frame.name)
                    table = self.tables[i]
                    keys_dtype = table.key_dtype
                    categories = tf.constant(frame.tolist(),
                        dtype=keys_dtype)
                    indices.append(table.lookup(categories))
                else:
                    raise KeyError(f'Column {column} can not be found in'
                    f'{repr(self.DataFrame)}, it has no table previously made')
            elif isinstance(frame, pd.DataFrame):
                for column in frame:
                    if column in self.keys:
                        new_keys.append(column)
                        i = self.keys.index(column)
                        table = self.tables[i]
                        keys_dtype = table.key_dtype
                        categories = tf.constant(frame[column].tolist(),
                            dtype=keys_dtype)
                        indices.append(table.lookup(categories))
                    else:
                        raise KeyError(f'Column {column} can not be found in'
                            f'{repr(self.DataFrame)}, it has no table previously'
                            'made')
            # elif isinstance(frame, tf.keras.layers.Embedding):
                # pass
            else:
                raise TypeError(f'frame input must be {pd.DataFrame}, '
                    f'{pd.Series}, {list} or {str}, '
                    f'input type: {type(frame)}')

            return indices, new_keys
        else:
            frame = frame.loc[:, self.keys]
            for i, column in enumerate(frame):
                table = self.tables[i]
                keys_dtype = table.key_dtype
                categories = tf.constant(frame[column].tolist(), dtype=keys_dtype)
                indices.append(table.lookup(categories))

            return indices, self.keys
    def get_lenght(self, name):
        i = self.keys.index(name)
        return self.lengths[i]
    def get_dtype(self, name):
        i = self.keys.index(name)
        return self.tables[i].key_dtype
    def __getitem__(self, key):
        index = self.keys.index(key)
        table = self.tables[index]
        return table

class CatProcess(BaseEstimator, TransformerMixin):
    def __init__(self,lowcase=True,subj_tocat=True,ignote_letter=True,inplace_cat=True): # no *args or **kargs
        self.lowcase, self.ignote_letter = lowcase, ignote_letter
        self.inplace_cat, self.subj_tocat = inplace_cat, subj_tocat
    def fit(self,X,y=None):
        self.MarkTools = MarkTools(X)
        return self # nothing else to do

    def transform(self,X,y=None):
        self.cat_columns = X.select_dtypes(include=np.object).columns
        if self.lowcase:
            X[self.cat_columns] = X[self.cat_columns].applymap(lambda x: x.lower().strip())
        if self.ignote_letter:
            X['curso'] = X['curso'].apply(lambda x: x[:-1].strip())#IGNORAR LETRA DE LOS CURSOS
        if self.subj_tocat:
            i = '' if self.inplace_cat else '_cat'
            X[f'curso{i}'.format(i)] = X['curso'].apply(lambda course:
                self.MarkTools.num_turn(course))#Tornar a numero en jerarquia
        self.categories_ = X.values
        return X

class PivotMark(BaseEstimator, TransformerMixin):
    """
    Esta clase requiere de CatProsses, debido a los dtypes de fecha y nota, y to_datetime func.
    """
    def __init__(self,freq='D',columns=['rut','asignatura'],fillna=True):
        self.freq = freq
        self.columns = columns
        self.fillna = fillna
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        X = X.pivot_table(values='nota', index=['fecha'],
            columns=self.columns,aggfunc=np.mean)
        X= X.resample('D', axis=0).asfreq()
        X = X.resample(self.freq).mean()
        if self.fillna: X.fillna(0, inplace=True)
        return X

class GradeProcess(BaseEstimator, TransformerMixin):
    def __init__(self,outliers_in=True):
        self.outliers_in = outliers_in
    def fit(self,X,y=None, grade_col=['nota']):
        self.grade_col = grade_col
        return self
    def transform(self,X,y=None):
        #Cambiando dtypes para hacer pivot y demaces
        X = X.astype(dtype={'nota':'int64'}, copy=False)
        if self.outliers_in:
            X.loc[X['nota'] < 10, 'nota'] = round(X.loc[X['nota'] < 10, 'nota']*10)
            X.loc[X['nota'] > 70, 'nota'] = 70
        else: X.dropna(X.loc[(X['nota'] < 10) | (X['nota'] > 70)].index)
        return X
# class GradeProcess():

class DateProcess(BaseEstimator, TransformerMixin):
    def __init__(self,pandas_todate=False):
        self.pandas_todate = pandas_todate
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        X = X.astype(dtype={'fecha':'int64'}, copy=False)
        if self.pandas_todate:
            X['fecha'] = X['fecha']*100 #a microsegundo
            X['fecha'] = pd.to_datetime(X['fecha'],unit='ms',origin='unix', exact=True)
        else:
            X['fecha'] = X['fecha']//10
            first = datetime.datetime.fromtimestamp(X['fecha'].min())
            last = datetime.datetime.fromtimestamp(X['fecha'].max())
            first_year = first.strftime('%Y')
            last_year = last.strftime('%Y')

            first_unix_year = datetime.date(int(first_year),1,1)
            first_unix_year = time.mktime(first_unix_year.timetuple())

            last_unix_year = datetime.date(int(last_year),12,31)
            last_unix_year = time.mktime(last_unix_year.timetuple())

            delta = (last_unix_year - first_unix_year)
            X.loc[:, 'fecha'] = (X['fecha'] - first_unix_year)/delta
        return X

class DropSparce(BaseEstimator, TransformerMixin):
    def __init__(self, min_sample=6, dtypes=np.object, exclude=None):
        self.min_sample = min_sample
        self.input_dtypes = dtypes
        self.exclude = exclude
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        self.cat_columns = X.select_dtypes(include=self.input_dtypes).columns.tolist()
        self.cat_columns = X.columns.tolist()
        if self.exclude:
            for e in self.exclude:
                try:
                    i = self.cat_columns.index(e)
                except ValueError:
                    raise ValueError(f'{e} is not in DataFrame')
                self.cat_columns.pop(i)
        v = X[self.cat_columns]
        X = X[v.replace(v.apply(pd.Series.value_counts)).gt(self.min_sample).all(1)]
        return X
