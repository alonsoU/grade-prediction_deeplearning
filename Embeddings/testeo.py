import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import pylab as pyl
import scipy as sp
from classfolder.tools import MarkTools
from classfolder.frameprocess import CatProcess, GradeProcess,PivotMark, DateProcess, DropSparce
from classfolder.frameprocess import LookupFrame
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import time
import datetime
import os
from classfolder.frameprocess import FrametoDataset
"""
s = time.time()
print(s)
date = datetime.datetime.fromtimestamp(s)
print(date.strftime('%m'))
"""
path = pl.Path('data/notas.csv')
col_label = ['asignatura', 'curso', 'año', 'rut', 'fecha', 'nota']
col_use=['asignatura','nota','rut','fecha','curso','año']
method= pd.read_csv
crude = method(path, names=col_label, delimiter='|', usecols=col_use)
df = pd.DataFrame(data=crude)
model1 = keras.models.Sequential([

])
nn = np.array([1,2,3,4])
print("\t".join([str(x) for x in nn]) + "\n" + "hola")
"""
st = '(asdas) asda dfff'
print(st.split())
vava = st.split()
pp = vava.pop(0)
print(vava)
part = '(la   mansa vola)'
hola = 'hola'
# h = list(hola).append('')
# h[1] = ''
print(part.split().index('mansa'))
"""
"""
from sklearn.preprocessing import StandardScaler
def norm(array):
    n = StandardScaler()
    array = np.array(array).reshape(-1,1)
    print(array)
    array = n.fit_transform(array)
    print(n.mean_)
    array = array.reshape(-1)
    print(array)
    return array

arat = [1,2,3,4,5,-2,-7,-1]
print(norm(arat))
"""
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""
"""
curr = pl.Path(os.curdir)
root_logdir = os.path.join(curr, "pCM_logs")
sdk = pl.Path('./trash/fuck/caca/data')
tt = pl.Path('./trash/fuck')
root = sdk
rootpath = curr / root if root is not None else curr
print(rootpath)
print(sdk.parts)
print(sdk.parents[2] / sdk.name)
"""
"""
def get_run_logdir(model_path):
    curr = pl.Path(os.curdir)
    root_logdir = curr / 'trash'
    model_path = pl.Path(model_path)
    import time
    run_id = time.strftime(model_path.name+"_%Y-%m-%d_%H-%M-%S")
    for i in range(1, len(model_path.parents) + 1):
        parent = model_path.parents[len(model_path.parents)-(i)]
        print(parent)
        print(parent.exists())
        if parent.exists() or parent.is_dir():
            if parent.is_dir():
                continue
            else:
                newdir = root_logdir / parent
                newdir.mkdir()
        else:
            newdir = root_logdir / parent
            newdir.mkdir()
    return root_logdir / model_path / run_id
print(get_run_logdir('hola/que/tal'))
"""
def get_run_logdir(model_path):
    """Entrega un path donde crear datos de la sesion de TensorFlow.
    Típicamente mediante callback a TensorBoard.
    """
    curr = pl.Path(os.curdir)
    rootpath = curr / 'trash'
    model_path = pl.Path(model_path)
    import time
    run_id = time.strftime(model_path.name+"_%Y-%m-%d_%H-%M-%S")
    for part in model_path.parts:
        rootpath = rootpath / part
        rootpath.mkdir(exist_ok=True)
    return rootpath / run_id
# print(get_run_logdir('hola/que/tal'))
"""
zero = np.zeros((5,))
nozero = np.array([1,2,3,4,5])
embed_inputs = np.concatenate((zero, nozero), axis=0)
print(embed_inputs)
embed = keras.layers.Embedding(20, 10,
    mask_zero=True,
    embeddings_initializer=tf.keras.initializers.TruncatedNormal(),
    )
vecs = embed(embed_inputs, training=True)
print(vecs)
tf.random.set_seed(6)
noise = keras.layers.GaussianNoise(3)#, noise_shape=(4,3,10))
zeros = np.zeros((1,3,10))
data = np.arange(90).reshape(3, 3, 10).astype(np.float64)
data = np.concatenate((zeros, data), axis=0)
print(data,"\n", tf.reduce_sum(data))
outputs = noise(data, training=True)
print(outputs,"\n", tf.reduce_sum(outputs))

tf.random.set_seed(0)
layer = keras.layers.Dropout(.2, input_shape=(2,))
data = np.arange(10).reshape(5, 2).astype(np.float32)
print(data)
outputs = layer(data, training=True)
print(outputs)
"""
"""
# print(tt.samefile(sdk.parent))
import time
delta = time.time()
ot = pd.DataFrame({'a':['w','x','y','z'], 'b':[2,6,10,14],
    'c':[3,7,11,15], 'd':[4,8,12,16]}, index=pd.Index([-1,0,1,1]))
t = LookupFrame(ot, input_dtypes=np.object)
# print(t.keys)
# print(t['a'])
def gen():
    for i, rows in iter(ot.groupby(ot.index)):
        rows = rows.to_numpy()
        yield (i, rows[:,0]), rows[:,1:].astype(np.int64)

kk = tf.data.Dataset.from_generator(gen,
    output_types=((tf.int64, tf.string), tf.int64))
tuple = next(iter(kk))
(i, s), num = tuple
print(t['a'].lookup(s))
print(tuple)
print(next(kk.as_numpy_iterator()))
"""
"""
kk = kk.map(lambda x, y: x,y)
li1 = list(kk.as_numpy_iterator())
print(li1)
def asd(x):
    return x*10
kk = kk.interleave(lambda x, y: tf.data.Dataset.from_tensor_slices(10*y),
    num_parallel_calls=1,
    # block_length=3,
    cycle_length=8,
    )
li2 = list(kk.as_numpy_iterator())
delta = time.time() - delta
print(li2)
print(delta)
"""
"""
print([bool(i) for i in [1,1,76,0,0]])
t = tf.constant([1, 2, 3])
paddings = tf.constant([[0, 2]])
print(tf.pad(t, paddings))
"""
"""
#cat_columns = df.select_dtypes(include=np.object).columns.tolist()
#v = df[cat_columns]
#df = df[v.replace(v.apply(pd.Series.value_counts)).gt(150).all(1)]
# print(int(np.nan))
"""
"""
def force_dtype(value, dtype=int):
    try:
        value = dtype(value)
        return value
    except (ValueError, TypeError):
        if isinstance(value, str):
            values = [v for v in value.split()]
            allnum = all([v.isnumeric() for v in values])
            if allnum:
                return values
            else: pass
        else: pass
def force(DataFrame, dtype=int, dropna=False, any=True):
    DataFrame = DataFrame.applymap(lambda x: force_dtype(x, dtype=dtype))
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
            raise ValueError('Allow type inside array:', allow_types,
                'Input type:', type(v))
        elif int(v):
            v = int(v)
            if v > interval[1]:
                v = 70
            elif v < interval[0] and v >= 1:
                v *= 10
            else:
                continue
            sum+=v
            i += 1
    try:
        return round(sum/i, 1)
    except ZeroDivisionError:
        # print('Array is full of zeros')
        return None


ot = pd.DataFrame({'at':['a','m','b'], 'bt':['l','n','b'],
    2:['y','r','w'], 3:['g','g','g']})
# f = ot.groupby(by=lambda x: isinstance(x, int), axis=1)


class PowTwo:
    def __init__(self, max=0):
        self.max = max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.max:
            result = 2 ** self.n
            self.n += 1
            return result
        else:
            raise StopIteration

dd = tf.constant([[1,2,3],[3,2,1]])
# print(dd.get_shape())
# print(tf.reshape(dd, tf.TensorShape([3, len(dd)])))
if [None, None] is None:
    print("\n\nasdfasdfadfas\n\n")
else:
    print("\n\nMalafadfas\n\n")
def branching(array):
    total_leafs = 1
    for value in array[::-1]:
        total_leafs *= value
    return total_leafs

c = [1,2,3,4]
f = [-4,-3,-2,-1]
cc, ff = np.meshgrid(c, f)
print(cc)
print(ff)
from itertools import product
f = ['0_0', '0_1','0_2']
g = ['1_0','1_1','1_2']
print(product(f, g))

class asd(pd.DataFrame):
    def __init__(self, DataFrame, **kwargs):
        super().__init__(DataFrame, **kwargs)
        def ppp(self):
            print(self.index)

df = asd({'a':[1,2,3],'b':[3,2,1]}, index=pd.Index(['x','y','z']))

og = pd.DataFrame({'eg':[8,2,1,4], 'fg':[3.3,2.2,1.1,2],
    9:[4,5,9,9], 'hg':[2,7,7,2]}, index=pd.Index([1,1,2,3]))
a1 = tf.Variable([[1,2,3,4],[4,3,2,1]])
b1 = tf.Variable([[1,3,5,7]])
print(a1)
zzz = og.groupby(level=og.index.nlevels-1).__iter__()
for z in zzz:
    print(z)
memory = []
for i,val in og.iterrows():
    print('otheriter: ', val.tolist())
rr = og.iterrows()
jndex, data = next(rr)
print('datos: ', data.tolist())
course = tf.constant([data.tolist()])
b = True
for index, data in rr:
    # if b:
        # b = False
        # continue
    data = tf.constant([data.tolist()])
    print('dato:', data)
    if index == jndex:
        course = tf.concat([course, data], axis=0)
        jndex = index
    else:
        print('fulltensor: ', course)
        memory.append(course)
        course = tf.constant(data)
        jndex = index

print('holyfuck\n',memory)
print('original: ', og)
"""
# print(bool('a' is not None and isinstance('a', str)))
"""
def force_dtype(DataFrame, columns=None, index=None dtype=int):
    kick_index = []
    for j, column in enumerate(DataFrame):
        for i, value in enumerate(DataFrame[column]):
            try:
                new_value = dtype(value)
                DataFrame.iloc[i,j] = new_value
            except (ValueError, TypeError):
                if pd.isna(value):
                    DataFrame.iloc[i,j] = None
                DataFrame.iloc[i,j] = None
                kick_index.append(i)
        DataFrame[column] = DataFrame[column].convert_dtypes()
    DataFrame = DataFrame.drop(kick_index)
    return DataFrame
"""

"""
for i, value in enumerate(ot['c']):
    try:
        ot.loc[i] = int(value)
        print(value)
    except ValueError:
        ot = ot.drop(i)
ot['c'] = ot['c'].convert_dtypes()
"""
"""
l = ["3","2","1"]
mapper = dict(list(enumerate(l)))
print(mapper)
"""
"""
from shutil import move
otherpath = pl.Path("D:\Programillas/MachineLearning/grade-prediction_deeplearning/proyectoColegioMayor/archivos_mayor")
file = otherpath / "mayor_tobalaba" / "2008.txt"
parent = file.parent
put_in_trash = True
try:
    with open(file, 'r') as f:
        baddybad = pd.read_csv(f,
            sep='|',
            engine='python'
            )
except:
    trash = parent / f'trash_{parent.name}'
    if not trash.is_dir():
        trash.mkdir()
    if put_in_trash:
        p = trash / file.name
        print(p)
        move(file, p)
"""
"""
a = np.array([[1,2,0,0,0],[0,0,3,0,0],[6,2,0,0,9]])
s = sp.sparse.csr_matrix(a)
print(s.indices, s.data, s.nnz, s.indptr, s)

y = np.array([5, 0, 1, 0, 8, 0, 5])
print(y.shape)
y.shape += (1, )
y = tf.sparse.from_dense(y)
setattr(y,'values',-y.values)
y = tf.SparseTensor(y.indices, tf.pow(y.values,2), y.dense_shape)
print(y)
"""
"""
h = tf.constant([[
    [[1],[1],[1]],
    [[-1],[-1],[-1]],
]])
X = tf.constant([
    [-1,-2,-3],
    [1,2,3],
    [3,2,1]
])
f = tf.constant([
    [[1,1,1],[1,1,2],[1,1,3]],
    [[2,2,1],[2,2,2],[2,2,3]],
    [[3,3,1],[3,3,2],[3,3,3]],
    [[4,4,1],[4,4,2],[4,4,3]]
])
g = tf.constant([[5,5,5]])
#tf.reduce_sum(f, axis=2, keepdims=True) * tf.transpose(X)
#tf.reduce_sum(result, axis=1)#, perm=[0,2,1])
result = tf.matmul(X,f)
result2 = tf.reduce_sum(result, axis=2)
#print(result, result2)
v = tf.TensorShape((None,3)).as_list()[:-1]
"""
"""
cat_columns = df.select_dtypes(include=np.object).columns
print(cat_columns)
"""
"""
numeric_columns = train.select_dtypes(include=np.number).columns.tolist()
cat_columns = train.select_dtypes(include=np.object).columns.tolist()
"""
"""
def num_turn(grade, level, subj):
    n = np.array([])
    for frase in subj:
        frase = [word.strip() for word in frase.split()]
        new_n = grade[frase[0]] + level[frase[1]]
        n = np.append(n, new_n)
    return n
print(num_turn(grade, level, subj))

from sklearn.preprocessing import OneHotEncoder

o = OneHotEncoder()
a = np.array([[2,1,3],[5,9,2],[2,2,9],[2,2,0]])
b = np.array([['perro'],['gato']])
#print(a)
onehot1 = o.fit_transform(a)
print(onehot1.toarray())
onehot2 = o.fit_transform(b)
print(onehot2.toarray())
print(sum(a))

import pathlib as pl
import pandas as pd
path = pl.Path('data/notas.csv')
col_label = ['asignatura', 'curso', 'año', 'rut', 'fecha', 'nota']
col_use=['asignatura','nota','rut','fecha','curso','año']
crude = pd.read_csv(path, names=col_label, delimiter='|', usecols=col_use)
df = pd.DataFrame(data=crude)
print(df.iloc[:,-1])

vsd = 5
asd = 6
list = ['vsd', 'asd']
var = ['vsd', 'asd']
all_list = [f"list[{i}] == {v}".format() for i, v in enumerate(var)]
all_ = " | ".join(all_list)
print(eval(all_))
#df.drop(df.loc[exec(all_)])

cat_columns = df.select_dtypes(include=np.object)
for column in cat_columns:
    count_df = df[column].value_counts()
    var = count_df.loc[min_sample >= count_df].index.tolist()
    all_list = [f"(df[column] == {v})".format(v=v) for v in var]
    all_str = " | ".join(all_list)
    print(all_str)
    df.drop(df.loc[eval(all_str)].index)
-------------------------------------------------------------------------------
def reprop(x, new, old):
    return x*(1./old)*new

raw_df = raw_df.pivot_table(values='nota', index=['rut','asignatura'], columns=['fecha'], aggfunc=np.mean)
#raw_df = raw_df.apply(lambda x: reprop(x,10,7))
print(raw_df)
#piv_df['fecha'] = piv_df.reindex(piv_df['fecha'].apply(lambda x: x/unix_day))
#sub_df = raw_df.xs(key=2019, level=0, axis=1)

def delta_date(df, level=None):
    unix_day = 864000
    biweek = 15.
    lst = mono(df.columns, unix_day*biweek)#QUINCENAS
    return lst

def mono(array, step=1):
    i = array[-1] - array[0]
    new_i = i/step
    new_ilist = list(range(1, int(new_i+1)))
    return new_ilist

"""#Toma dataframe.index en unix time y lo lleva a unidades de tiempo arbitrario, compriminedo el contenido
#segun la media 'np.mean'
"""
def mapp(df, lapse=15, axis=0, unix_day=864000):
    if axis==0 or axis=='index':
        a = df.index
    elif axis==1 or axis=='columns':
        a = df.columns
    b = np.array(a.tolist())
    lapse = float(lapse)
    b -= b[0] #inicio en primer valor
    aux_index = []
    for i in b:
        n_bw = int(round(i)/(unix_day*lapse))
        aux_index.append(n_bw+1)

    dict_i = dict(zip(a, aux_index))
    df = df.rename(columns=dict_i)

    df = df.groupby(df.columns, axis=1).agg(np.mean)
    #df.groupby()
    return df, df.columns

asd, a = mapp(raw_df, axis=1, lapse=1)
asd.fillna(0, inplace=True)
asd = asd.apply(lambda x: round(x))
asd = asd.astype('int64')
asd = asd.to_datetime()
print(asd.tail(20))
print(a)
"""
