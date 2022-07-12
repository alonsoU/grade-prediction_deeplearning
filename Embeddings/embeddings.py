import pandas as pd
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import classfolder.frameprocess as pre
from tensorflow import keras
import tensorflow as tf
import os

def get_run_dir(model_path):
    rootpath = pl.Path(os.curdir)
    import time
    model_path = pl.Path(model_path) / time.strftime("%Y-%m-%d_%H-%M-%S")
    for part in model_path.parts:
        rootpath /= part
        rootpath.mkdir(exist_ok=True)
    return rootpath
def split_label(DataFrame, label='nota', label_toframe=True):
    # Separa dataframe de su designado target o label
    if label_toframe:
        data_label = DataFrame.loc[:,label].to_frame()
    else:
        data_label = DataFrame.loc[:,label]
    data = DataFrame.drop([label], axis=1)
    return data, data_label
def batching_dataset(train, target, batch_size=32, prefetch=1):
    # Recive train data y target, lo trasforma en tf.data.Dataset,
    # lo junta en tuplas(requerido por Model.fit(x) al pasar objeto Dataset),
    # agrupa la secuencia en bachas y finalmente aplica prefetch eficiencia de
    # lectura.
    # batch_size: tamaño de la bacha.
    # prefetch: cantidad de datos al que se le aplica prefetch.
    train = tf.data.Dataset.from_tensor_slices(train)
    target = tf.data.Dataset.from_tensor_slices(target)
    data = tf.data.Dataset.zip((train, target))
    data = data.batch(batch_size).prefetch(prefetch)
    return data

path = pl.Path('D:/Proyectos/Python_proyects/MachineLearning/grade-prediction_deeplearning/Embeddings/data/archivos_mayor-cleanframe.csv')
df = pd.read_csv(path, delimiter=',', low_memory=False, index_col=0,)

# names: ['sede', 'ano', 'curso', 'asignatura', 'profesor-rut',
# 'semestre', 'alumno-rut', 'nota']
subjects = df['asignatura'].unique()
subj = df['asignatura'].value_counts().index.to_numpy()
# print(subjects)
# print(subj)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
def split_data(DataFrame, split=split, prop_column='asignatura'):
    # Separa indices de dataframe en train y test data, mediante objeto
    # scikit-learn, para luego construir un generador pero retorna diferentes
    # particiones del dataframe en cada interación.
    # split: instancia de sk-learn dedicada a generar indices en dataframe
    # prop_column: columna que se utiliza como mascara para la partición,
    #    correspondiendo la proporción de los elementos en esta.
    #
    for train_index, test_index in split.split(DataFrame, DataFrame[prop_column]):
        train_frame = DataFrame.iloc[train_index].copy()
        test_frame = DataFrame.iloc[test_index].copy()
        yield train_frame, test_frame

train_data, test_data = split_data(df).__next__()
train_data, validation_data = split_data(train_data).__next__()
test_data.to_csv('D:/Proyectos/Python_proyects/MachineLearning/grade-prediction_deeplearning/Embeddings/data/test_embeddings_re0.csv', index=False)
# train_data['nota'].hist(grid=True, bins=70)
# plt.show()
from sklearn.preprocessing import StandardScaler, PowerTransformer
def norm(array):
    """
    Aqui las notas pasaran a escala logaritmica. (experimental)
    """
    a = array
    # a /= 70
    # a = np.exp(a/70)
    standard = StandardScaler()
    # power = PowerTzransformer(method='yeo-johnson')
    a = np.array(a).reshape(-1, 1)
    # a = power.fit_transform(a)
    a = standard.fit_transform(a)
    a = a.reshape(-1)
    # a = np.log(a)
    return a

# Se separan las notas de cada sede, y a cada uno de los grupos se aplica
# StandardScaler de scikit para tener notas centradas en el origen con varianza 1
def sscaler_frame(data):
    mask = data['sede'] == 'colegio mayor tobalaba'
    data.loc[mask, 'nota'] = data.loc[mask, 'nota'].to_frame().apply(norm,
        axis=0, raw=True)
    data.loc[~mask, 'nota'] = data.loc[~mask, 'nota'].to_frame().apply(norm,
        axis=0, raw=True)
    return data

train_data = sscaler_frame(train_data)
validation_data = sscaler_frame(validation_data)
test_data = sscaler_frame(test_data)

print("Nota estandarizada máxima: ", train_data['nota'].max())
print("Nota estandarizada mínima: ", train_data['nota'].min())
train_data['nota'].hist(grid=True, bins=140)
# plt.show()

# Iterator que entrega grupos, en este caso de curso por asignatura
multi_label = ['sede', 'ano', 'curso', 'semestre', 'asignatura',
    'profesor-rut']
def indexed_frame(data):
    data_indexed = data.set_index(multi_label, drop=True).sort_index()
    return data_indexed

train_data_indx = indexed_frame(train_data)
names = train_data_indx.index.names
validation_data_indx = indexed_frame(validation_data)
test_data_indx = indexed_frame(test_data)#^2+145 #WHY?

# Tablas de variables categoricas basado en clase de indexado tensorflow
tf_tables = pre.LookupFrame(df,
    ['asignatura', 'curso', 'ano', 'semestre', 'alumno-rut', 'profesor-rut',],
    prop_bucket=0.01,
    )

# Numero naximo de alumnos en un curso o clase de profesor en particular
maxnum = 0
minnum = 20
numdata = 0
for index, data in train_data_indx.groupby(level=names).__iter__():
    curr_len = data.shape[0]
    maxnum = np.amax([curr_len, maxnum])
    minnum = np.amin([curr_len, minnum])
    numdata+=1
maxnum += 5
print("\nNumero total de datasets: ", numdata)
print("\nCurso con la menor cantidad: ", minnum)
print()

def name_toindex(names, guide=multi_label):
    if isinstance(names, list):
        in_list = []
        for name in names:
            i = guide.index(name)
            in_list.append(i)
        return in_list
    elif isinstance(names, str):
        i = guide.index(name)
        return i
    else:
        raise TypeError(f'name must be {repr(list)} or {repr(str)}')

train_groups = train_data_indx.groupby(level=names)


def train_data_gen(groups=train_groups):
    while True:
        for index, frame in iter(groups):
            matrix = frame.to_numpy()
            students = matrix[:,0]
            grades = matrix[:,1].astype(np.float64)
            index = np.array([str(i) for i in index])
            yield (index, students), grades


train_dataset = tf.data.Dataset.from_generator(train_data_gen,
    output_types=((tf.string, tf.string), tf.float64),
    output_shapes=((tf.TensorShape([len(multi_label)]),
        tf.TensorShape([None])),
        tf.TensorShape([None])),
    )

match = name_toindex(['curso', 'semestre', 'profesor-rut'])
@tf.function
def transform_data(X, y):
    global match
    index, _students = X
    grades = y
    students = tf_tables['alumno-rut'].lookup(_students)

    # Busca el index que le corresponde al nombre en la lista de
    # Multilabel
    level = tf_tables['curso'].lookup(index[match[0]])
    level = tf.reshape(level, [1])
    semester = tf_tables['semestre'].lookup(index[match[1]])
    semester = tf.reshape(semester, [1])
    profesor =  tf_tables['profesor-rut'].lookup(index[match[2]])
    profesor = tf.reshape(profesor, [1])

    _subjects = tf.constant(subjects, dtype=tf.string)
    all_subjects = tf_tables['asignatura'].lookup(_subjects)
    all_subjects = tf.reshape(all_subjects, [len(subjects)])

    return {'input_students': students,
        'input_level': level,
        'input_semester': semester,
        'input_profesor': profesor,
        'input_subjects': all_subjects,
        }, {'output_grades': grades}
buffer = numdata + 1
train_dataset = train_dataset.shuffle(buffer)
train_dataset = train_dataset.map(lambda x, y: transform_data(x, y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # deterministic=False
    )
#%%
batch_size = 64
steps_per_epoch = numdata//batch_size
def batching(dataset):
    dataset = dataset.padded_batch(batch_size,
        padded_shapes=({'input_students': maxnum,
            'input_level': 1,
            'input_semester': 1,
            'input_profesor': 1,
            'input_subjects': len(subjects),
            }, {'output_grades': maxnum}),
        # padded_values=0,
        ).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = batching(train_dataset)
# run_cachedir = get_run_dir('atte_default_train', root='caches')
# train_dataset = train_dataset.cache(run_cachedir)

###############################################################################
"""                                MODEL                                    """
###############################################################################

# Dimension del embedding para todos los inputs
embed_dim = 2 ** 5 #8 #7 cambio realizado el 20-09-09
# Embedding del un curso con un profesor en particular
total_student = tf_tables.get_lenght('alumno-rut')
input_students = keras.layers.Input(shape=[maxnum],
    batch_size=batch_size,
    name="input_students",
    # ragged=True,
    )
students = keras.layers.Embedding(input_dim=total_student,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="alumno-rut",
    )
students_tensor = students(input_students)
noise = keras.layers.GaussianNoise(stddev=0.8)
students_tensor = noise(students_tensor)

# Embedding del nivel conjunto con el semestre, será una multiplicación de estos dos.
# Como los diccionarios de tensorflow parten de 0, se suma 1 al tensor
# para asegurar que dos tuplas; (nivel, semestre), no repita sus indices
total_levels = tf_tables.get_lenght('ano')
num_semesters = tf_tables.get_lenght('semestre')
input_level = keras.layers.Input(shape=[1],
    name="input_level")
input_semester = keras.layers.Input(shape=[1],
    name="input_semester")
combined = input_semester*input_level

semester = keras.layers.Embedding(input_dim=total_levels*num_semesters,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="embed_level-semester",
    )
semester_vec = semester(combined)

# Suma del embedding correspondientes a los alumnos y "semestre", respecticamente
# Notar que semester_vec es la suma del embed original semestre junto con embed
# nivel.
course_tensor =  students_tensor + semester_vec

# Embedding de todas las asignaturas existentes
total_subjects = tf_tables.get_lenght('asignatura')
input_subjects = keras.Input(shape=[len(subjects)],
    name="input_subjects")
embed_subjects = keras.layers.Embedding(input_dim=total_subjects,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="asignatura",
    )
subjects_tensor = embed_subjects(input_subjects)

# Vector de el profesor del curso
total_profesors = tf_tables.get_lenght('profesor-rut')
input_profesor = keras.Input(shape=[1],
    name="input_profesor"
    )
profesors = keras.layers.Embedding(input_dim=total_profesors,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="profesor-rut",
    )
profesor_vec = profesors(input_profesor)
noise2 = keras.layers.GaussianNoise(stddev=0.4)
profesor_vec = noise2(profesor_vec)

# Self-Atencion de cada curso, es decir de cada "alumno-vector" con todos los
# demas
# """fuera self-attention"""
a1 = keras.layers.Attention(use_scale=True)([course_tensor, course_tensor])
# a1 = course_tensor

# Nuevamente Atencion por parte de cada "atencion del alumno-vector" hacia
# todas las asignaturas existentes
a2 = keras.layers.Attention(use_scale=True)([a1, subjects_tensor])

# Luego de consegir el vector de atención que le dedican los alumnos del curso,
# este se proyecta con el vector del para medir su influecia, dando la nota
# especifica del alumno en la asignatura correspondiente
output_grades = keras.layers.Dot(axes=2, name="output_grades")([a2, profesor_vec])
###############################################################################
"""                                  END                                    """
###############################################################################
encoder = keras.Model(inputs=[input_students,
        input_level,
        input_semester,
        input_subjects,
        input_profesor
        ],
    outputs=[output_grades])
encoder.summary()

run_logdir = get_run_dir("pCM_logs/clean_subjects/glorot/5bit-dim/noise_s08-p04/subj_focus.h5")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_logdir,
    write_images=True,
    histogram_freq=5,
    # embeddings_freq=5,
    )
early_cb = keras.callbacks.EarlyStopping(patience=35,
    restore_best_weights=True)
def best_lr(history):
    lrs = np.array(history.history["lr"])
    losses = np.array(history.history["loss"])
    dloss = losses[1:] - losses[:-1]
    min_index = np.argmin(dloss)
    return lrs[min_index]
init_lr = 1e-10
final_lr = 1e-2
total_epochs = 100
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: init_lr*(final_lr/init_lr)**(epoch/total_epochs)
)

encoder.compile(loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=0.00012),
    metrics=['mean_absolute_error'],
    )

test_groups = test_data_indx.groupby(level=names)
def test_data_gen(groups=test_groups):
        for index, frame in iter(groups):
            matrix = frame.to_numpy()
            students = matrix[:,0]
            grades = matrix[:,1].astype(np.float64)
            index = np.array([str(i) for i in index])
            yield (index, students), grades
test_dataset = tf.data.Dataset.from_generator(test_data_gen,
    output_types=((tf.string, tf.string), tf.float64),
    output_shapes=((tf.TensorShape([len(multi_label)]),
        tf.TensorShape([None])),
        tf.TensorShape([None])),
    )
test_dataset = test_dataset.map(map_func=lambda x, y: transform_data(x, y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # deterministic=False
    )
test_dataset = batching(test_dataset)
test_steps = int((numdata*0.2/0.8**2)/batch_size)
loss_untrained, metrics_untrained = encoder.evaluate(test_dataset, steps=test_steps)

val_groups = validation_data_indx.groupby(level=names)
def val_data_gen(groups=val_groups):
    while True:
        for index, frame in iter(groups):
            matrix = frame.to_numpy()
            students = matrix[:,0]
            grades = matrix[:,1].astype(np.float64)
            index = np.array([str(i) for i in index])
            yield (index, students), grades
val_dataset = tf.data.Dataset.from_generator(val_data_gen,
    output_types=((tf.string, tf.string), tf.float64),
    output_shapes=((tf.TensorShape([len(multi_label)]),
        tf.TensorShape([None])),
        tf.TensorShape([None])),
    )
val_dataset = val_dataset.map(lambda x, y: transform_data(x, y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # deterministic=False
    )
val_dataset = batching(val_dataset)
val_steps = int((numdata*0.2/0.8)/batch_size)

history= encoder.fit(train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    callbacks=[tensorboard_callback, early_cb, lr_schedule],
    use_multiprocessing=True,
    workers=tf.data.experimental.AUTOTUNE,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps
    )
print(best_lr(history))
# batch_size: 32
# best_lr = 0.00012022645
plt.semilogx(history.history["loss"], history.history["lr"])
run_savedir = get_run_dir("records/clean_subjects/glorot/5bit-dim/noise_s08-p04")
encoder.save(run_savedir.as_posix()+'/subj_focus.h5')

loss_trained, metrics_trained = encoder.evaluate(test_dataset, steps=test_steps)
