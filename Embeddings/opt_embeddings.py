import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Model, callbacks, preprocessing
import psutil
from tensorflow import keras
import pathlib as pl
import time
import pandas as pd
import random
import os

def get_run_dir(model_path):
    rootpath = pl.Path(os.curdir)
    import time
    model_path = pl.Path(model_path) / time.strftime("%Y-%m-%d_%H-%M-%S")
    for part in model_path.parts:
        rootpath /= part
        rootpath.mkdir(exist_ok=True)
    return rootpath

path = pl.Path('data/archivos_mayor-cleanframe.csv')
print(psutil.virtual_memory())
with open(path) as file:
    df = pd.read_csv(file).iloc[:, 1:]
    names = ['sede', 'ano', 'curso', 'semestre', 'asignatura',
             'profesor-rut']
    cats = df.drop(columns=["nota"])
    # nested dict{names:{categories: _ }}
    cats = {column:{value:i+1 for i, value in enumerate(cats[column].unique())} for column in cats}
    df = df.set_index(names, drop=True).sort_index()
    df = df.groupby(level=names)
    courses = []
    maxnum = 0
    for tuple in iter(df):
        pairs = tuple[1].to_numpy()
        courses.append((np.array(tuple[0], dtype=np.object), pairs))
        maxnum = np.amax([maxnum, len(pairs[:,1])])
    maxnum += 1

print(cats["curso"])
total_data = len(courses)
print(maxnum)

test_prop = 0.2
test_index = int(test_prop * total_data)
train_data = courses[test_index:]
test_data = courses[:test_index]
val_prop = 0.2
val_index = int(len(train_data)*val_prop)
val_data = train_data[:val_index]
train_data = train_data[val_index:]

print(psutil.virtual_memory())

def transform(tuples):
    metadata, pairs = tuples
    meta_indices = {names:cats[names][metadata[i]] for i, names in enumerate(names)}
    ruts = pairs[:,0]
    ruts = np.array([cats["alumno-rut"][rut] for rut in ruts], dtype=np.int32)
    marks = {"nota": pairs[:,1]}
    meta_indices["alumno-rut"] = ruts
    return meta_indices, marks

def process_list(tuples_list):
    # Toma array de tuplas (meta-data, alumno-nota) y las convierte en el input rquerido
    # por el modelo ({names, alumno-rut}, {notas})
    inputs = {name:[] for name in names}
    inputs["alumno-rut"] = []
    marks = {"nota":[]}
    levels = {}
    for level in cats["curso"].keys():
        l = 8 if "media" in level else 0
        n = int(level.split()[1][0])
        levels[level] = l + n
    cats["curso"] = levels
    for metadata, pairs in tuples_list:
        meta_indices = {name: cats[name][metadata[i]] for i, name in enumerate(names)}
        ruts = pairs[:, 0]
        ruts = [cats["alumno-rut"][rut] for rut in ruts] # ruts llevados a tokens
        inputs["alumno-rut"].append(ruts)
        m = pairs[:, 1].astype(np.float32) / 70 # notas de un curso normalizadas
        marks["nota"].append(m)
        for name in names:
            inputs[name].append([meta_indices[name]])
    inputs["alumno-rut"] = preprocessing.sequence.pad_sequences(inputs["alumno-rut"], maxlen=maxnum, padding='post')
    marks["nota"] = preprocessing.sequence.pad_sequences(marks["nota"], maxlen=maxnum, padding='post', dtype='float32')
    inputs["asignaturas"] = [list(range(1, len(cats["asignatura"])+1)) for _ in range(len(tuples_list))]
    return inputs, marks

buffer = total_data + 1
batch_size = 512
tf.random.set_seed(1)

train_data = process_list(train_data)
train_data = tf.data.Dataset.from_tensor_slices(train_data)
train_data = train_data.shuffle(buffer)
train_data = train_data.batch(batch_size).prefetch(1)

val_data = process_list(val_data)
val_data = tf.data.Dataset.from_tensor_slices(val_data)
val_batch_size = int(batch_size*val_prop)
val_data = val_data.shuffle(buffer).batch(val_batch_size).prefetch(1)

print(psutil.virtual_memory())
###############################################################################
"""                                MODEL                                    """
###############################################################################
# Dimension del embedding para todos los inputs
embed_dim = 2 ** 6 #8 #7 cambio realizado el 20-09-09
# Embedding del un curso con un profesor en particular
student_noise = 0.2
prof_noise = 0.05
total_student = len(cats["alumno-rut"])
input_students = keras.layers.Input(shape=[maxnum],
    # batch_size=batch_size,
    name="alumno-rut")
students = keras.layers.Embedding(input_dim=total_student+1,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="alumno-rut_embeds")
students_tensor = students(input_students)
noise = keras.layers.GaussianNoise(stddev=student_noise)
students_tensor = noise(students_tensor)

# Embedding del nivel conjunto con el semestre, será una multiplicación de estos dos.
# Como los diccionarios de tensorflow parten de 0, se suma 1 al tensor
# para asegurar que dos tuplas; (nivel, semestre), no repita sus indices
total_levels = len(cats["curso"])
num_semesters = len(cats["semestre"])
input_level = keras.layers.Input(shape=[1],
                                 # batch_size=batch_size,
                                 name="curso")
input_semester = keras.layers.Input(shape=[1],
                                    # batch_size=batch_size,
                                    name="semestre")
combined = input_semester*input_level

semester = keras.layers.Embedding(input_dim=total_levels*num_semesters+1,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="level-semester_embed",
    )
semester_vec = semester(combined)

# Suma del embedding correspondientes a los alumnos y "semestre", respecticamente
# Notar que semester_vec es la suma del embed original semestre junto con embed
# nivel.
course_tensor =  students_tensor + semester_vec

# Embedding de todas las asignaturas existentes
total_subjects = len(cats["asignatura"])
input_subjects = keras.Input(shape=total_subjects,
    # batch_size=batch_size,
    name="asignaturas")
embed_subjects = keras.layers.Embedding(input_dim=total_subjects+1,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="asignaturas_embeds",
    )
subjects_tensor = embed_subjects(input_subjects)

# Vector de el profesor del curso
total_profesors = len(cats["profesor-rut"])
input_profesor = keras.Input(shape=[1],
    # batch_size=batch_size,
    name="profesor-rut"
    )
profesors = keras.layers.Embedding(input_dim=total_profesors+1,
    output_dim=embed_dim,
    mask_zero=True,
    embeddings_initializer=keras.initializers.GlorotNormal(),
    name="profesor-rut_embeds",
    )
profesor_vec = profesors(input_profesor)
noise2 = keras.layers.GaussianNoise(stddev=prof_noise)
profesor_vec = noise2(profesor_vec)

# Self-Atencion de cada curso, es decir de cada "alumno-vector" con todos los
# demas
# """fuera self-attention"""
a1 = keras.layers.Attention(use_scale=True)([course_tensor, course_tensor])

# Nuevamente Atencion por parte de cada "atencion del alumno-vector" hacia
# todas las asignaturas existentes
a2 = keras.layers.Attention(use_scale=True)([a1, subjects_tensor])

# Luego de consegir el vector de atención que le dedican los alumnos del curso,
# este se proyecta con el vector del para medir su influecia, dando la nota
# especifica del alumno en la asignatura correspondiente
output_grades = keras.layers.Dot(axes=2, name="nota")([a2, profesor_vec])
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
s_noise = "".join([num for num in str(student_noise) if num.isnumeric()])
p_noise = "".join([num for num in str(prof_noise) if num.isnumeric()])

run_logdir = get_run_dir(f"pCM_logs/seed_1/6bit-dim/noise_s{s_noise}-p{p_noise}/atte_scaled")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_logdir,
    write_images=True,
    histogram_freq=10,
    embeddings_freq=10,
    )
early_cb = keras.callbacks.EarlyStopping(patience=20,
                                         min_delta=0
)

def best_lr(history):
    lrs = np.array(history.history["lr"])
    losses = np.array(history.history["loss"])
    dloss = losses[1:] - losses[:-1]
    min_index = np.argmin(dloss)
    return lrs[min_index]
init_lr = 1e-10
final_lr = 1e-1
total_epochs = 1000
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: init_lr*(final_lr/init_lr)**(epoch/total_epochs)
)

# batch_size: 512
# best_lr =

encoder.compile(loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=["mae"],
    )
import time
init = time.time()
h = encoder.fit(train_data,
                epochs=total_epochs,
                validation_data=val_data,
                callbacks=[tensorboard_callback],
                use_multiprocessing=True,
                workers=tf.data.experimental.AUTOTUNE
                )

fin = time.time()
print("Tiempo entrenando: ", int((fin - init)//60), "m ", round((fin-init)%60, 1), "s")

from matplotlib import pyplot as plt
def plot_loss(type, history):
    [l] = plt.semilogx(history.history[type], label=type)
    [v] = plt.semilogx(history.history[f"val_{type}"], label=f"val_{type}")
    plt.legend(handles=[l, v], loc='upper right')
    plt.show()

plot_loss("loss", h)
plot_loss("mae", h)

#[ploss] = plt.semilogx(h.history["lr"], h.history["loss"], label="loss")
#[pmae] = plt.semilogx(h.history["lr"], h.history["val_loss"], label="val_loss")
#plt.legend(handles=[ploss, pmae], loc='upper right')
#plt.show()

#run_savedir = get_run_dir("records//glorot/5bit-dim/noise_s08-p04")
#encoder.save(run_savedir.as_posix()+'/subj_focus.h5')


