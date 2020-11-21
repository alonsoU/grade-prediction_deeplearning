import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow import keras

class PositionalEncoding(keras.layers.Layer):
    """Embedding posicional para
    """
    def __init__(self, max_steps, max_dims, dtype=tf.float64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1: max_dims += 1 # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] )
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, self.W__name__:self.W}

class FactorizationMachine(keras.layers.Layer):
    """Neurona con igual funcionamiento que Dense(), agregando
    un grado de relación entre inputs mediante la matriz 'M',
    la cual se encuentra factorizada en la matriz 'V' y su traspuesta.
    Generando asi un embedding dentro de la dimención 'k' de la matriz,
    donde cada fila de 'V' representa una variable categorica dentro de un
    input representado en onehotencoder.
        units: unidades de neuronas a diponer
        activation: activación en cada neurona.
        k: dimension del embedding para las variables categoricas.
        initializer: iniciador de los los pesos en cada tensor lineal.
    """
    def __init__(self, units, activation=None, k=40,
            initializer="glorot_normal",
            linear=True,
            **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.k = k
        self.initializer = initializer
        self.linear = linear
    def build(self, batch_input_shape):
        self.p =  batch_input_shape[-1]
        self.V = self.add_weight(shape=(self.units, self.p, self.k),
            initializer=self.initializer,
            trainable=True, name='factor_matrix')
        self.w = self.add_weight(shape=(self.p, self.units),
            initializer=self.initializer,
            trainable=True, name='linear')
        self.w0 = self.add_weight(shape=(self.units,),
            initializer='zeros',
            trainable=True, name='bias')
        super().build(batch_input_shape)
    def call(self, inputs):
        lin = self.w0 + inputs @ self.w
        factor_a = tf.transpose(tf.reduce_sum((inputs @ self.V)**2, axis=2))
        aux_V2 = tf.reduce_sum(self.V**2, axis=2)
        factor_b = inputs**2 @ tf.transpose(aux_V2)
        return self.activation(lin + 0.5*(factor_a - factor_b))
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, self.V.__name__:self.V, 'num_categories':self.p}

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
            keras.layers.Conv2D(filters, 1, strides=strides,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

class Proyection(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.columns = input_shape[-1]
        self.W = self.add_weight(initializer=self.initializer,
            trainable=True, name='proyection',
            shape=(self.columns, self.units),)
        super().build(input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.W)
        return self.activation(z)
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, self.W__name__:self.W}

class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
        shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

class MultiHeadAttention(keras.layers.Layer):
    # Multi-Head Attention. Tal como 'attention is all you need'
    # todos los vector; query, key, value son proyectados, es decir, transformados
    # linealmente, en las dimensiones (Tq,dim)->(Tq,d_scores) mediante
    # Wq: (dim,scores), (Tv,dim)->(Tv,d_value) mediante Wk,v: (dim, d_value).
    # Por lo que el output de cada Attention layer seria de dimension:
    # output (Tq,Tv) para cada pryección 'h'.
    # call(self, inputs):
        # inputs: es un array de tensores, ya sea 2 o 3, el primero siendo query,
            # luego value y key. Si inputs solo tiene 2 tensores, el utimo valor
            # es key-value.
    def __init__(self,
            h,
            d_scores=None,
            d_value=None,
            identity=False,
            trainable_proyection=True,
            use_bias=False,
            activation=None, **kwargs):
        super().__init__(**kwargs)
        self.h = h # Número de proyecciones
        self.d_scores = d_scores
        self.d_value = d_value
        self.trainable_proyection = trainable_proyection
        self.use_bias = use_bias
        self.proyections = list([])
        self.activation = keras.activations.get(activation)
    def build(self, input_shape):
        if len(input_shape)==2:
            self.Q = input_shape[0]
            self.V, self.K = input_shape[1]
        elif len(input_shape)==3:
            self.Q = input_shape[0]
            self.V = input_shape[1]
            self.K = input_shape[2]
        self.Tq = len(self.Q[-2])
        self.Tv = len(self.V[-2])
        self.dim = len(self.Q[-1])
        if self.d_scores == None:
            self.d_scores = self.dim//self.h
        if self.d_value == None:
            self.d_value = self.dim//self.h
        self.concat = keras.layers.Concatenate(axis=-1)
        self.reconstruction = keras.layers.Dense(units=self.dim,
            use_bias=self.use_bias,
            name='reconstruction',
            trainable=self.trainable_proyection,
            initializer='glorot_normal',
            )
        for i in range(self.h):
            Wq = keras.layers.Dense(units=self.d_scores,
                use_bias=self.use_bias,
                name=f'query_proyect_{i}',
                trainable=self.trainable_proyection,
                initializer='glorot_normal',
                )
            Wk = keras.layers.Dense(units=self.d_scores,
                use_bias=self.use_bias,
                name=f'key_proyect_{i}',
                trainable=self.trainable_proyection,
                initializer='glorot_normal',
                )
            Wv = keras.layers.Dense(units=self.d_value,
                use_bias=self.use_bias,
                name=f'value_proyect_{i}',
                trainable=self.trainable_proyection,
                initializer='glorot_normal',
                )
            self.proyections.append([Wq, Wv, Wk])
        super().build(input_shape)
    def call(self, inputs):
        concat_list = []
        for proyection in self.proyections:
            if len(inputs)==2:
                self.Q = proyection[0](inputs[0])
                self.V = proyection[1](inputs[1])
                self.K = proyection[2](inputs[1])
            elif len(inputs)==3:
                self.Q = proyection[0](inputs[0])
                self.V = proyection[1](inputs[1])
                self.K = proyection[2](inputs[2])
            concat_list.append(keras.layers.Attention(use_scale=True)([self.Q, self.V, self.K],
                    )
                )
        output = self.reconstruction(self.concat(concat_list))
        return output
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0,:-1], self.dim)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
"""
class MultiHeadAttention(keras.layers.Layer):
    'Layer que proyecta 'h' veces el scaled-dot-product-attention: Attention()
    lo concate y lo vuelve a proyectar, para obtener output_shape igual al
    input_shape.
    '
    def __init__(self, h=8):
        self.h = h
    def build(self, input_shape):
        self.d_inter = input_shape[-1]//self.h
        self.d_output = input_shape[-1]
        self.poyect = keras.layers.Dense()
        self.concat = keras.layers.Concatenate(axis=-1)
        self.W0 = self.add_weight(name='reconstructing_proyection',
            initializer='glorot_normal',
            shape=(self.d_inter * h, self.output))
    def call(self, inputs):
        differ_attention = []
        for i in range(h):
            att_i = Attention(d_k=self.d_inter, d_v=self.d_inter)(inputs)
            differ_attention.append(att_i)
        concat = self.concat(differ_attention)
        output = self.proyect(concat)
        return output
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.d_output])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
"""
