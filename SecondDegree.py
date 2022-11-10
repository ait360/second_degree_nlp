import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers



class SecondDegreeCell(Layer):
    def __init__(self, units, activation='sigmoid', use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',**kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for units, expected '
                             f'a positive integer, got {units}.')
        super(SecondDegreeCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units, self.units),
            name='kernel',
            initializer=self.kernel_initializer)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0] if tf.nest.is_nested(states) else states
        input_kernel = tf.einsum('ij,jbk->ibk', inputs, self.kernel)
        
        states = tf.einsum('bh, bhh->bh', prev_output, input_kernel)

        output = backend.bias_add(states, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        new_state = [output] if tf.nest.is_nested(states) else output

        return output, new_state

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),

        }







