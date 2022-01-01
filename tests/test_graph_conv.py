import unittest
import os
import tempfile

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from keras_gcn import GraphConv


class TestGraphConv(unittest.TestCase):

    input_data = np.array([
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [7, 7, 8],
        ]
    ], dtype=K.floatx())
    input_edge = np.array([
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ], dtype='int32')

    def test_average_step_1(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            step_num=1,
            kernel_initializer='ones',
            bias_initializer='ones',
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model.summary()
        predicts = model.predict([self.input_data, self.input_edge])[0]
        expects = np.asarray([
            [10., 10.],
            [7., 7.],
            [10., 10.],
            [23., 23.],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)

    def test_average_step_inf(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            step_num=60000000,
            kernel_initializer='ones',
            use_bias=False,
            bias_initializer='ones',
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'GraphConv': GraphConv})
        predicts = model.predict([self.input_data, self.input_edge])[0].tolist()
        expects = np.asarray([
            [9., 9.],
            [9., 9.],
            [9., 9.],
            [22., 22.],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)

    def test_fit(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'],
        )
        expects = np.asarray([[
            [9.5, 0.7],
            [6.5, 0.7],
            [9.5, 0.7],
            [22.8, 1.0],
        ]])
        model.fit(
            x=[self.input_data, self.input_edge],
            y=expects,
            epochs=10000,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            ],
            verbose=False,
        )
        predicts = model.predict([self.input_data, self.input_edge])
        self.assertTrue(np.allclose(expects, predicts, rtol=0.1, atol=0.1), predicts)
