import unittest
import os
import tempfile
import random
import keras
import numpy as np
from keras_gcn import GraphConv


class TestGraphConv(unittest.TestCase):

    def test_call(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            step_num=1,
            kernel_initializer='ones',
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model.summary()
        input_data = [
            [
                [0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [7, 7, 8],
            ]
        ]
        input_edge = [
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ]
        predicts = model.predict([input_data, input_edge])[0]
        expects = np.asarray([
            [27., 27.],
            [12., 12.],
            [18., 18.],
            [22., 22.],
        ])
        self.assertTrue(np.allclose(expects, predicts))

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
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'GraphConv': GraphConv})
        predicts = model.predict([input_data, input_edge])[0].tolist()
        expects = np.asarray([
            [27., 27.],
            [27., 27.],
            [27., 27.],
            [22., 22.],
        ])
        print(predicts)
        self.assertTrue(np.allclose(expects, predicts))
