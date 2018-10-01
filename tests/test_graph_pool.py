import unittest
import os
import tempfile
import random
import keras
import numpy as np
from keras_gcn import GraphMaxPool, GraphAveragePool
from keras_gcn.layers import GraphPool


class TestGraphPool(unittest.TestCase):

    input_data = [
        [
            [0, 4, 8],
            [1, 5, 9],
            [2, 6, 1],
            [3, 7, 2],
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

    def test_max_pool(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphMaxPool(
            step_num=1,
            name='GraphMaxPool',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'GraphMaxPool': GraphMaxPool})
        model.summary()
        predicts = model.predict([self.input_data, self.input_edge])[0]
        expects = np.asarray([
            [2, 6, 9],
            [1, 5, 9],
            [2, 6, 8],
            [3, 7, 2],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)
        conv_layer = GraphMaxPool(
            step_num=2,
            name='GraphMaxPool',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        predicts = model.predict([self.input_data, self.input_edge])[0]
        expects = np.asarray([
            [2, 6, 9],
            [2, 6, 9],
            [2, 6, 9],
            [3, 7, 2],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)

    def test_average_pooling(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphAveragePool(
            step_num=1,
            name='GraphAveragePool',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'GraphAveragePool': GraphAveragePool})
        model.summary()
        predicts = model.predict([self.input_data, self.input_edge])[0]
        expects = np.asarray([
            [1, 5, 6],
            [0.5, 4.5, 8.5],
            [1, 5, 4.5],
            [3, 7, 2],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)
        conv_layer = GraphAveragePool(
            step_num=2,
            name='GraphAveragePool',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        predicts = model.predict([self.input_data, self.input_edge])[0]
        expects = np.asarray([
            [1, 5, 6],
            [1, 5, 6],
            [1, 5, 6],
            [3, 7, 2],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
            edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
            conv_layer = GraphPool(
                step_num=1,
                name='GraphPool',
            )([data_layer, edge_layer])
            model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
            model.compile(
                optimizer='adam',
                loss='mae',
                metrics=['mae'],
            )
            model.summary()
            model.predict([self.input_data, self.input_edge])
