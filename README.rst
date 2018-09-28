
Keras Graph Convolutional Network
=================================


.. image:: https://travis-ci.org/CyberZHG/keras-gcn.svg
   :target: https://travis-ci.org/CyberZHG/keras-gcn
   :alt: Travis


.. image:: https://coveralls.io/repos/github/CyberZHG/keras-gcn/badge.svg?branch=master
   :target: https://coveralls.io/github/CyberZHG/keras-gcn
   :alt: Coverage


.. image:: https://img.shields.io/pypi/pyversions/keras-gcn.svg
   :target: https://pypi.org/project/keras-gcn/
   :alt: PyPI


Graph convolutional layers.

Install
-------

.. code-block:: bash

   pip install keras-gcn

Usage
-----

``GraphConv``
^^^^^^^^^^^^^^^^^


.. image:: https://user-images.githubusercontent.com/853842/46240178-19ea4000-c3d6-11e8-99e7-c2ce4c311e97.png
   :target: https://user-images.githubusercontent.com/853842/46240178-19ea4000-c3d6-11e8-99e7-c2ce4c311e97.png
   :alt: 


.. code-block:: python

   import keras
   from keras_gru import GraphConv


   DATA_DIM = 3

   data_layer = keras.layers.Input(shape=(None, DATA_DIM))
   edge_layer = keras.layers.Input(shape=(None, None))
   conv_layer = GraphConv(
       units=32,
       step_num=1,
   )([data_layer, edge_layer])

``step_num`` is the maximum distance of two nodes that could be considered as neighbors. If ``step_num`` is greater than 1, then the inputs of edges must be 0-1 matrices.

``GraphMaxPool`` & ``GraphAveragePool``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pooling layers with the ``step_num`` argument.
