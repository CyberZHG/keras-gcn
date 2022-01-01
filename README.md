# Keras Graph Convolutional Network

Graph convolutional layers.

## Install

```bash
pip install keras-gcn
```

## Usage

### `GraphConv`

![](https://user-images.githubusercontent.com/853842/46645052-88d54f00-cbb5-11e8-9acb-70f4ae5ec654.png)

```python
from tensorflow import keras
from keras_gcn import GraphConv


DATA_DIM = 3

data_layer = keras.layers.Input(shape=(None, DATA_DIM))
edge_layer = keras.layers.Input(shape=(None, None))
conv_layer = GraphConv(
    units=32,
    step_num=1,
)([data_layer, edge_layer])
```

`step_num` is the maximum distance of two nodes that could be considered as neighbors. If `step_num` is greater than 1, then the inputs of edges must be 0-1 matrices.

### `GraphMaxPool` & `GraphAveragePool`

Pooling layers with the `step_num` argument.
