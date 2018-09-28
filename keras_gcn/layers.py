import keras
import keras.backend as K


class GraphLayer(keras.layers.Layer):

    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())


class GraphConv(GraphLayer):

    def __init__(self,
                 units,
                 step_num=1,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.supports_masking = True
        self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'step_num': self.step_num,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_bias': self.use_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        features = K.dot(features, self.W)
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = K.map_fn(
            lambda x: self._call_single(x[0], x[1]),
            (features, edges),
            dtype=K.floatx(),
        )
        outputs = self.activation(outputs)
        return outputs

    def _call_single(self, feature, edge):
        return K.map_fn(
            lambda index: self._call_pos(feature, edge, index),
            K.arange(K.shape(feature)[0]),
            dtype=K.floatx(),
        )

    def _call_pos(self, feature, edge, index):
        if self.use_bias:
            feature += self.b
        return K.sum(feature * K.expand_dims(edge[index]), axis=0) / K.sum(edge[index])


class GraphPool(GraphLayer):

    def __init__(self,
                 step_num=1,
                 **kwargs):
        self.step_num = step_num
        self.supports_masking = True
        super(GraphPool, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'step_num': self.step_num,
        }
        base_config = super(GraphPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(GraphPool, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = K.map_fn(
            lambda x: self._call_single(x[0], x[1]),
            (features, edges),
            dtype=K.floatx(),
        )
        return outputs

    def _call_single(self, feature, edge):
        return K.map_fn(
            lambda index: self._call_pos(feature, edge, index),
            K.arange(K.shape(feature)[0]),
            dtype=K.floatx(),
        )

    def _call_pos(self, feature, edge, index):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphMaxPool(GraphPool):

    def _call_pos(self, feature, edge, index):
        return K.max(feature + K.expand_dims((1.0 - edge[index]) * -1e10), axis=0)


class GraphAveragePool(GraphPool):

    def _call_pos(self, feature, edge, index):
        return K.sum(feature * K.expand_dims(edge[index]), axis=0) / K.sum(edge[index])
