import tensorflow as tf
import tensorflow.contrib as tc

class Model(object):
    def __init__(self, num_dense_layers, dense_layer_size, layer_norm,name):
        self.name = name
        self.num_dense_layers = num_dense_layers
        self.dense_layer_size = dense_layer_size
        self.layer_norm = layer_norm

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, n_state,  num_dense_layers, dense_layer_size, layer_norm, conv_size='small',name='actor'):
        super(Actor, self).__init__( num_dense_layers, dense_layer_size, layer_norm,name=name)
        self.nb_actions = nb_actions
        self.n_state = n_state
        self.conv_size = conv_size


    def __call__(self, obs, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            if len(obs.shape) > 2:
                normalizer_fn = tc.layers.layer_norm if self.layer_norm else None                
                if self.conv_size == 'small':
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=2, normalizer_fn=normalizer_fn)                    
                elif self.conv_size == 'large':
                    x = tc.layers.conv2d(x, 32, kernel_size=(8, 8), stride=4, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(4, 4), stride=2, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=1, normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(x, 32, kernel_size=(3, 3), stride=1, normalizer_fn=normalizer_fn)
                else:
                    raise Exception('Unknow size')
                x = tf.layers.flatten(x)
            x = tf.concat([x, aux], axis=-1)

            x = tf.layers.dense(x, self.dense_layer_size)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.dense_layer_size)
            
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            obj_dim = 3
            target_dim = 3

            x = tf.layers.dense(x, self.dense_layer_size + 3 + obj_dim + target_dim)

            x, object_conf, gripper, target = tf.split(x, [self.dense_layer_size, obj_dim, 3, target_dim], 1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.dense_layer_size)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            pi = tf.nn.tanh(x)
        return pi, object_conf, gripper, target


class Critic(Model):
    def __init__(self,  num_dense_layers, dense_layer_size, layer_norm, name='critic',):
        super(Critic, self).__init__(num_dense_layers, dense_layer_size, layer_norm,name=name)

    def __call__(self, state, action, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat([state, action, aux], axis=-1)

            for i in range(self.num_dense_layers):
                x = tf.layers.dense(x, self.dense_layer_size)
                if self.layer_norm:

                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
