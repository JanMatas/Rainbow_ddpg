import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, dense_layer_size, layer_norm, name):
        self.name = name
        self.dense_layer_size = dense_layer_size
        self.layer_norm = layer_norm

    @property
    def vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [
            var for var in self.trainable_vars if 'LayerNorm' not in var.name
        ]


class Actor(Model):
    def __init__(self,
                 nb_actions,
                 dense_layer_size,
                 layer_norm,
                 conv_size='small',
                 name='actor'):
        super(Actor, self).__init__(dense_layer_size, layer_norm, name=name)
        self.nb_actions = nb_actions
        self.conv_size = conv_size

    def __call__(self, obs, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            # Only use the convolutional layers if we are dealing with high-dimensional state.
            if len(obs.shape) > 2:
                normalizer_fn = tc.layers.layer_norm if self.layer_norm else None
                if self.conv_size == 'small':
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=2,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=2,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=2,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=2,
                        normalizer_fn=normalizer_fn)
                elif self.conv_size == 'large':
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(8, 8),
                        stride=4,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(4, 4),
                        stride=2,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=1,
                        normalizer_fn=normalizer_fn)
                    x = tc.layers.conv2d(
                        x,
                        32,
                        kernel_size=(3, 3),
                        stride=1,
                        normalizer_fn=normalizer_fn)
                else:
                    raise Exception('Unknown size')
                x = tf.layers.flatten(x)

            # Concatenate with auxiliary input (eg. joint angles) and create dense layers.
            x = tf.concat([x, aux], axis=-1)
            x = tf.layers.dense(x, self.dense_layer_size)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, self.dense_layer_size)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # Create auxiliary outputs - they help the network recognize import env features,
            # in our case the object position, target position and gripper position. Those
            # outputs are also useful for debugging.
            obj_dim = 3
            target_dim = 3
            gripper_pos_dim = 3
            x = tf.layers.dense(
                x, self.dense_layer_size + gripper_pos_dim + obj_dim + target_dim)
            x, object_conf, gripper, target = tf.split(
                x, [self.dense_layer_size, obj_dim, gripper_pos_dim, target_dim], 1)

            # Add two more dense layers.
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, self.dense_layer_size)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.layers.dense(
                x,
                self.nb_actions,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))

            # Finish with action output (the tanh function normalizes it to be in range [-1, 1]).
            pi = tf.nn.tanh(x)
        return pi, object_conf, gripper, target


class Critic(Model):
    def __init__(
            self,
            num_dense_layers,
            dense_layer_size,
            layer_norm,
            name='critic',
    ):
        super(Critic, self).__init__(
            dense_layer_size, layer_norm, name=name)
        self.num_dense_layers = num_dense_layers

    def __call__(self, state, action, aux, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # Create input layer - note the asymmetry with Actor - critic is fed the low dimensional state
            # while actor receives high dimensional observation.
            x = tf.concat([state, action, aux], axis=-1)

            # Initialize dense layers.
            for i in range(self.num_dense_layers):
                x = tf.layers.dense(x, self.dense_layer_size)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            # Output a scalar Q function.
            x = tf.layers.dense(
                x,
                1,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [
            var for var in self.trainable_vars if 'output' in var.name
        ]
        return output_vars
