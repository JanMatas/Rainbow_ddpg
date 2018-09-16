import resource
from copy import copy
from functools import reduce
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_running_mean_std import RunningMeanStd
import os
import sys

tmp = os.path.dirname(sys.modules['__main__'].__file__) + "/tmp"

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(
            tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


class DDPG(object):
    def __init__(self,
                 actor,
                 critic,
                 memory,
                 observation_shape,
                 action_shape,
                 state_shape,
                 aux_shape,
                 lambda_obj_conf_predict,
                 lambda_gripper_predict,
                 lambda_target_predict,
                 action_noise=None,
                 gamma=0.99,
                 tau=0.001,
                 enable_popart=False,
                 normalize_observations=True,
                 normalize_state=True,
                 normalize_aux=True,
                 batch_size=128,
                 observation_range=(-10., 10.),
                 action_range=(-1., 1.),
                 state_range=(-4, 4),
                 return_range=(-250, 10),
                 aux_range=(-10, 10),
                 critic_l2_reg=0.001,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 clip_norm=None,
                 reward_scale=1.,
                 replay_beta=0.4,
                 lambda_1step=1.0,
                 lambda_nstep=1.0,
                 nsteps=10,
                 run_name="unnamed_run",
                 lambda_pretrain=0.0,
                 target_policy_noise=0.2,
                 target_policy_noise_clip=0.5,
                 policy_and_target_update_period=2,
                 num_critics=2,
                 **kwargs):

        # Inputs.
        self.obs0 = tf.placeholder(
            tf.float32, shape=(None, ) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(
            tf.float32, shape=(None, ) + observation_shape, name='obs1')

        self.state0 = tf.placeholder(
            tf.float32, shape=(None, ) + state_shape, name='state0')
        self.state1 = tf.placeholder(
            tf.float32, shape=(None, ) + state_shape, name='state1')

        self.terminals1 = tf.placeholder(
            tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(
            tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(
            tf.float32, shape=(None, ) + action_shape, name='actions')
        self.critic_target = tf.placeholder(
            tf.float32, shape=(None, 1), name='critic_target')

        self.nstep_steps = tf.placeholder(
            tf.float32, shape=(None, 1), name='nstep_reached')
        self.nstep_critic_target = tf.placeholder(
            tf.float32, shape=(None, 1), name='nstep_critic_target')
        self.memory_size = tf.placeholder(
            tf.float32, shape=None, name='memory_size')
        self.rss = tf.placeholder(tf.float32, shape=None, name='rss')

        self.aux0 = tf.placeholder(
            tf.float32, shape=(None, ) + aux_shape, name='aux0')
        self.aux1 = tf.placeholder(
            tf.float32, shape=(None, ) + aux_shape, name='aux1')

        self.pretraining_tf = tf.placeholder(
            tf.float32, shape=(None, 1),
            name='pretraining_tf')  # whether we use pre training or not

        self.aux_shape = aux_shape
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_state = normalize_state
        self.normalize_aux = normalize_aux
        self.action_noise = action_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor = actor
        self.actor_lr = actor_lr
        self.state_range = state_range
        self.aux_range = aux_range
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.lambda_nstep = lambda_nstep
        self.lambda_1step = lambda_1step
        self.lambda_obj_conf_predict = lambda_obj_conf_predict
        self.lambda_gripper_predict = lambda_gripper_predict
        self.lambda_target_predict = lambda_target_predict
        self.nsteps = nsteps
        self.replay_beta = replay_beta
        self.run_name = run_name
        self.lambda_pretrain = lambda_pretrain
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.ep = 0
        self.policy_and_target_update_period = policy_and_target_update_period

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        if self.normalize_state:
            with tf.variable_scope('state_rms'):
                self.state_rms = RunningMeanStd(shape=state_shape)
        else:
            self.state_rms = None

        if self.normalize_aux:
            with tf.variable_scope('normalize_aux'):
                self.aux_rms = RunningMeanStd(shape=aux_shape)
        else:
            self.aux_rms = None

        with tf.name_scope('obs_preprocess'):
            self.normalized_obs0 = tf.clip_by_value(
                normalize(self.obs0, self.obs_rms), self.observation_range[0],
                self.observation_range[1])
            self.normalized_obs1 = tf.clip_by_value(
                normalize(self.obs1, self.obs_rms), self.observation_range[0],
                self.observation_range[1])
        with tf.name_scope('state_preprocess'):
            self.normalized_state0 = tf.clip_by_value(
                normalize(self.state0, self.state_rms), self.state_range[0],
                self.state_range[1])
            self.normalized_state1 = tf.clip_by_value(
                normalize(self.state1, self.state_rms), self.state_range[0],
                self.state_range[1])

        with tf.name_scope('aux_preprocess'):
            self.normalized_aux0 = tf.clip_by_value(
                normalize(self.aux0, self.aux_rms), self.aux_range[0],
                self.aux_range[1])
            self.normalized_aux1 = tf.clip_by_value(
                normalize(self.aux1, self.aux_rms), self.aux_range[0],
                self.aux_range[1])

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor

        self.actor_tf, self.obj_conf, self.gripper, self.target = actor(
            self.normalized_obs0, self.normalized_aux0)
        next_actions, _, _, _ = target_actor(self.normalized_obs1,
                                             self.normalized_aux1)
        noise = tf.distributions.Normal(
            tf.zeros_like(next_actions), self.target_policy_noise).sample()
        noise = tf.clip_by_value(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip,
        )

        self.num_critics = num_critics
        self.critics = [None] * num_critics
        self.target_critics = [None] * num_critics
        self.critic_tfs = [None] * num_critics
        self.critic_with_actor_tfs = [None] * num_critics
        self.step_1_td_losses = [None] * num_critics
        self.n_step_td_losses = [None] * num_critics
        self.td_errors = [None] * num_critics
        self.critic_losses = [None] * num_critics
        self.critic_grads = [None] * num_critics
        self.critic_optimizers = [None] * num_critics
        Q_obs1s = [None] * num_critics
        for i in range(num_critics):
            current_critic = copy(critic)
            current_critic.name = "critic" + str(i)
            self.critics[i] = current_critic
            self.target_critics[i] = copy(current_critic)
            self.target_critics[i].name = 'target_critic' + str(i)
            self.critic_tfs[i] = tf.clip_by_value(
                current_critic(self.normalized_state0, self.actions,
                               self.normalized_aux0), self.return_range[0],
                self.return_range[1])
            self.critic_with_actor_tfs[i] = tf.clip_by_value(
                current_critic(
                    self.normalized_state0,
                    self.actor_tf,
                    self.normalized_aux0,
                    reuse=True), self.return_range[0], self.return_range[1])
            Q_obs1s[i] = self.target_critics[i](self.normalized_state1,
                                                next_actions + noise,
                                                self.normalized_aux1)

        if num_critics == 2:
            minQ = tf.minimum(Q_obs1s[0], Q_obs1s[1])
        else:
            minQ = Q_obs1s[0]

        self.target_Q = self.rewards + \
            (1. - self.terminals1) * tf.pow(gamma, self.nstep_steps) * minQ
        self.importance_weights = tf.placeholder(
            tf.float32, shape=(None, 1), name='importance_weights')
        self.setup_actor_optimizer()
        self.setup_stats()
        self.setup_target_network_updates()
        for i in range(num_critics):
            self.setup_critic_optimizer(i)
        self.setup_summaries()

    def setup_target_network_updates(self):
        with tf.name_scope('target_net_updates'):
            actor_init_updates, actor_soft_updates = get_target_updates(
                self.actor.vars, self.target_actor.vars, self.tau)
            target_init_updates = [actor_init_updates]
            target_soft_updates = [actor_soft_updates]
            for i in range(self.num_critics):
                init, soft = get_target_updates(self.critics[i].vars,
                                                self.target_critics[i].vars,
                                                self.tau)
                target_init_updates.append(init)
                target_soft_updates.append(soft)
            self.target_init_updates = target_init_updates
            self.target_soft_updates = target_soft_updates

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        with tf.name_scope('actor_optimizer'):

            self.action_diffs = tf.reduce_mean(
                tf.square(self.actions - self.actor_tf), 1)
            demo_better_than_critic = self.critic_tfs[
                0] < self.critic_with_actor_tfs[0]
            demo_better_than_critic = self.pretraining_tf * \
                tf.cast(demo_better_than_critic, tf.float32)
            self.bc_loss = (
                tf.reduce_sum(demo_better_than_critic * self.action_diffs) *
                self.lambda_pretrain /
                (tf.reduce_sum(self.pretraining_tf) + 1e-6))
            self.original_actor_loss = - tf.reduce_mean(self.critic_with_actor_tfs[0])

            self.obj_conf_loss = tf.reduce_mean(
                tf.square(self.obj_conf -
                          self.state0[:, 8:11])) * self.lambda_obj_conf_predict
            self.gripper_loss = tf.reduce_mean(
                tf.square(self.gripper -
                          self.state0[:, 0:3])) * self.lambda_gripper_predict
            self.target_loss = tf.reduce_mean(
                tf.square(self.target -
                          self.state0[:, 3:6])) * self.lambda_target_predict

            self.actor_loss = self.original_actor_loss + self.bc_loss + \
                self.obj_conf_loss + self.gripper_loss + self.target_loss
            self.number_of_demos_better = tf.reduce_sum(
                demo_better_than_critic)
            actor_shapes = [
                var.get_shape().as_list() for var in self.actor.trainable_vars
            ]
            actor_nb_params = sum(
                [reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
            self.actor_grads = U.flatgrad(
                self.actor_loss,
                self.actor.trainable_vars,
                clip_norm=self.clip_norm)
            self.actor_optimizer = MpiAdam(
                var_list=self.actor.trainable_vars,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08)

    def setup_critic_optimizer(self, i):
        with tf.name_scope('critic_optimizer' + str(i)):
            critic_target_tf = tf.clip_by_value(
                self.critic_target, self.return_range[0], self.return_range[1])

            nstep_critic_target_tf = tf.clip_by_value(self.nstep_critic_target,
                                                      self.return_range[0],
                                                      self.return_range[1])

            td_error = tf.square(self.critic_tfs[i] - critic_target_tf)

            self.step_1_td_losses[i] = tf.reduce_mean(
                self.importance_weights * td_error) * self.lambda_1step

            nstep_td_error = tf.square(self.critic_tfs[i] -
                                       nstep_critic_target_tf)

            self.n_step_td_losses[i] = tf.reduce_mean(
                self.importance_weights * nstep_td_error) * self.lambda_nstep

            self.td_errors[i] = td_error + nstep_td_error
            self.critic_losses[i] = self.step_1_td_losses[i] + \
                self.n_step_td_losses[i]

            if self.critic_l2_reg > 0.:
                critic_reg_vars = [
                    var for var in self.critics[i].trainable_vars
                    if 'kernel' in var.name and 'output' not in var.name
                ]
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(
                    self.critic_l2_reg))
                critic_reg = tc.layers.apply_regularization(
                    tc.layers.l2_regularizer(self.critic_l2_reg),
                    weights_list=critic_reg_vars)
                self.critic_losses[i] += critic_reg
            critic_shapes = [
                var.get_shape().as_list()
                for var in self.critics[i].trainable_vars
            ]
            critic_nb_params = sum(
                [reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
            self.critic_grads[i] = U.flatgrad(
                self.critic_losses[i],
                self.critics[i].trainable_vars,
                clip_norm=self.clip_norm)
            self.critic_optimizers[i] = MpiAdam(
                var_list=self.critics[i].trainable_vars,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08)

    def setup_summaries(self):
        tf.summary.scalar("actor_loss", self.actor_loss)
        for i in range(self.num_critics):
            name_sufffix = str(i)
            tf.summary.scalar("critic_loss" + name_sufffix,
                              self.critic_losses[i])
            tf.summary.scalar("1step_loss" + name_sufffix,
                              self.step_1_td_losses[i])
            tf.summary.scalar("nstep_loss" + name_sufffix,
                              self.n_step_td_losses[i])

        tf.summary.scalar("percentage_of_demonstrations",
                          tf.reduce_sum(self.pretraining_tf) / self.batch_size)
        tf.summary.scalar("number_of_demos_better_than_actor",
                          self.number_of_demos_better)
        tf.summary.histogram("pretrain_samples", self.pretraining_tf)
        tf.summary.scalar("bc_loss", self.bc_loss)
        tf.summary.scalar("obj_conf_loss", self.obj_conf_loss)
        tf.summary.scalar("target_loss", self.target_loss)
        tf.summary.scalar("gripper_loss", self.gripper_loss)
        tf.summary.scalar("original_actor_loss", self.original_actor_loss)
        tf.summary.scalar("memory_size", self.memory_size)
        tf.summary.scalar("rss", self.rss)
        self.scalar_summaries = tf.summary.merge_all()
        # reward
        self.r_plot_in = tf.placeholder(tf.float32, name='r_plot_in')
        self.r_plot = tf.summary.scalar("returns", self.r_plot_in)
        self.r_plot_in_eval = tf.placeholder(tf.float32, name='r_plot_in_eval')
        self.r_plot_eval = tf.summary.scalar("returns_eval",
                                             self.r_plot_in_eval)

        self.obj_conf_in_eval = tf.placeholder(
            tf.float32, name='obj_conf_in_eval')
        self.obj_conf_eval = tf.summary.scalar("obj_conf_eval",
                                               self.obj_conf_in_eval)

        self.grip_in_eval = tf.placeholder(tf.float32, name='grip_in_eval')
        self.grip_eval = tf.summary.scalar("grip_eval", self.grip_in_eval)

        self.target_in_eval = tf.placeholder(tf.float32, name='target_in_eval')
        self.target_eval = tf.summary.scalar("target_eval",
                                             self.target_in_eval)

        self.writer = tf.summary.FileWriter(
            tmp + '/summaries/' + self.run_name, graph=tf.get_default_graph())

    def save_reward(self, r):
        self.ep += 1
        summary = self.sess.run(self.r_plot, feed_dict={self.r_plot_in: r})
        self.writer.add_summary(summary, self.ep)

    def save_aux_prediction(self, obj_conf, grip, target):
        self.ep += 1
        obj_conf_summ, grip_summ, target_summ = self.sess.run(
            [self.obj_conf_eval, self.grip_eval, self.target_eval],
            feed_dict={
                self.obj_conf_in_eval: obj_conf,
                self.grip_in_eval: grip,
                self.target_in_eval: target
            })
        self.writer.add_summary(obj_conf_summ, self.ep)
        self.writer.add_summary(grip_summ, self.ep)
        self.writer.add_summary(target_summ, self.ep)

    def save_eval_reward(self, r, ep):
        summary = self.sess.run(
            self.r_plot_eval, feed_dict={self.r_plot_in_eval: r})
        self.writer.add_summary(summary, ep)

    def setup_stats(self):
        with tf.name_scope('stats'):
            ops = []
            names = []

            if self.normalize_observations:
                ops += [
                    tf.reduce_mean(self.obs_rms.mean),
                    tf.reduce_mean(self.obs_rms.std)
                ]
                names += ['obs_rms_mean', 'obs_rms_std']

            ops += [tf.reduce_mean(self.critic_tfs[0])]
            names += ['reference_Q_mean']
            ops += [reduce_std(self.critic_tfs[0])]
            names += ['reference_Q_std']

            ops += [tf.reduce_mean(self.critic_with_actor_tfs[0])]
            names += ['reference_actor_Q_mean']
            ops += [reduce_std(self.critic_with_actor_tfs[0])]
            names += ['reference_actor_Q_std']

            ops += [tf.reduce_mean(self.actor_tf)]
            names += ['reference_action_mean']
            ops += [reduce_std(self.actor_tf)]
            names += ['reference_action_std']

            self.stats_ops = ops
            self.stats_names = names

    def pi(self, obs, aux, state0, apply_noise=True, compute_Q=True):

        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs], self.aux0: [aux], self.state0: [state0]}
        if compute_Q:

            action, q, obj_conf, gripper, target = self.sess.run(
                [
                    actor_tf, self.critic_with_actor_tfs[0], self.obj_conf,
                    self.gripper, self.target
                ],
                feed_dict=feed_dict)

        else:
            action, obj_conf, gripper, target = self.sess.run(
                [actor_tf, self.obj_conf, self.gripper, self.target],
                feed_dict=feed_dict)
            q = None
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise

        action = np.clip(action, self.action_range[0], self.action_range[1])

        return action, q, obj_conf, gripper, target

    def store_transition(self,
                         state,
                         obs0,
                         action,
                         reward,
                         state1,
                         obs1,
                         terminal1,
                         aux0,
                         aux1,
                         i,
                         demo=False):
        reward *= self.reward_scale
        if demo:
            self.memory.append_demonstration(state, obs0, action, reward,
                                             state1, obs1, terminal1, aux0, aux1, i)
        else:
            assert i is None
            self.memory.append(state, obs0, action, reward, state1, obs1,
                               terminal1, aux0, aux1, i)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

        if self.normalize_state:
            self.state_rms.update(np.array([state]))

        if self.normalize_aux:
            self.aux_rms.update(np.array([aux0]))

    def train(self, iteration, pretrain=False):
        # Get a batch.
        batch, n_step_batch, percentage = self.memory.sample_rollout(
            batch_size=self.batch_size,
            nsteps=self.nsteps,
            beta=self.replay_beta,
            gamma=self.gamma,
            pretrain=pretrain)

        target_Q_1step = self.sess.run(
            self.target_Q,
            feed_dict={
                self.obs1: batch['obs1'],
                self.state1: batch['states1'],
                self.aux1: batch['aux1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
                self.nstep_steps: np.ones((self.batch_size, 1)),
            })

        target_Q_nstep = self.sess.run(
            self.target_Q,
            feed_dict={
                self.obs1: n_step_batch['obs1'],
                self.state1: n_step_batch['states1'],
                self.aux1: n_step_batch['aux1'],
                self.rewards: n_step_batch['rewards'],
                self.nstep_steps: n_step_batch['step_reached'],
                self.terminals1: n_step_batch['terminals1'].astype('float32'),
            })
        critic_grads = [None] * self.num_critics
        critic_losses = [None] * self.num_critics
        td_errors = [None] * self.num_critics
        # Get all gradients and perform a synced update.
        ops = [
            self.actor_grads, self.actor_loss, *self.critic_grads,
            *self.critic_losses, *self.td_errors, self.scalar_summaries
        ]
        ret = self.sess.run(
            ops,
            feed_dict={
                self.obs0: batch['obs0'],
                self.importance_weights: batch['weights'],
                self.state0: batch['states0'],
                self.aux0: batch['aux0'],
                self.actions: batch['actions'],
                self.critic_target: target_Q_1step,
                self.nstep_critic_target: target_Q_nstep,
                self.pretraining_tf: batch['demos'].astype('float32'),
                self.memory_size: len(self.memory.storage),
                self.rss: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            })
        if self.num_critics == 2:
            actor_grads, actor_loss, critic_grads[0], critic_grads[1], critic_losses[0], critic_losses[1], td_errors[0], \
                td_errors[1], scalar_summaries = ret
        else:
            actor_grads, actor_loss, critic_grads[0], critic_losses[
                0], td_errors[0], scalar_summaries = ret
        self.memory.update_priorities(batch['idxes'], td_errors[0])
        for i in range(self.num_critics):
            self.critic_optimizers[i].update(
                critic_grads[i], stepsize=self.critic_lr)
        self.writer.add_summary(scalar_summaries, iteration)
        if iteration % self.policy_and_target_update_period == 0:
            self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        return critic_losses[0], actor_loss

    def set_sess(self, sess):
        self.sess = sess

    def initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def sync_optimizers(self):
        self.actor_optimizer.sync()
        for i in range(self.num_critics):
            self.critic_optimizers[i].sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            self.stats_sample = self.memory.sample_prioritized(
                batch_size=self.batch_size, replay_beta=self.replay_beta)
        values = self.sess.run(
            self.stats_ops,
            feed_dict={
                self.obs0: self.stats_sample['obs0'],
                self.actions: self.stats_sample['actions'],
                self.aux0: self.stats_sample['aux0'],
                self.state0: self.stats_sample['states0'],
            })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        return stats

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()

    def write_summary(self, summary):
        agent_summary = {
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_observations": self.normalize_observations,
            "normalize_state": self.normalize_state,
            "normalize_aux": self.normalize_aux,
            "action_noise": self.action_noise,
            "action_range": self.action_range,
            "return_range": self.return_range,
            "observation_range": self.observation_range,
            "actor_lr": self.actor_lr,
            "state_range": self.state_range,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "enable_popart": self.enable_popart,
            "reward_scale": self.reward_scale,
            "batch_size": self.batch_size,
            "critic_l2_reg": self.critic_l2_reg,
            "lambda_nstep": self.lambda_nstep,
            "lambda_1step": self.lambda_1step,
            "nsteps": self.nsteps,
            "replay_beta": self.replay_beta,
            "run_name": self.run_name,
            "lambda_pretrain": self.lambda_pretrain,
            "target_policy_noise": self.target_policy_noise,
            "target_policy_noise_clip": self.target_policy_noise_clip,
            "lambda_obj_conf_predict": self.lambda_obj_conf_predict,
            "lambda_target_predict": self.lambda_target_predict,
            "lambda_gripper_predict": self.lambda_gripper_predict,
        }
        summary["agent_summary"] = agent_summary
        md_string = self._markdownize_summary(summary)
        summary_op = tf.summary.text("param_info",
                                     tf.convert_to_tensor(md_string))
        text = self.sess.run(summary_op)
        self.writer.add_summary(text)
        self.writer.flush()
        print(md_string)

    @staticmethod
    def _markdownize_summary(data):
        result = []
        for section, params in data.items():
            result.append("### " + section)
            for param, value in params.items():
                result.append("* {} : {}".format(str(param), str(value)))
        return "\n".join(result)
