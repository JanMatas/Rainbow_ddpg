import argparse
from baselines import logger
from baselines.common.misc_util import (
    boolean_flag,
)
import rainbow_ddpg.distributed_train as training
from rainbow_ddpg.models import Actor, Critic
from rainbow_ddpg.prioritized_memory import PrioritizedMemory
from rainbow_ddpg.noise import *
import gym
import tensorflow as tf
from mpi4py import MPI
from micoenv import demo_policies as demo


def run(env_id, eval_env_id, noise_type, evaluation, demo_policy, num_dense_layers, dense_layer_size, layer_norm,
        demo_epsilon, replay_alpha, conv_size, **kwargs):

    # Create envs.
    env = gym.make(env_id)
    if evaluation:
        eval_env = gym.make(eval_env_id)
    else:
        eval_env = None
    demo_env = gym.make(env_id)

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError(
                'unknown noise type "{}"'.format(current_noise_type))

    # Initialize prioritized memory buffer
    memory = PrioritizedMemory(
        limit=int(1e4 * 5), alpha=replay_alpha, demo_epsilon=demo_epsilon)

    # Initialize neural nets.
    critic = Critic(num_dense_layers, dense_layer_size, layer_norm)
    actor = Actor(
        nb_actions,
        dense_layer_size,
        layer_norm,
        conv_size=conv_size)
    tf.reset_default_graph()

    # Instantiate demo policy object
    demo_policy_object = None
    if demo.policies[demo_policy]:
        demo_policy_object = demo.policies[demo_policy]()

    # Kick of the training
    eval_avg = training.train(
        env=env,
        env_id=env_id,
        eval_env=eval_env,
        action_noise=action_noise,
        actor=actor,
        critic=critic,
        memory=memory,
        demo_policy=demo_policy_object,
        demo_env=demo_env,
        **kwargs
    )

    # Clean up
    env.close()
    if eval_env is not None:
        eval_env.close()
    demo_env.close()

    return eval_avg


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--env-id',
        type=str,
        default='Pusher-v1')
    parser.add_argument('--eval-env-id', type=str, default='')
    boolean_flag(parser, 'render-eval', default=True)
    boolean_flag(parser, 'render-demo', default=True)
    boolean_flag(parser, 'layer-norm', default=False)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    boolean_flag(parser, 'normalize-state', default=True)
    boolean_flag(parser, 'normalize-aux', default=True)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)
    parser.add_argument('--nb-eval-steps', type=int, default=2)
    parser.add_argument('--nb-rollout-steps', type=int, default=100)
    parser.add_argument('--noise-type', type=str, default='normal_0.2')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save-folder', type=str, default='')
    parser.add_argument('--conv-size', type=str, default='small')
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--num-demo-steps', type=int, default=20)
    parser.add_argument('--num-pretrain-steps', type=int, default=2000)
    parser.add_argument('--run-name', type=str, default='ignore')
    parser.add_argument('--demo-policy', type=str, default='pusher')
    parser.add_argument('--lambda-pretrain', type=float, default=5.0)
    parser.add_argument('--lambda-nstep', type=float, default=0.5)
    parser.add_argument('--lambda-1step', type=float, default=1.0)
    parser.add_argument('--replay-beta', type=float, default=0.4)
    parser.add_argument('--reset-to-demo-rate', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--target-policy-noise', type=float, default=0.0)
    parser.add_argument('--target-policy-noise-clip', type=float, default=0.0)
    parser.add_argument(
        '--policy-and-target-update-period', type=int, default=2)
    parser.add_argument('--dense-layer-size', type=int, default=256)
    parser.add_argument('--num-dense-layers', type=int, default=4)
    parser.add_argument('--num-critics', type=int, default=2)
    parser.add_argument('--nsteps', type=int, default=10)
    parser.add_argument('--demo-terminality', type=int, default=5)
    parser.add_argument('--replay-alpha', type=float, default=0.8)
    parser.add_argument('--demo-epsilon', type=float, default=0.2)
    parser.add_argument(
        '--lambda-obj-conf-predict', type=float, default=500000.0)
    parser.add_argument(
        '--lambda-target-predict', type=float, default=500000.0)
    parser.add_argument(
        '--lambda-gripper-predict', type=float, default=500000.0)

    boolean_flag(parser, 'positive-reward', default=True)
    boolean_flag(parser, 'only-eval', default=False)

    boolean_flag(parser, 'evaluation', default=True)

    kwargs = parser.parse_args()

    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if kwargs.num_timesteps is not None:
        assert (kwargs.num_timesteps == kwargs.nb_epochs * kwargs.nb_epoch_cycles *
                kwargs.nb_rollout_steps)
    dict_args = vars(kwargs)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if not args["eval_env_id"]:
        args["eval_env_id"] = args["env_id"]
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)
