import argparse
import time
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
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


def run(env_id, eval_env_id, seed, noise_type, evaluation,demo_policy,use_velocities, num_dense_layers, dense_layer_size, layer_norm,demo_epsilon,replay_alpha,conv_size, **kwargs):
    # Configure things.
    if use_velocities:
        assert "velos" in env_id
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    if isinstance(env.reset(),dict) :
        print ("wrapping env")
        env = gym.wrappers.FlattenDictWrapper(
            env, dict_keys=['observation', 'desired_goal'])
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(eval_env_id)
        if isinstance(eval_env.reset(),dict) :
            print ("wrapping env")
            eval_env = gym.wrappers.FlattenDictWrapper(
                eval_env, dict_keys=['observation', 'desired_goal'])
        # eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.

    #TODO:

    #memory = Memory(limit=int(1e4 * 5))
    memory = PrioritizedMemory(limit=int(1e4 * 5), alpha=replay_alpha, demo_epsilon=demo_epsilon)
    critic = Critic(num_dense_layers, dense_layer_size, layer_norm)
    actor = Actor(nb_actions, env.state_space.shape[0], num_dense_layers, dense_layer_size, layer_norm, conv_size=conv_size)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    demo_env = gym.make(env_id)
    demo_policy_object = None
    if demo.policies[demo_policy]:
        demo_policy_object = demo.policies[demo_policy]()
    if use_velocities:
        demo_policy_object = demo.VelocityWrapper(demo_policy_object,demo_env)
    eval_avg = training.train(env=env,env_id=env_id, eval_env=eval_env,        action_noise=action_noise, actor=actor, critic=critic, memory=memory, demo_policy=demo_policy_object, demo_env=demo_env, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))
    return eval_avg


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='MicoEnv-pusher_fix-pixels-positive-ik-acton-v1')
    parser.add_argument('--eval-env-id', type=str, default='')
    boolean_flag(parser, 'render-eval', default=True)
    boolean_flag(parser, 'render-demo', default=True)
    boolean_flag(parser, 'layer-norm', default=False)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'use-velocities', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    boolean_flag(parser, 'normalize-state', default=True)
    boolean_flag(parser, 'normalize-aux', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
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
    parser.add_argument('--policy-and-target-update-period', type=int, default=2)
    parser.add_argument('--dense-layer-size', type=int, default=256)
    parser.add_argument('--num-dense-layers', type=int, default=4)
    parser.add_argument('--num-critics', type=int, default=2)
    parser.add_argument('--nsteps', type=int, default=10)
    parser.add_argument('--demo-terminality', type=int, default=5)
    parser.add_argument('--replay-alpha', type=float, default=0.8)
    parser.add_argument('--demo-epsilon', type=float, default=0.2)
    parser.add_argument('--lambda-obj-conf-predict', type=float, default=500000.0)
    parser.add_argument('--lambda-target-predict', type=float, default=500000.0)
    parser.add_argument('--lambda-gripper-predict', type=float, default=500000.0)

    boolean_flag(parser, 'positive-reward', default=True)
    boolean_flag(parser, 'only-eval', default=False)


    boolean_flag(parser, 'evaluation', default=True)


    args = parser.parse_args()


    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if not args["eval_env_id"]:
        args ["eval_env_id"] = args["env_id"]
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)
