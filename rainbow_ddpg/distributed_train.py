import os
from rainbow_ddpg.ddpg import DDPG
import baselines.common.tf_util as U
import itertools
from baselines import logger
import numpy as np
import tensorflow as tf
import cv2
import gym
from baselines.common.schedules import LinearSchedule
import sys

tmp = os.path.dirname(sys.modules['__main__'].__file__) + "/tmp"
demo_states_dir = tmp+"/demo_states"
import os
if not os.path.exists(demo_states_dir):
    os.makedirs(demo_states_dir)
demo_states_template = demo_states_dir+ "/{}/{}.bullet"
from threading import Thread

class Renderer(object):
    def __init__(self, type, run_name, epoch, seed=None):
        self.directory = tmp + '/ddpg_video_buffer/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.run_name= run_name
        if not seed is None:
            run_name = run_name + str(seed)
        self.fname= '{}-{}-{}.avi'.format(type, run_name, epoch + 1)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.rgb = cv2.VideoWriter(self.directory + self.fname, fourcc, 30.0, (84, 84))

    def record_frame(self, frame, reward, action, q):
        frame = np.array(frame[:,:,0:3].copy()*255, dtype=np.uint8)
        cv2.putText(frame,format(reward, '.2f'), (40,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        cv2.putText(frame,format(action[0], '.2f'), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv2.putText(frame,format(action[1], '.2f'), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv2.putText(frame,format(action[2], '.2f'), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv2.putText(frame,format(action[3], '.2f'), (40,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv2.putText(frame,format(q[0][0], '.2f'), (40,35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        self.rgb.write(frame)

    def finalize_and_upload(self):
        self.rgb.release()

##### Stuff to be done in workers ########
class RolloutWorker(object):
    def __init__(self, env_id, agent, num_steps, run_name, reset_to_demo_rate_sched, seed,demo_terminality):
        self.num_steps = num_steps
        self.reset_to_demo_rate_sched = reset_to_demo_rate_sched
        self.advance_epoch()
        self.run_name = run_name
        self.agent = agent
        self.env_id = env_id
        self.seed = seed
        self.demo_terminality = demo_terminality
        if seed == 0:
            self.renderer = Renderer("rollout", run_name, 0, seed)
        else:
            self.renderer = None
        self.rendering = seed == 0 # Render only first one
        self.steps_to_render = 300

    def advance_epoch(self):
        self.epoch_rewards = []
        self.epoch_qs = []

    def exec_rollouts(self):
        with self.agent.sess.as_default():
            env = gym.make(self.env_id)
            env.seed(self.seed)
            obs0, aux0, state0 = env.reset(), env.get_aux(), env.get_state()
            episode_reward = 0
            episodes = 1
            epoch_episodes = 1

            for i in range(self.num_steps):
                if not self.renderer and np.random.uniform(0,1) < 0.0005:
                    self.steps_to_render = 300
                    self.renderer = Renderer("rollout", self.run_name, i, self.seed)
                if self.renderer and self.steps_to_render == 0:
                    self.renderer.finalize_and_upload()
                    self.renderer = None

                action, q, _, _, _ = self.agent.pi(obs0, aux0, state0, apply_noise=True, compute_Q=True)
                self.epoch_qs.append(q)
                assert action.shape == env.action_space.shape
                obs1, r, done, info = env.step(action)
                episode_reward += r
                state1, aux1 = env.get_state(), env.get_aux()
                self.agent.store_transition(state0, obs0, action, r, state1, obs1, done, None, None, aux0, aux1, None)
                if self.renderer and self.steps_to_render > 0:
                    frame = env.render(mode="rgb_array")
                    self.renderer.record_frame(frame,r, action, q)
                    self.steps_to_render -= 1
                obs0, aux0, state0 = obs1, aux1, state1
                if done:

                    self.agent.save_reward(episode_reward)
                    episodes += 1
                    self.epoch_rewards.append(episode_reward)
                    episode_reward = 0.
                    if np.random.uniform(0, 1) < self.reset_to_demo_rate_sched.value(i):
                        while True:
                            memory = self.agent.memory
                            with memory.lock:
                                demo_index = np.random.randint(0, memory.num_demonstrations)
                                state = memory.storage[demo_index][0]
                                terminal_demo = False
                                for di in range(demo_index, demo_index + self.demo_terminality):
                                    terminal_demo = terminal_demo or memory.storage[di % memory.num_demonstrations][6]
                            if not terminal_demo:
                                break
                        fn = demo_states_template.format(self.run_name, memory.storage[demo_index][11])
                        obs0 = env.reset_to_state(state, fn=fn)
                    else:
                        obs0 = env.reset()
                    aux0, state0 = env.get_aux(), env.get_state()
                    
class DistributedTrain(object):
    def __init__(self, run_name, agent, env, nb_rollout_steps, num_pretrain_steps, nb_epochs, nb_epoch_cycles, nb_train_steps, demo_env, demo_policy, render_demo, num_demo_steps, reset_to_demo_rate, render_eval, eval_env, nb_eval_steps, env_id, policy_and_target_update_period, demo_terminality, load_file, save_folder, only_eval):
        # Misc params
        self.run_name = run_name
        self.agent = agent
        self.load_file = load_file
        self.save_folder = save_folder
        self.only_eval = only_eval
        self.saver = tf.train.Saver()

        # Main rollout params
        self.nb_rollout_steps = nb_rollout_steps

        # Train params
        self.num_pretrain_steps = num_pretrain_steps
        self.nb_epochs = nb_epochs
        self.nb_epoch_cycles = nb_epoch_cycles
        self.nb_train_steps = nb_train_steps
        self.policy_and_target_update_period = policy_and_target_update_period

        # Demo params
        self.demo_env = demo_env
        self.demo_policy = demo_policy
        self.render_demo = render_demo
        self.num_demo_steps = num_demo_steps
        self.reset_to_demo_rate = reset_to_demo_rate 
        self.demo_terminality = demo_terminality

        # Render params
        self.render_eval = render_eval
        self.eval_env = eval_env
        self.nb_eval_steps = nb_eval_steps

        self.env_id = env_id

    def start(self):
        with U.single_threaded_session() as sess:
            self.sess = sess
            self.agent.set_sess(sess)
            if self.load_file:
                self.saver.restore(sess, self.load_file)
            else:
                self.agent.initialize_vars()
            self.agent.sync_optimizers()

            self._write_summary()

            sess.graph.finalize()
            successes = 0
            if self.only_eval:
                for i in range(20):
                    done = False
                    obs = self.eval_env.reset()
                    self.agent.reset()
                    total_r = 0
                    while not done:
                        aux0 = self.eval_env.get_aux()
                        state0 = self.eval_env.get_state()
                        action, q, object_conf, gripper, target = self.agent.pi(obs, aux0, state0, apply_noise=False, compute_Q=True)
                        try:
                            obs, r, done, info = self.eval_env.step(action)
#                            self.eval_env.render()
                        except StopIteration:
                            print ("interrupted iteration")
                            done = True
                        if done and r > 0:
                          print("success")
                          successes += 1
                        elif done:
                          print("fail")


                        total_r += r
                    print(total_r)
                print (successes)
                return

            if self.demo_policy:
                self._initialize_memory_with_policy()
            self.agent.memory.demonstrationsDone()
            self.pretrain()
            ret = self.train()
            self.sess = None
            return ret

    #### Functions to be done synchronously ######
    def train(self):
        num_steps = self.nb_epochs * self.nb_epoch_cycles * self.nb_rollout_steps
        rws = []
        # Use a single rollout worker for now
        for i in range(1):
            rw = RolloutWorker(self.env_id, self.agent, num_steps,self.run_name, LinearSchedule(2e5, initial_p=self.reset_to_demo_rate, final_p=0.1), i, self.demo_terminality)
            thread = Thread(target=rw.exec_rollouts, daemon=True)
            thread.start()
            rws.append(rw)
        eval_episodes = 1
        final_evals = []
        iteration = self.num_pretrain_steps
        for epoch in range(self.nb_epochs):
            for cycle in range(self.nb_epoch_cycles):
                print ("Cycle: {}/{}".format(cycle, self.nb_epoch_cycles) +
                       "["+ "-" * cycle + " " * (self.nb_epoch_cycles - cycle) + "]"
                 , end="\r")
                self.agent.memory.grow_limit()
                for t_train in range(self.nb_train_steps):
                    cl, al = self.agent.train(iteration)
                    iteration += 1
                    if iteration % self.policy_and_target_update_period == 0:
                        self.agent.update_target_net()
            logger.record_tabular("epoch", epoch)
            logger.record_tabular("total transitions", self.agent.memory.total_transitions)
            logger.record_tabular("run_name", self.run_name)
            all_rewards = list(itertools.chain(*[rw.epoch_rewards for rw in rws]))
            all_qs = list(itertools.chain(*[rw.epoch_qs for rw in rws]))
            logger.record_tabular("rollout_rewards", np.mean(all_rewards) if all_rewards else "none")
            logger.record_tabular("rollout_qs", np.mean(all_qs) if all_qs else "none")
            for rw in rws:
                rw.advance_epoch()
            ### Evaluate #####
            print ("Executed epoch cycles, starting the evaluation.")
            eval_obs0, aux0, state0 = self.eval_env.reset(), self.eval_env.get_aux(), self.eval_env.get_state()
            eval_episode_reward = 0.
            eval_episode_rewards = []
            eval_qs = []
            if self.render_eval:
                renderer = Renderer("eval", self.run_name, epoch)
            for eval_episode in range(self.nb_eval_steps):

                eval_done = False
                print ("Evaluation {}/{}".format(eval_episode, self.nb_eval_steps), end="\r")
                while not eval_done:
                    eval_action, eval_q, object_conf, gripper, target = self.agent.pi(eval_obs0, aux0, state0, apply_noise=False, compute_Q=True)
                    eval_obs0, eval_r, eval_done, eval_info = self.eval_env.step( eval_action)
                    aux0, state0 = self.eval_env.get_aux(), self.eval_env.get_state()
                    eval_qs.append(eval_q)

                    if self.render_eval:
                        frame = self.eval_env.render(mode="rgb_array")
                        renderer.record_frame(frame, eval_r, eval_action, eval_q)
                    eval_episode_reward += eval_r

                    actual_object_conf = state0[8:11]
                    actual_grip = state0[0:3]
                    actual_target = state0[3:6]

                    diff_object_conf, diff_grip, diff_target = np.linalg.norm(actual_object_conf - object_conf), np.linalg.norm(actual_grip - gripper), np.linalg.norm(actual_target - target)
                    self.agent.save_aux_prediction(diff_object_conf, diff_grip, diff_target)
                eval_obs0, aux0, state0 = self.eval_env.reset(), self.eval_env.get_aux(), self.eval_env.get_state()
                eval_episode_rewards.append(eval_episode_reward)
                self.agent.save_eval_reward(eval_episode_reward, eval_episodes)
                eval_episodes += 1
                eval_episode_reward = 0.

            if self.render_eval:
                renderer.finalize_and_upload()
            if eval_episode_rewards and epoch > self.nb_epochs - 5:
                final_evals.append(np.mean(eval_episode_rewards))
            if  epoch % 5  == 0 and self.save_folder:
                path = self.save_folder + "/" +self.run_name + "epoch{}.ckpt".format(epoch)
                print("Saving model to " + path)
                save_path = self.saver.save(self.sess, path)

            logger.record_tabular("eval_rewards", np.mean(eval_episode_rewards) if eval_episode_rewards else "none")
            logger.record_tabular("eval_qs", np.mean(eval_qs) if eval_qs else "none")
            logger.dump_tabular()
            logger.info('')
        return - np.mean(final_evals)

    def pretrain(self):
        iteration = 0
        while self.num_pretrain_steps > 0:
            print ("Pretrain: {}/{}".format(iteration, self.num_pretrain_steps)
                 , end="\r")
            cl, al = self.agent.train(iteration, pretrain=True)
            iteration +=1
            if iteration % self.policy_and_target_update_period == 0:
                self.agent.update_target_net()
            self.num_pretrain_steps -= 1

    def _initialize_memory_with_policy(self):
        print("Start collecting demo transitions")
        obs0, aux0, state0 = self.demo_env.reset(), self.demo_env.get_aux(), self.demo_env.get_state()
        self.demo_policy.reset()
        os.makedirs(demo_states_dir+"/"+self.run_name, exist_ok=True)

        if self.render_demo:
            renderer = Renderer("demo", self.run_name, 0)
        iteration = -1
        successes = 0
        total_r = 0
        total_dones = 0

        for i in range(self.num_demo_steps):

            print ("Demo: {}/{}".format(i, self.num_demo_steps)
                 , end="\r")
            transitions = []
            frames = []
            while True:
                iteration += 1
                action = self.demo_policy.choose_action(state0)
                fn = demo_states_template.format(self.run_name, iteration)
                self.demo_env.store_state(fn)
                obs1, r, done, info = self.demo_env.step(action)
                total_r += r
                aux1, state1 = self.demo_env.get_aux(), self.demo_env.get_state()

                transitions.append((state0, obs0, action, r, state1, obs1, done, None, None, aux0, aux1, iteration))
                obs0, aux0, state0 = obs1, aux1, state1
                if self.render_demo:
                    frame = self.demo_env.render(mode="rgb_array")
                    frames.append(frame)

                if done:
                    total_dones += 1

                    obs0, aux0, state0 = self.demo_env.reset(), self.demo_env.get_aux(), self.demo_env.get_state()
                    self.demo_policy.reset()
                    if r > 0:
                        successes += 1

                        for t in transitions:
                            self.agent.store_transition(*t, demo=True)
                        if self.render_demo:
                            for (j, frame) in enumerate(frames):
                                renderer.record_frame(frame, transitions[j][3], transitions[j][2],[[0]])
                        break
                    else:
                        print("Bad demo - throw away")
                    transitions = []
                    frames = []
        if self.render_demo:
            renderer.finalize_and_upload()

        print("Collected {} demo transition.".format(self.agent.memory.num_demonstrations))
        print("Successes {} .".format(successes))
        print("Reward {} .".format(total_r / total_dones))
    def _write_summary(self):
        training_text_summary = {
            "env_data": {
                "env:": str(self.eval_env),
                "run_name": self.run_name,
                "obs_shape":  self.eval_env.observation_space.shape,
                "action_shace":  self.eval_env.action_space.shape,
                "aux_shape":  self.eval_env.aux_space.shape,
                "call_command": " ".join(sys.argv),
            },
            "demo_data": {
                "policy": self.demo_policy.__class__.__name__,
                "number_of_steps": self.num_demo_steps,
                "demo_terminality": self.demo_terminality,
                "reset_to_demo_rate": self.reset_to_demo_rate,
            },
            "training_data": {
                "nb_train_steps": self.nb_train_steps,
                "nb_rollout_steps": self.nb_rollout_steps,
                "num_pretrain_steps": self.num_pretrain_steps,
                "nb_epochs": self.nb_epochs,
                "nb_epoch_cycles": self.nb_epoch_cycles,

            },

        }
        self.agent.write_summary(training_text_summary)

def train(env,env_id, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic, normalize_observations, normalize_aux, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, load_file, save_folder, only_eval,
    run_name, lambda_pretrain, lambda_1step, lambda_nstep, replay_beta,reset_to_demo_rate, tau=0.01, eval_env=None, demo_policy=None, num_demo_steps=0, demo_env=None, render_demo=False, num_pretrain_steps=0, policy_and_target_update_period=2, demo_terminality=5, **kwargs):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape, env.state_space.shape, env.aux_space.shape,
        gamma=gamma, tau=tau, normalize_observations=normalize_observations,normalize_aux=normalize_aux,
        batch_size=batch_size, action_noise=action_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, run_name=run_name, lambda_pretrain=lambda_pretrain, lambda_nstep=lambda_nstep, lambda_1step=lambda_1step,
        replay_beta=replay_beta, policy_and_target_update_period=policy_and_target_update_period, **kwargs)
    dt  = DistributedTrain(run_name, agent, env, nb_rollout_steps, num_pretrain_steps, nb_epochs, nb_epoch_cycles, nb_train_steps, demo_env, demo_policy, render_demo, num_demo_steps, reset_to_demo_rate, render_eval, eval_env, nb_eval_steps, env_id, policy_and_target_update_period, demo_terminality, load_file, save_folder, only_eval)

    return dt.start()




