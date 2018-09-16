import numpy as np
import gym
import time
from micoenv import demo_policies as demo
import cv2
import argparse
from baselines.common.misc_util import (
    boolean_flag, )
parser = argparse.ArgumentParser()
parser.add_argument('--env-id', help='env')
parser.add_argument('--policy', help='policy')

boolean_flag(parser, 'render', default=False)
args = vars(parser.parse_args())
import math
# cap =  cv2.VideoCapture(0)
# ret, frame = cap.read()
# real_arm.step()
# print(frame.shape)
env = gym.make(args["env_id"])
policy = demo.policies[args["policy"]]()

# a = cv2.resize(frame[:,65:-65,:], (84, 84))
# from mico_real import MicoReal

rs = []
successes = 0
step = 0
prev = time.time()
times = []
for i in range(5):
    obs = env.reset()
    done = False
    # goal = env.goalstate()
    r = 0
    policy.reset()
    # real_arm = MicoReal(cartesian_control=False)
    num_steps = 0
    while not done:

        step += 1
        state = env.get_state()
        action = policy.choose_action(state)
        obs, reward, done, info = env.step(action)

        if args["render"]:
            sim = np.uint8(env.render("rgb_array")[:, :, 0:3].copy() * 255)
            cv2.putText(sim, format(reward, '.2f'), (40, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        # ret, frame = cap.read()
        angles = list(map(math.degrees, env.arm.getJointPoses()[0:7]))
        # real_arm.set_goal_angles(angles)
        # real_arm.applyAction(action)
        # real_arm.step()
        # a = cv2.resize(frame[:,65:-65,:], (84, 84))
        # state = env.get_state()
        # aux = env.get_aux()

        # if np.random.uniform(0,1) < 0.02:
        #     env.store_state("test.bullet")
        #     env.reset_to_state(state,fn="test.bullet")

        if args["render"]:
            cv2.imshow("sim", cv2.resize(sim, (400, 400)))

            k = cv2.waitKey(1)
            if k == ord('s'):
                cv2.imwrite('rand_{}.png'.format(step), sim)

        num_steps += 1

    #print (reward, num_steps)
    if reward > 0:
        successes += 1
print(successes)
