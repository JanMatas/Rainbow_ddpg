import numpy as np
import gym
from micoenv import demo_policies as demo
import cv2
import argparse
from baselines.common.misc_util import (
    boolean_flag, )
"""
Small example code to show how the demo policy interacts with 
the environment.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--env-id', help='env', default="Pusher-v1")
parser.add_argument('--policy', help='policy', default="pusher")
boolean_flag(parser, 'render', default=True)
args = vars(parser.parse_args())

#  Instantiate env and policy object.
env = gym.make(args["env_id"])
policy = demo.policies[args["policy"]]()


#  Run five full episodes.
for i in range(5):
    obs = env.reset()
    done = False
    policy.reset()
    while not done:
        state = env.get_state()
        action = policy.choose_action(state)
        obs, reward, done, info = env.step(action)

        if args["render"]:
            #  Write the reward for the current step on screen
            sim = np.uint8(env.render("rgb_array")[:, :, 0:3].copy() * 255)
            cv2.putText(sim, format(reward, '.2f'), (40, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.imshow("sim", cv2.resize(sim, (400, 400)))
            k = cv2.waitKey(1)
            if k == ord('s'):
                cv2.imwrite('demo_screen.png', sim)
