# Pybullet imports
import pybullet as p
import sys
import time
from collections import deque

import cv2
import gym
import numpy as np
import pybullet_data
from gym import spaces
from gym.utils import seeding

sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

timeStep = 1/240.0


class PhysClientWrapper:
    def __init__(self, other, physicsClientId):
        self.other = other
        self.physicsClientId = physicsClientId

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return  attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func( *args, **kwargs)

class BulletRobotEnv(gym.GoalEnv):
    def __init__(self, n_actions, n_substeps, observation_type="low_dim", useDone=False, doneAfter=float("inf"), use_gui=False, action_type="cont", neverDone=False, frameMemoryLen=0):
        self.viewer = None
        self.n_substeps = n_substeps
        self.metadata = {
            'render.modes': ['rgbd_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.numSteps = 0
        if use_gui:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        self.p = PhysClientWrapper(p, physicsClient)
        self.useDone = useDone
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.doneAfter = doneAfter
        self.observation_type = observation_type
        self.seed()
        self.neverDone = neverDone
        self.frameMemoryLen = frameMemoryLen
        if frameMemoryLen:
            self.frameMemory = deque(maxlen=frameMemoryLen)

        self.action_type = action_type
        self.viewMatrix = p.computeViewMatrix([-1.05,0,0.68],[0.1,0,0],[-0.5,0,1])
        self.projMatrix = p.computeProjectionMatrixFOV(fov=45,aspect=4./3.,nearVal=0.01,farVal=2.5)
        self.light = {
            "diffuse": 0.4,
            "ambient": 0.5,
            "spec": 0.2,
            "dir": [10,10, 100],
            "col":[1,1,1]
        }
        self._env_setup(initial_qpos=None)
        if action_type == "cont":
            self.action_space = spaces.Box(-1, 1, shape=(n_actions,), dtype='float32')
        elif action_type == "disc":
            self.action_space = spaces.Discrete(6)
        else:
            raise Exception("Unknown action space")

        self.pixels_space = spaces.Box(-np.inf, np.inf, shape=(84,84,3), dtype='float32')
        if observation_type == "low_dim":
            self.observation_space = self.low_dim_space
        elif observation_type == "pixels":
            self.observation_space = self.pixels_space
        elif observation_type == "pixels_stacked":
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(84,84,12), dtype='float32')
        elif observation_type == "pixels_depth":
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(84,84), dtype='float32')
        elif observation_type == "combined":
            self.observation_space = spaces.Dict(dict(
                low_dim=low_dim_space,
                pixels=pixels_space,
                observation=saces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')
            ))
        else:
            raise Exception("Unimplemented observation_type")


    @property
    def dt(self):
        return timeStep * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.action_type == "disc":
            action_index = action // 2
            action_arr = np.zeros(4)
            action_arr[action_index] = -1 if action % 2 == 0 else 1
            action = action_arr
        elif self.action_type == "cont":
            action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.numSteps == 0:
            self.startTime = time.time()

        self._set_action(action)
        for i in range(self.n_substeps):
            self.p.stepSimulation()
        self._step_callback()
        self.numSteps += 1
        done = False
        current_obs = self._get_obs()

        while self.frameMemoryLen > 0 and len(self.frameMemory) < self.frameMemoryLen:
            self.frameMemory.append(current_obs)
            for i in range(self.n_substeps):
                self.p.stepSimulation()
            current_obs = self._get_obs()
        if self.frameMemoryLen > 0:
            self.frameMemory.append(current_obs)


        info = {}
        done = False
        reward = self._get_reward()

        if self.useDone:

            done = self._is_success(reward) or self.numSteps > self.doneAfter
            if done:
                info = {"episode": {"l":self.numSteps, "r":reward}}
            if done and self.neverDone:
                self.goal = self._sample_goal()
                self.draw_goal()

                done = self.numSteps > self.doneAfter

        if self.frameMemoryLen > 0:

                return np.concatenate(self.frameMemory, 2), reward, done, info
        else:
                return current_obs, reward, done, info




    def reset_to_state(self, state, fn=None):
        self.numSteps = 0
        if not self._reset_sim(state=state, stateFile=fn):
            self.reset()
        self.p.setTimeStep(timeStep)
        self.p.setGravity(0,0,-10)
        current_obs = self._get_obs()
        while self.frameMemoryLen > 0 and len(self.frameMemory) < self.frameMemoryLen:
            self.frameMemory.append(current_obs)
            for i in range(self.n_substeps):
                self.p.stepSimulation()
            current_obs = self._get_obs()
        if self.frameMemoryLen > 0:
            self.frameMemory.append(current_obs)
        if self.frameMemoryLen > 0:
                return np.concatenate(self.frameMemory, 2)
        else:
                return current_obs

    def reset(self):
        self.numSteps = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.p.setTimeStep(timeStep)
        self.p.setGravity(0,0,-10)
        current_obs = self._get_obs()
        while self.frameMemoryLen > 0 and len(self.frameMemory) < self.frameMemoryLen:
            self.frameMemory.append(current_obs)
            for i in range(self.n_substeps):
                self.p.stepSimulation()
            current_obs = self._get_obs()
        if self.frameMemoryLen > 0:
            self.frameMemory.append(current_obs)
        if self.frameMemoryLen > 0:

                return np.concatenate(self.frameMemory, 2)
        else:
                return current_obs


    def close(self):
        pass

    def render(self, mode='human'):
        width, height = 106, 84
        img = self.p.getCameraImage(width,height,self.viewMatrix,self.projMatrix, shadow=1, lightAmbientCoeff=self.light["ambient"], lightDiffuseCoeff=self.light["diffuse"], lightSpecularCoeff=self.light["spec"], lightDirection=self.light["dir"], lightColor=self.light["col"])
        # img = self.p.getCameraImage(width,height)
        # img = self.p.getCameraImage(width,height)
        rgb =  np.array(img[2], dtype=np.float).reshape(height,width,4) / 255

        rgb[:,:,3],rgb[:,:,2] = rgb[:,:,2],rgb[:,:,0]
        rgb[:,:,0] = rgb[:,:,3]
        rgb = rgb[:,11:-11,:]


        # d = np.array(img[3], dtype=np.float).reshape(width,height) /3
        # d = d 
        # d -= d.min()
        # d *= 1/d.max()
        if mode == 'rgb_array':

            #rgb[:,:,3] = d # No depth
            
            return rgb[:,:,0:3]
        elif mode == 'human':
            cv2.imshow("test", rgb[:,:,0:3])
            cv2.waitKey(1)


    def _get_viewer(self):
        raise NotImplementedError()



    def renderDebugText(self, text, position, textColorRGB=(1,1,1)):

        self.p.addUserDebugText(text, position, textColorRGB=textColorRGB)
    def clearDebugText(self):
        self.p.removeAllUserDebugItems()


    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d



    def timePerStep(self):
        return (time.time() - self.startTime) / self.numSteps
    # Extension methods
    # ----------------------------

    def _reset_sim(self, state=None, stateFile=None):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass


    def _render_cleanup(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass


    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
