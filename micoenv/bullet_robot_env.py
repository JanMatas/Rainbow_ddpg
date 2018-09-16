import pybullet as p
import sys
import time
from collections import deque
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
from gym.utils import seeding

sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")
timeStep = 1 / 240.0


class PhysClientWrapper:
    """
    This is used to make sure each BulletRobotEnv has its own physicsClient and
    they do not cross-communicate.
    """
    def __init__(self, other, physics_client_id):
        self.other = other
        self.physicsClientId = physics_client_id

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func(*args, **kwargs)


class BulletRobotEnv(Env):
    def __init__(self,
                 n_actions,  # Dimension of action vector.
                 n_substeps,  # Number of simulation steps to do in every env step.
                 observation_type="low_dim",
                 done_after=float("inf"),
                 use_gui=False,
                 frame_memory_len=0):
        self.n_substeps = n_substeps
        self.metadata = {
            'render.modes': ['rgbd_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.numSteps = 0
        if use_gui:
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)
        self.p = PhysClientWrapper(p, physics_client)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.doneAfter = done_after
        self.observation_type = observation_type
        self.seed()
        self.frameMemoryLen = frame_memory_len
        if frame_memory_len:
            self.frameMemory = deque(maxlen=frame_memory_len)

        self.viewMatrix = p.computeViewMatrix([-1.05, 0, 0.68], [0.1, 0, 0],
                                              [-0.5, 0, 1])
        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=45, aspect=4. / 3., nearVal=0.01, farVal=2.5)
        self.light = {
            "diffuse": 0.4,
            "ambient": 0.5,
            "spec": 0.2,
            "dir": [10, 10, 100],
            "col": [1, 1, 1]
        }
        self._env_setup(initial_qpos=None)

        self.action_space = spaces.Box(
            -1, 1, shape=(n_actions, ), dtype='float32')

        self.pixels_space = spaces.Box(
            -np.inf, np.inf, shape=(84, 84, 3), dtype='float32')
        if observation_type == "low_dim":
            self.observation_space = self.low_dim_space
        elif observation_type == "pixels":
            self.observation_space = self.pixels_space
        elif observation_type == "pixels_stacked":
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(84, 84, 12), dtype='float32')
        elif observation_type == "pixels_depth":
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(84, 84), dtype='float32')
        else:
            raise Exception("Unimplemented observation_type")

    @property
    def dt(self):
        return timeStep * self.n_substeps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        action = np.clip(action, self.action_space.low,
                         self.action_space.high)
        if self.numSteps == 0:
            self.startTime = time.time()

        self._set_action(action)
        for i in range(self.n_substeps):
            self.p.stepSimulation()
        self._step_callback()
        self.numSteps += 1
        current_obs = self._get_obs()

        while self.frameMemoryLen > 0 and len(
                self.frameMemory) < self.frameMemoryLen:
            self.frameMemory.append(current_obs)
            for i in range(self.n_substeps):
                self.p.stepSimulation()
            current_obs = self._get_obs()
        if self.frameMemoryLen > 0:
            self.frameMemory.append(current_obs)

        info = {}
        reward = self._get_reward()
        done = self._is_success(reward) or self.numSteps > self.doneAfter
        if done:
            info = {"episode": {"l": self.numSteps, "r": reward}}
        if self.frameMemoryLen > 0:
            return np.concatenate(self.frameMemory, 2), reward, done, info
        else:
            return current_obs, reward, done, info

    def reset_to_state(self, state, fn=None):
        self.numSteps = 0
        if not self._reset_sim(state=state, state_file=fn):
            self.reset()
        self.p.setTimeStep(timeStep)
        self.p.setGravity(0, 0, -10)
        current_obs = self._get_obs()
        while self.frameMemoryLen > 0 and len(
                self.frameMemory) < self.frameMemoryLen:
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
        self.p.setGravity(0, 0, -10)
        current_obs = self._get_obs()
        while self.frameMemoryLen > 0 and len(
                self.frameMemory) < self.frameMemoryLen:
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
        img = self.p.getCameraImage(
            width,
            height,
            self.viewMatrix,
            self.projMatrix,
            shadow=1,
            lightAmbientCoeff=self.light["ambient"],
            lightDiffuseCoeff=self.light["diffuse"],
            lightSpecularCoeff=self.light["spec"],
            lightDirection=self.light["dir"],
            lightColor=self.light["col"])

        rgb = np.array(img[2], dtype=np.float).reshape(height, width, 4) / 255
        rgb[:, :, 3], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
        rgb[:, :, 0] = rgb[:, :, 3]
        rgb = rgb[:, 11:-11, :]
        if mode == 'rgb_array':
            # rgb[:,:,3] = d # No depth
            return rgb[:, :, 0:3]
        elif mode == 'human':
            cv2.imshow("test", rgb[:, :, 0:3])
            cv2.waitKey(1)
            
    def render_debug_text(self, text, position, text_color_rgb=(1, 1, 1)):
        self.p.addUserDebugText(text, position, textColorRGB=text_color_rgb)

    def clear_debug_text(self):
        self.p.removeAllUserDebugItems()

    def _reset_sim(self, state=None, state_file=None):
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        raise NotImplementedError()

    def _sample_goal(self):
        raise NotImplementedError()

    def _set_action(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _step_callback(self):
        raise NotImplementedError()

    def _is_success(self, action):
        raise NotImplementedError()

    def draw_goal(self):
        raise NotImplementedError()

    def _get_reward(self):
        raise NotImplementedError()

