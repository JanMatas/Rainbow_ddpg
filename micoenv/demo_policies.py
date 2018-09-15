import numpy as np


class DemoPolicy(object):

    def choose_action(self, state):
        return np.clip(self._choose_action(state), -0.5, 0.5)

    def reset(self):
        raise Exception("Not implemented")



class Waypoints(DemoPolicy):

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.currentWaypoint = 0

    def goToWaypoint(self, grip_pos, waypoint):
        return (
            np.concatenate((waypoint - grip_pos, [0.4])),
            np.linalg.norm(grip_pos - waypoint) < 0.05,
        )

    def _choose_action(self, state):
        grip_pos = state[0:3]
        action, done = self.goToWaypoint(
            grip_pos, self.waypoints[min(self.currentWaypoint, len(self.waypoints) - 1)]
        )
        if done:
            self.currentWaypoint += 1

        return action

    def done(self):
        return self.currentWaypoint >= len(self.waypoints)



class Pusher(DemoPolicy):

    def __init__(self):
        self.policy = None

    def _choose_action(self, state):
        if not self.policy:
            goal_pos = state[3:6]
            goal_pos[2] = 0.03
            object_pos = state[8:11]

            object_rel = object_pos - goal_pos
            behind_obj = object_pos + object_rel / np.linalg.norm(object_rel) * 0.06
            behind_obj[2] = 0.03
            waypoints = []
            waypoints.append(np.concatenate([behind_obj[:2], [0.2]]))
            waypoints.append(behind_obj)

            waypoints.append(goal_pos)

            self.policy = Waypoints(waypoints)
        action = self.policy._choose_action(state)

        if self.policy.done():
            self.policy = None
        # action += np.random.normal([0,0,0,0], 0.15)
        return action * 2

    def reset(self):
        self.policy = None




class ArmData(object):

    def __init__(self, data):
        assert data.shape == (13,)
        self.grip_pos, self.grip_velp, self.gripper_state, self.isGrasping, self.goalPosition, self.goalGripper = (
            data[0:3],
            data[3:6],
            data[6:8],
            data[8],
            data[9:12],
            data[12],
        )


policies = {
    "pusher": Pusher,
    "None": None,
}
