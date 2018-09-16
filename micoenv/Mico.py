import math
import os
import sys
import numpy as np

rootdir = os.path.dirname(sys.modules['__main__'].__file__)
micoUrdf = rootdir + "/mico_description/urdf/mico.urdf"
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


class Mico:
    def __init__(self,
                 p,  # Simulator object
                 spawn_pos=(0, 0, 0),  # Position where to spawn the arm.
                 reach_low=(-1, -1, 0),  # Lower limit of the arm workspace (might be used for safety).
                 reach_high=(1, 1, 1),  # Lower limit of the arm workspace.
                 randomize_arm=False,  # Whether arm initial position should be randomized.
                 urdf=micoUrdf  # Where to load the arm definition.
                 ):
        self.p = p
        self.armId = self.p.loadURDF(
            urdf,
            spawn_pos,
            self.p.getQuaternionFromEuler([0, 0, -math.pi / 2]),
            flags=self.p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.numJoints = self.p.getNumJoints(self.armId)
        assert self.numJoints

        self.jointVelocities = [0] * self.numJoints
        self.goalOrientation = [0, -math.pi, -math.pi / 2]
        self.goalGripper = 1.2
        self.goalReached = False
        self.goalEpsilon = 0.1
        self.truncated = False
        self.graspableObject = None
        self.reach_low = np.array(reach_low) + spawn_pos
        self.reach_low[2] = reach_low[2]
        self.reach_high = np.array(reach_high) + spawn_pos
        self.manual_control = False
        self.gripperConstraints = []
        self.fakeGrippers = []
        self.gripperPositions = [
            self.p.getJointState(self.armId, i)[0] for i in [7, 9]
        ]

        self.compute_ik_information()
        if randomize_arm:
            init_pos = np.random.normal(self.IKInfo["restPoses"], 0.05)
        else:
            init_pos = self.IKInfo["restPoses"]

        for i in range(len(self.IKInfo["restPoses"])):
            self.p.resetJointState(self.armId, i, init_pos[i])
        self.p.resetJointState(self.armId, 7, 1)
        self.p.resetJointState(self.armId, 9, 1)
        grip_pos = np.array(
            self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)[0])
        self.goalPosition = grip_pos

    def set_graspable_object(self, graspable_object, should_be_grasping=False):
        """ Sets the object the arm can create constraints to. Used for fake grasping."""
        self.graspableObject = graspable_object
        if should_be_grasping:
            assert graspable_object is not None
            self.create_gripper_constraints(graspable_object)

    def set_joint_poses(self, poses):
        """ Immediately sets the joint poses of the arm to poses."""
        for i in range(len(poses)):
            self.p.resetJointState(self.armId, i, poses[i])

    def compute_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self.armId, i),
                range(self.numJoints)))
        self.IKInfo = {}
        assert all(
            [len(joint_information[i]) == 17 for i in range(self.numJoints)])
        self.IKInfo["lowerLimits"] = list(
            map(lambda i: joint_information[i][8], range(8)))
        self.IKInfo["upperLimits"] = list(
            map(lambda i: joint_information[i][9], range(8)))

        self.IKInfo["jointRanges"] = list(
            map(lambda i: joint_information[i][9] - joint_information[i][8],
                range(8)))

        self.IKInfo["restPoses"] = list(
            map(math.radians, [
                250, 193.235290527, 52.0588226318, 348.0, -314.522735596,
                238.5, 0.0, 0
            ]))

        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.005] * self.numJoints
        self.IKInfo["endEffectorLinkIndex"] = 6

    def _truncate_val(self, val, lower_bound, upper_bound):
        new_val = min(max(val, lower_bound), upper_bound)
        self.truncated = val != new_val
        return new_val

    def apply_action(self, action):
        """ Sets the action of the arm for the next simulation step."""
        da = action[3]
        grip_state = self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)
        grip_pos = np.array(grip_state[0])

        # This clips the IK goal to stay in the workspace.
        self.goalPosition = np.clip(self.goalPosition + action[0:3],
                                    self.reach_low, self.reach_high)

        # This prevents the divergence between the IK goal and actual arm
        # position.
        self.goalPosition = np.clip(self.goalPosition + action[0:3],
                                    grip_pos - [0.07] * 3,
                                    grip_pos + [0.07] * 3)

        self.goalGripper = self._truncate_val(self.goalGripper + da, 0.3, 1.4)

    def step_simulation(self):
        """Step the simulation."""
        self.ik_step()
        self.check_contacts()

    def compute_ik_poses(self):
        """ Use the IK solver to compute goal joint poses to achieve IK goal."""
        joint_poses = self.p.calculateInverseKinematics(
            self.armId,
            targetPosition=self.goalPosition,
            targetOrientation=self.p.getQuaternionFromEuler(
                self.goalOrientation),
            **self.IKInfo)
        return joint_poses

    def ik_step(self):
        joint_poses = self.compute_ik_poses()

        # Set all body joints.
        for i in range(len(joint_poses)):
            self.p.setJointMotorControl2(
                bodyIndex=self.armId,
                jointIndex=i,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1)

        # Set gripper joints.
        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=7,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition=self.goalGripper,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1)
        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=9,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition=self.goalGripper,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1)

        # Save whether the arm has reached IK goal.
        gripper_reached = [
            abs(self.p.getJointState(self.armId, i)[0] - self.goalGripper) <
            self.goalEpsilon for i in [7, 9]
        ]
        self.goalReached = all(gripper_reached) and all(
            map(
                lambda joint: abs(self.p.getJointState(self.armId, joint)[
                              0] - joint_poses[joint]) < self.goalEpsilon, range(6)
            )
        )

    def check_contacts(self):
        """ Checks whether the gripper closed around graspable object in this step.
        and create fake griper constraints if it did. """
        self.prevGripperPositions = self.gripperPositions
        self.gripperPositions = [
            self.p.getJointState(self.armId, i)[0] for i in [7, 9]
        ]
        if self.graspableObject is not None:
            points = self.p.getContactPoints(
                bodyA=self.armId, bodyB=self.graspableObject)
            left_points = [p for p in points if p[3] == 7 or p[3] == 8]
            right_points = [p for p in points if p[3] == 9 or p[3] == 10]
            if all([
                    pos > 0.7 for pos in self.gripperPositions
            ]) and not all([pos > 0.9 for pos in self.prevGripperPositions
                            ]) and left_points and right_points:

                self.create_gripper_constraints(self.graspableObject)
            elif not all([pos > 0.6 for pos in self.gripperPositions]):
                if self.gripperConstraints:
                    self.remove_gripper_constraints()

    def get_joint_poses(self):
        """ Returns current joint poses."""
        return [self.p.getJointState(self.armId, i)[0] for i in range(10)]

    def init_gripper(self):
        """  Initialize objects on the gripper tips to anchor objects against. Note
         that those objects are not multi-bodies so they can interact with soft bodies,
         when necessary. """
        if self.fakeGrippers:
            return
        fake_gripper_object = self.p.createCollisionShape(
            self.p.GEOM_CYLINDER, height=0.001, radius=0.001)
        for i in range(2):  # Left and Right
            parent_link = 8 if i == 0 else 10
            for j in range(-1, 2):  # Multiple objects for stability
                new_body = self.p.createMultiBody(
                    baseMass=0.0001,
                    baseCollisionShapeIndex=fake_gripper_object,
                    basePosition=self.p.getLinkState(self.armId,
                                                     parent_link)[0],
                    useMaximalCoordinates=1)
                self.fakeGrippers.append(new_body)
                self.p.createConstraint(
                    parentBodyUniqueId=self.armId,
                    parentLinkIndex=parent_link,
                    childBodyUniqueId=new_body,
                    childLinkIndex=-1,
                    jointType=self.p.JOINT_POINT2POINT,
                    jointAxis=[1] * 3,
                    # Objects will be spaced in 1 cm distances
                    parentFramePosition=[0.022, -0.005, 0.01 * j],
                    childFramePosition=[0, 0, 0],
                )

    def is_grasping(self):
        return self.fakeGrippers != []

    def remove_gripper_constraints(self):
        if not self.gripperConstraints:
            return
        for constraint in self.gripperConstraints:
            self.p.removeConstraint(constraint)
        self.gripperConstraints = []

    def create_gripper_constraints(self, object_id):
        if not self.gripperConstraints:
            self.gripperConstraints.append(
                self.p.createConstraint(
                    parentBodyUniqueId=self.armId,
                    parentLinkIndex=6,
                    childBodyUniqueId=object_id,
                    childLinkIndex=-1,
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                    jointAxis=[1, 1, 1],
                    jointType=self.p.JOINT_POINT2POINT,
                ))

