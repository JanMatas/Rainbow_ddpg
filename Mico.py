import math
import os
import sys

import numpy as np

rootdir = os.path.dirname(sys.modules['__main__'].__file__)
micoUrdf = rootdir + "/fyp/mico_description/urdf/mico.urdf"
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


class Mico:
    def __init__(self, p, spawnPos=(0, 0, 0), enableDebug=False, reach_low=(-1, -1, 0), reach_high=(1, 1, 1),
                 randomizeArm=False, velocityControl=False, urdf=micoUrdf, rotation="normal", unreliableGrasp=0):
        self.p = p
        print(urdf)

        self.armId = self.p.loadURDF(urdf, spawnPos, self.p.getQuaternionFromEuler([0, 0, -math.pi / 2]),
                                     flags=self.p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.numJoints = self.p.getNumJoints(self.armId)
        assert self.numJoints

        self.jointVelocities = [0] * self.numJoints
        self.velocityControls = [0] * self.numJoints

        if rotation == "normal":
            self.goalOrientation = [0, -math.pi, -math.pi / 2]
        elif rotation == "alternative":
            self.goalOrientation = [0, -math.pi, 0]

        self.goalGripper = 1.2
        self.goalReached = False
        self.goalEpsilon = 0.1
        self.truncated = False
        self.graspableObject = None
        self.ikEnabled = not velocityControl  # If true, the arm will compute IK to goal position
        self.reach_low = np.array(reach_low) + spawnPos
        self.reach_low[2] = reach_low[2]
        self.reach_high = np.array(reach_high) + spawnPos
        self.manual_control = False
        self.enableDebug = enableDebug
        self.gripperConstraints = []
        self.fakeGrippers = []
        self.gripperPositions = [self.p.getJointState(self.armId, i)[0] for i in [7, 9]]
        self.unreliableGrasp = unreliableGrasp

        self.computeIKInformation(rotation)
        initPos = self.IKInfo["restPoses"]
        if randomizeArm:
            initPos = np.random.normal(initPos, 0.05)
        for i in range(len(self.IKInfo["restPoses"])):
            self.p.resetJointState(self.armId, i, initPos[i])
        self.p.resetJointState(self.armId, 7, 1)
        self.p.resetJointState(self.armId, 9, 1)
        if enableDebug:
            self.initiaizeDebugSliders()
        grip_pos = np.array(self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)[0])
        self.goalPosition = grip_pos

    def setGraspableObject(self, graspableObject, shouldBeGrasping=False):
        self.graspableObject = graspableObject
        if shouldBeGrasping:
            assert not graspableObject is None
            self.createGripperConstraints(graspableObject)

    def setJointPoses(self, poses):
        for i in range(len(poses)):
            self.p.resetJointState(self.armId, i, poses[i])
        # self.goalPosition = self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)[0]
        # print(self.goalPosition)

    def computeIKInformation(self, rotation):
        jointInformation = list(map(lambda i: self.p.getJointInfo(self.armId, i), range(self.numJoints)))
        self.IKInfo = {}
        assert all([len(jointInformation[i]) == 17 for i in range(self.numJoints)])
        self.IKInfo["lowerLimits"] = list(map(lambda i: jointInformation[i][8], range(8)))
        self.IKInfo["upperLimits"] = list(map(lambda i: jointInformation[i][9], range(8)))

        self.IKInfo["jointRanges"] = list(map(lambda i: jointInformation[i][9] - jointInformation[i][8], range(8)))

        if rotation == "normal":
            self.IKInfo["restPoses"] = list(
                map(math.radians, [250, 193.235290527, 52.0588226318, 348.0, -314.522735596, 238.5, 0.0, 0]))
        elif rotation == "alternative":
            self.IKInfo["restPoses"] = list(map(math.radians,
                                                [259.51617532423575, 195.64145210015167, 59.07982919102969,
                                                 346.65538012555663, -309.2623679328356, 336.1405174953582, 0.0]))
        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.005] * self.numJoints
        self.IKInfo["endEffectorLinkIndex"] = 6

    def truncateVal(self, val, lowerBound, upperBound):
        new_val = min(max(val, lowerBound), upperBound)
        self.truncated = val != new_val
        return new_val

    # Use either applyAction or setGoal, not both
    def applyAction(self, action):
        da = action[3]
        grip_state = self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)
        grip_pos = np.array(grip_state[0])
        shouldPenalize = np.any(self.goalPosition + action[0:3] > self.reach_high) or np.any(
            self.goalPosition + action[0:3] < self.reach_low)

        self.goalPosition = np.clip(self.goalPosition + action[0:3], self.reach_low, self.reach_high)
        grip_state = self.p.getLinkState(self.armId, 6)
        self.goalPosition = np.clip(self.goalPosition + action[0:3], np.array(grip_state[0]) - [0.07] * 3,
                                    np.array(grip_state[0]) + [0.07] * 3)

        self.goalGripper = self.truncateVal(self.goalGripper + da, 0.3, 1.4)
        return shouldPenalize

    def applyVelocities(self, velocities):
        assert len(velocities) == 10
        jointPoses = self.getJointPoses()
        for i in range(8):
            if jointPoses[i] > self.IKInfo["upperLimits"][i] and velocities[i] > 0 or jointPoses[i] < \
                    self.IKInfo["lowerLimits"][i] and velocities[i] < 0:
                velocities[i] = 0
        for i in range(7, 10):
            if jointPoses[i] < 0 and velocities[i] < 0 or jointPoses[i] > 1.4 and velocities[i] > 0:
                velocities[i] = 0
        for i in range(10):
            self.p.setJointMotorControl2(
                bodyIndex=self.armId,
                jointIndex=i,
                controlMode=self.p.VELOCITY_CONTROL,
                targetVelocity=velocities[i],
                force=500
            )

    def computeVelocities(self, action):
        action = action * 0.05
        self.applyAction(action)
        poses = np.zeros((10,))
        poses[0:8] = self.computeIkPoses()
        res = poses - self.getJointPoses()
        res[7] = action[3]
        res[9] = action[3]
        return np.clip(res, -1, 1)

    def stepSimulation(self):
        if self.enableDebug:
            self.checkDebugInputs()
        if self.ikEnabled:
            self.ikStep()
        self.checkContacts()

    def computeIkPoses(self):
        jointPoses = self.p.calculateInverseKinematics(
            self.armId,
            targetPosition=self.goalPosition,
            targetOrientation=self.p.getQuaternionFromEuler(self.goalOrientation),
            **self.IKInfo
        )
        return jointPoses

    def ikStep(self):
        jointPoses = self.computeIkPoses()

        for i in range(len(jointPoses)):
            self.p.setJointMotorControl2(
                bodyIndex=self.armId,
                jointIndex=i,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )
        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=7,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition=self.goalGripper,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1
        )
        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=9,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition=self.goalGripper,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1
        )
        gripperReached = [abs(self.p.getJointState(self.armId, i)[0] - self.goalGripper) < self.goalEpsilon for i in
                          [7, 9]]

        self.goalReached = all(gripperReached) and all(
            map(
                lambda i: abs(self.p.getJointState(self.armId, i)[0] - jointPoses[i]) < self.goalEpsilon, range(6)
            )
        )

    def checkContacts(self):
        self.prevGripperPositions = self.gripperPositions
        self.gripperPositions = [self.p.getJointState(self.armId, i)[0] for i in [7, 9]]

        if not self.graspableObject is None:

            points = self.p.getContactPoints(bodyA=self.armId, bodyB=self.graspableObject)
            leftPoints = [p for p in points if p[3] == 7 or p[3] == 8]
            rightPoints = [p for p in points if p[3] == 9 or p[3] == 10]

            if all([pos > 0.7 for pos in self.gripperPositions]) and not all(
                    [pos > 0.9 for pos in self.prevGripperPositions]) and leftPoints and rightPoints:

                self.createGripperConstraints(self.graspableObject)
            elif not all([pos > 0.6 for pos in self.gripperPositions]):
                if self.gripperConstraints:
                    self.removeGripperConstraints()

    def getJointPoses(self):
        return [self.p.getJointState(self.armId, i)[0] for i in range(10)]

    def initializeGripper(self):
        if self.fakeGrippers:
            return

        fakeGripperObject = self.p.createCollisionShape(self.p.GEOM_CYLINDER, height=0.001, radius=0.001)
        for i in range(2):  # Left and Right
            parentLink = 8 if i == 0 else 10
            for j in range(-1, 2):  # Multiple objects for stability
                newBody = self.p.createMultiBody(baseMass=0.0001,
                                                 baseCollisionShapeIndex=fakeGripperObject,
                                                 basePosition=self.p.getLinkState(self.armId, parentLink)[0],
                                                 useMaximalCoordinates=1
                                                 )

                self.fakeGrippers.append(newBody)
                self.p.createConstraint(
                    parentBodyUniqueId=self.armId,
                    parentLinkIndex=parentLink,
                    childBodyUniqueId=newBody,
                    childLinkIndex=-1,
                    jointType=self.p.JOINT_POINT2POINT,
                    jointAxis=[1] * 3,
                    parentFramePosition=[0.022, -0.005, 0.01 * j],  # Objects will be spaced in 1 cm distances
                    childFramePosition=[0, 0, 0],
                )

    def removeGripperConstraints(self):
        if not self.gripperConstraints:
            return
        for constraint in self.gripperConstraints:
            self.p.removeConstraint(constraint)
        self.gripperConstraints = []

    def createGripperConstraints(self, objectId):
        if not self.gripperConstraints:
            self.gripperConstraints.append(self.p.createConstraint(
                parentBodyUniqueId=self.armId,
                parentLinkIndex=6,
                childBodyUniqueId=objectId,
                childLinkIndex=-1,
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                jointAxis=[1, 1, 1],
                jointType=self.p.JOINT_POINT2POINT,

            )
            )

    ############# DEBUG MODE ONLY ##############################################
    def initiaizeDebugSliders(self):
        self.debug_x = self.p.addUserDebugParameter("x", -0.5, 0.5, 0.0)
        self.debug_y = self.p.addUserDebugParameter("y", -0.5, 0.5, 0.0)
        self.debug_z = self.p.addUserDebugParameter("z", 0, 1, 0.5)
        self.debug_gripper = self.p.addUserDebugParameter("gripper", 0, 1.45, 0)
        self.debug_camera_on = self.p.addUserDebugParameter("camera", 0, 1, 0)
        self.debug_manual_control = self.p.addUserDebugParameter("manual_control", 0.0, 1.0, 0)

    def isGrasping(self):
        return len(self.gripperConstraints) > 0

    def checkDebugInputs(self):
        self.manual_control = self.p.readUserDebugParameter(self.debug_manual_control) > 0.5
        if self.manual_control:
            x = self.p.readUserDebugParameter(self.debug_x)
            y = self.p.readUserDebugParameter(self.debug_y)
            z = self.p.readUserDebugParameter(self.debug_z)
            self.goalPosition = [x, y, z]
            self.goalGripper = self.p.readUserDebugParameter(self.debug_gripper)
