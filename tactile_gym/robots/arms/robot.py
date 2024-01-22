import os, inspect
import sys
import numpy as np
import copy
import math
import time

from tactile_gym.assets import get_assets_path, add_assets_path
from tactile_gym.robots.arms.ur5.ur5 import UR5
from tactile_gym.robots.arms.franka_panda.franka_panda import FrankaPanda
from tactile_gym.robots.arms.kuka_iiwa.kuka_iiwa import KukaIiwa
from tactile_gym.sensors.tactip.tactip import TacTip

# clean up printing
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})


class Robot:
    def __init__(
        self,
        pb,
        rest_poses,
        workframe_pos,
        workframe_rpy,
        TCP_lims,
        image_size=[128, 128],
        turn_off_border=False,
        arm_type="ur5",
        tactip_type="standard",
        tactip_core="no_core",
        tactip_dynamics={},
        contact_obj_id=None,
        show_gui=True,
        show_tactile=True,
    ):

        self._pb = pb
        self.arm_type = arm_type
        self.tactip_type = tactip_type
        self.tactip_core = tactip_core
        self.contact_obj_id = contact_obj_id

        # load the urdf file
        self.robot_id = self.load_robot()

        if self.arm_type == "ur5":
            self.arm = UR5(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "franka_panda":
            self.arm = FrankaPanda(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "kuka_iiwa":
            self.arm = KukaIiwa(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        else:
            sys.exit("Incorrect arm type specified {}".format(self.arm_type))


        # get relevent link ids for turning off collisions, connecting camera, etc
        tactip_link_ids = {}
        tactip_link_ids['body'] = self.arm.link_name_to_index["tactip_body_link"]
        tactip_link_ids['tip'] = self.arm.link_name_to_index["tactip_tip_link"]
        if tactip_type == "right_angle":
            tactip_link_ids['adapter'] = self.arm.link_name_to_index[
                "tactip_adapter_link"
            ]

        # connect the sensor the tactip
        self.tactip = TacTip(
            pb,
            robot_id=self.robot_id,
            tactip_link_ids=tactip_link_ids,
            image_size=image_size,
            turn_off_border=turn_off_border,
            tactip_type=tactip_type,
            tactip_core=tactip_core,
            tactip_dynamics=tactip_dynamics,
            show_tactile=show_tactile,
            tactip_num=1
        )

    def load_robot(self):
        """
        Load the robot arm model into pybullet
        """
        self.base_pos = [0, 0, 0]
        self.base_rpy = [0, 0, 0]
        self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
        robot_urdf = add_assets_path(os.path.join(
            "robot_assets",
            self.arm_type,
            self.arm_type + "_with_" + self.tactip_type + "_tactip.urdf",
        ))

        self.robot_id = self._pb.loadURDF(
            robot_urdf, self.base_pos, self.base_orn, useFixedBase=True
        )
        return self.robot_id

    def reset(self, reset_TCP_pos, reset_TCP_rpy):
        """
        Reset the pose of the UR5 and TacTip
        """
        self.arm.reset()
        self.tactip.reset()

        # move to the initial position
        self.arm.tcp_direct_workframe_move(reset_TCP_pos, reset_TCP_rpy)
        self.blocking_move(max_steps=1000, constant_vel=0.0001)


    def full_reset(self):
        self.load_robot()
        self.tactip.turn_off_tactip_collisions()

    def step_sim(self):
        """
        Take a step of the simulation whilst applying neccessary forces
        """

        # compensate for the effect of gravity
        self.arm.apply_gravity_compensation()

        # step the simulation
        self._pb.stepSimulation()

        # debugging
        # self.arm.draw_EE()
        # self.arm.draw_TCP() # only works with visuals enabled in urdf file
        # self.arm.draw_workframe()
        # self.arm.draw_TCP_box()
        # self.arm.print_joint_pos_vel()
        # self.arm.print_TCP_pos_vel()
        # self.arm.test_workframe_transforms()
        # self.arm.test_workvec_transforms()
        # self.arm.test_workvel_transforms()
        # self.tactip.draw_camera_frame()
        # self.tactip.draw_tactip_frame()

    def apply_action(
        self,
        motor_commands,
        control_mode="TCP_velocity_control",
        velocity_action_repeat=1,
        max_steps=100,
    ):
        
        if control_mode == "TCP_position_control":
            self.arm.tcp_position_control(motor_commands)

        elif control_mode == "TCP_velocity_control":
            self.arm.tcp_velocity_control(motor_commands)

        elif control_mode == "joint_velocity_control":
            self.arm.joint_velocity_control(motor_commands)

        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))

        final_contact_after_action = None
        if control_mode == "TCP_position_control":
            # repeatedly step the sim until a target pose is met or max iters
            # self.blocking_move(max_steps=max_steps, constant_vel=None)
            pybullet_steps, final_contact_after_action = self.blocking_move(max_steps=max_steps, constant_vel=0.0004)
        elif control_mode == "TCP_velocity_control":
            # apply the action for n steps to match control rate
            for i in range(velocity_action_repeat):
                self.step_sim()
            pybullet_steps = velocity_action_repeat
        else:
            # just do one step of the sime
            self.step_sim()
            pybullet_steps = 1
        
        return pybullet_steps, final_contact_after_action

    def blocking_move(self, max_steps=1000, constant_vel=None):
        """
        step the simulation until a target position has been reached or the max
        number of steps has been reached
        """
        # get target position
        targ_pos   = self.arm.target_pos_worldframe
        targ_rpy   = self.arm.target_rpy_worldframe
        targ_orn   = self.arm.target_orn_worldframe
        targ_j_pos = self.arm.target_joints

        pybullet_steps = 0
        contact_list = []
        final_contact = None
        for i in range(max_steps):

            # get the current position and veloicities (worldframe)
            (
                cur_TCP_pos,
                cur_TCP_rpy,
                cur_TCP_orn,
                _,
                _,
            ) = self.arm.get_current_TCP_pos_vel_worldframe()

            # get the current joint positions and velocities
            cur_j_pos, cur_j_vel = self.arm.get_current_joint_pos_vel()
            cur_j_pos = np.array(cur_j_pos)
            cur_j_vel = np.array(cur_j_vel)
            targ_j_pos = np.array(targ_j_pos)

            # get the diffence between targer and current joint positions
            diff_j = targ_j_pos - cur_j_pos
            
            # Move with constant velocity (from google-ravens)
            # break large position move to series of small position moves.
            if constant_vel is not None:
                norm = np.linalg.norm(diff_j)
                v = diff_j / norm if norm > 0 else 0
                step_j = cur_j_pos + v * constant_vel

                # break if joints are close enough
                if all(np.abs(diff_j) < constant_vel):
                    break

                # set joint control
                self._pb.setJointMotorControlArray(
                    self.robot_id,
                    self.arm.control_joint_ids,
                    self._pb.POSITION_CONTROL,
                    targetPositions=step_j,
                    targetVelocities=[0.0] * self.arm.num_control_dofs,
                    positionGains= [self.arm.pos_gain] * self.arm.num_control_dofs,
                    velocityGains= [self.arm.vel_gain] * self.arm.num_control_dofs
                )

            # step the simulation
            self.step_sim()
            pybullet_steps += 1

            # Get contact information at every step of the simulation
            contacts = self._pb.getContactPoints(bodyA=self.robot_id, bodyB=self.contact_obj_id)
            contact_points, is_contact = self.validate_contacts(contacts)
            if is_contact:
                # contact_list.append(contacts[np.random.randint(0, len(contacts))][6])
                contact_list.append(np.mean(contact_points, axis=0))

            # calc totoal velocity
            total_j_vel = np.sum(np.abs(cur_j_vel))

            # calculate the difference between target and actual pose
            pos_error = np.sum(np.abs(targ_pos - cur_TCP_pos))
            orn_error = np.arccos(
                np.clip((2 * (np.inner(targ_orn, cur_TCP_orn) ** 2)) - 1, -1, 1)
            )

            # break if the pose error is small enough
            # and the velocity is low enough
            if (pos_error < 2e-4) and (orn_error < 1e-3) and (total_j_vel < 0.1):
                break
        
        # Get the last contact point
        if contact_list:
            final_contact = (contact_list[-1][0], contact_list[-1][1], contact_list[-1][2])

        return pybullet_steps, final_contact

    def validate_contacts(self, contacts):
        if len(contacts) == 0:
            return None, False
        else:
            # Check if contact is tip with the object
            returned_contact_points = []
            for contact in contacts:
                if ((contact[1] != self.robot_id) or 
                    (contact[3] != self.tactip.tactip_link_ids['tip']) or
                    (contact[2] != self.contact_obj_id) or 
                    (contact[4] != -1)):
                    return None, False
                returned_contact_points.append(contact[6])
            return np.array(returned_contact_points), True


    def get_tactile_observation(self):
        return self.tactip.get_observation()
