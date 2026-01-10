"""
Optimized version for high-performance mocap visualization
Target: 120Hz data rate
"""

import mujoco
import mujoco_viewer
import numpy as np
import sys
import os
import time
import yaml
import argparse

import lumosdk.LuMoSDKClient as LuMoSDKClient

argparser = argparse.ArgumentParser()
argparser.add_argument('--ip', type=str, default='192.168.2.30', help='IP address of the LuMo mocap system')
argparser.add_argument('--xml_path', type=str, default='../robots/g1/g1_29dof_rev_1_0.xml', help='Path to G1 Mujoco XML file')
argparser.add_argument('--config', type=str, default='../config/g1.yaml', help='Path to G1 config file')
argparser.add_argument('--stats', action='store_true', help='Print performance statistics')
args = argparser.parse_args()

# G1 Joint names in Mujoco order (29 DOF)
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def live_fk():
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, args.xml_path)
    config_path = os.path.join(script_dir, args.config)
    
    with open(config_path, 'r') as f:
        joint_mapping = yaml.safe_load(f)['Motor_Joint_Map']
    
    # Pre-compute mapping list for fast lookup
    lumo_names = [joint_mapping.get(name, None) for name in G1_JOINT_NAMES]
    
    # Load Mujoco model
    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Initialize viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    
    # Connect to LuMoSDK
    print(f"Connecting to {args.ip}...")
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)
    print("Connected! Running at maximum speed...\n")
    
    # Pre-allocate arrays
    joint_angles = np.zeros(29, dtype=np.float64)
    pos_mm = np.zeros(3, dtype=np.float64)
    quat_yup = np.zeros(4, dtype=np.float64)
    
    # Performance monitoring (optional)
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    # Find hips bone index once
    hips_idx = None
    first_skeleton = None
    
    try:
        while viewer.is_alive:
            frame = LuMoSDKClient.ReceiveData(1)  # Non-blocking
            
            if frame is None or len(frame.skeletons) == 0:
                viewer.render()
                continue
            
            skeleton = frame.skeletons[0]
            if not skeleton.IsTrack:
                viewer.render()
                continue
            
            # Find hips bone index on first frame
            if hips_idx is None:
                for i, bone in enumerate(skeleton.skeletonBones):
                    if 'hips' in bone.Name.lower() or 'hip' in bone.Name.lower():
                        hips_idx = i
                        break
                if hips_idx is None:
                    hips_idx = 0
            
            # Extract base pose (optimized)
            hips = skeleton.skeletonBones[hips_idx]
            pos_mm[0] = hips.X
            pos_mm[1] = hips.Y
            pos_mm[2] = hips.Z
            
            # Y-UP to Z-UP conversion + mm to m
            data.qpos[0] = pos_mm[0] * 0.001
            data.qpos[1] = pos_mm[2] * 0.001
            data.qpos[2] = pos_mm[1] * 0.001
            
            # Quaternion Y-UP to Z-UP
            quat_yup[0] = hips.qx
            quat_yup[1] = hips.qy
            quat_yup[2] = hips.qz
            quat_yup[3] = hips.qw
            
            data.qpos[3] = quat_yup[3]  # w
            data.qpos[4] = quat_yup[0]  # x
            data.qpos[5] = quat_yup[2]  # z (was y)
            data.qpos[6] = quat_yup[1]  # y (was z)
            
            # Extract joint angles (optimized)
            motor_angle = skeleton.MotorAngle
            for i, lumo_name in enumerate(lumo_names):
                if lumo_name and lumo_name in motor_angle:
                    joint_angles[i] = motor_angle[lumo_name]
                else:
                    joint_angles[i] = 0.0
            
            # Update joint positions
            data.qpos[7:36] = joint_angles
            
            # Forward kinematics and render
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.qpos[0:3]
            viewer.render()
            
            # Stats (optional)
            if args.stats:
                frame_count += 1
                current_time = time.time()
                if current_time - last_print_time >= 2.0:  # Every 2 seconds
                    fps = frame_count / (current_time - last_print_time)
                    print(f"FPS: {fps:6.1f} | Frame: {frame.FrameId:6d}")
                    frame_count = 0
                    last_print_time = current_time
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        LuMoSDKClient.Close()
        viewer.close()
        total_time = time.time() - start_time
        print(f"Session: {total_time:.1f}s")

if __name__ == '__main__':
    live_fk()
