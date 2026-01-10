"""
live for mocap data from mujoco and send to robot

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
args = argparser.parse_args()

# G1 Joint names in Mujoco order (29 DOF)
G1_JOINT_NAMES = [
    # Left Leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right Leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left Arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right Arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def load_joint_mapping(config_path):
    """Load joint mapping from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['Motor_Joint_Map']

def extract_joint_angles(skeleton, joint_mapping):
    """Extract joint angles from skeleton.MotorAngle using the mapping"""
    joint_angles = []
    missing_joints = []
    for mujoco_name in G1_JOINT_NAMES:
        lumo_name = joint_mapping.get(mujoco_name, None)
        if lumo_name and lumo_name in skeleton.MotorAngle:
            joint_angles.append(skeleton.MotorAngle[lumo_name])
        else:
            joint_angles.append(0.0)  # Default to 0 if missing
            if lumo_name:
                missing_joints.append(lumo_name)
    
    if missing_joints:
        print(f"Warning: Missing joint data for: {missing_joints[:3]}..." if len(missing_joints) > 3 else f"Warning: Missing joint data for: {missing_joints}")
    
    return np.array(joint_angles)

def extract_base_pose(skeleton, verbose=False):
    """Extract base position and orientation from skeleton bones"""
    # Try to find hips/hip bone
    hips_bone = None
    
    # First, try to find hips specifically
    for bone in skeleton.skeletonBones:
        if bone.Name.lower() == 'hips':
            hips_bone = bone
            if verbose:
                print(f"Found hips bone: '{bone.Name}'")
            break
    
    # If not found, search for hip
    if hips_bone is None:
        for bone in skeleton.skeletonBones:
            bone_name_lower = bone.Name.lower()
            if 'hips' in bone_name_lower or 'hip' in bone_name_lower:
                hips_bone = bone
                if verbose:
                    print(f"Found base bone: '{bone.Name}'")
                break
    
    if hips_bone is None and len(skeleton.skeletonBones) > 0:
        # Fallback to first bone and list all available bones
        hips_bone = skeleton.skeletonBones[0]
        print(f"Warning: Could not find hips/hip bone. Using first bone '{hips_bone.Name}' as base")
        print(f"Available bones: {[bone.Name for bone in skeleton.skeletonBones[:5]]}")  # Show first 5
    
    if hips_bone is None:
        return None, None
    
    # Extract position (in mm) and convert to meters
    # LuMo coordinate system: Y-UP (X-right, Y-up, Z-forward)
    pos_mm = np.array([hips_bone.X, hips_bone.Y, hips_bone.Z])
    
    # Convert from Y-UP to Z-UP coordinate system and mm to meters
    # Y-UP: (X, Y, Z) -> Z-UP: (X, Z, Y)
    base_pos = np.array([pos_mm[0], pos_mm[2], pos_mm[1]]) / 1000.0  # mm to meters
    
    # Extract quaternion
    quat_xyzw_yup = np.array([hips_bone.qx, hips_bone.qy, hips_bone.qz, hips_bone.qw])
    
    # Convert quaternion from Y-UP to Z-UP coordinate system
    # For Y-UP to Z-UP: swap Y and Z components
    quat_xyzw_zup = np.array([quat_xyzw_yup[0], quat_xyzw_yup[2], quat_xyzw_yup[1], quat_xyzw_yup[3]])
    
    # Convert to Mujoco format [w,x,y,z]
    base_quat = quat_xyzw_zup[[3, 0, 1, 2]]  # [x,y,z,w] -> [w,x,y,z]
    
    if verbose:
        print(f"Hips '{hips_bone.Name}' raw position (mm, Y-UP): [{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}]")
        print(f"Hips '{hips_bone.Name}' converted position (m, Z-UP): [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        print(f"Hips '{hips_bone.Name}' raw quaternion (xyzw, Y-UP): [{quat_xyzw_yup[0]:.3f}, {quat_xyzw_yup[1]:.3f}, {quat_xyzw_yup[2]:.3f}, {quat_xyzw_yup[3]:.3f}]")
        print(f"Hips '{hips_bone.Name}' converted quaternion (wxyz, Z-UP): [{base_quat[0]:.3f}, {base_quat[1]:.3f}, {base_quat[2]:.3f}, {base_quat[3]:.3f}]")
    
    return base_pos, base_quat

def live_fk():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, args.xml_path)
    config_path = os.path.join(script_dir, args.config)
    
    # Load joint mapping
    print(f"Loading joint mapping from {config_path}")
    joint_mapping = load_joint_mapping(config_path)
    print(f"Loaded {len(joint_mapping)} joint mappings")
    
    # Load Mujoco model
    print(f"Loading Mujoco model from {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print(f"Model loaded: {model.nq} DOF total (7 base + 29 joints)")
    
    # Initialize viewer
    first_frame = True  # Flag to print detailed info for first frame
    print("Initializing Mujoco viewer...")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Set camera parameters
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    
    # Connect to LuMoSDK
    print(f"Connecting to LuMo mocap system at {args.ip}...")
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)
    print("Connected successfully!")
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    print("\n" + "="*60)
    print("Starting FK visualization loop...")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    try:
        while viewer.is_alive:
            # Receive data (non-blocking)
            frame = LuMoSDKClient.ReceiveData(1)
            
            if frame is None:
                # No data available, just render current state
                viewer.render()
                time.sleep(0.001)
                continue
            
            # Check if we have skeleton data
            if len(frame.skeletons) == 0:
                viewer.render()
                continue
            
            skeleton = frame.skeletons[0]
            
            # Check if skeleton is being tracked
            if not skeleton.IsTrack:
                viewer.render()
                continue
            
            # Extract base pose (verbose for first frame)
            base_pos, base_quat = extract_base_pose(skeleton, verbose=first_frame)
            if base_pos is None:
                viewer.render()
                continue
            
            # Extract joint angles using mapping
            joint_angles = extract_joint_angles(skeleton, joint_mapping)
            
            # Print detailed info for first successful frame
            if first_frame:
                print(f"\nFirst frame data:")
                print(f"  Skeleton ID: {skeleton.Id}, Name: {skeleton.Name}")
                print(f"  Total bones: {len(skeleton.skeletonBones)}")
                print(f"  Motor angles available: {len(skeleton.MotorAngle)}")
                print(f"  Base position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
                print(f"  Base quaternion: [{base_quat[0]:.3f}, {base_quat[1]:.3f}, {base_quat[2]:.3f}, {base_quat[3]:.3f}]")
                print()
                first_frame = False
            
            # Update Mujoco state
            # Base position and orientation (first 7 elements of qpos)
            data.qpos[0:3] = base_pos
            data.qpos[3:7] = base_quat
            
            # Joint angles (remaining 29 elements)
            data.qpos[7:36] = joint_angles
            
            # Perform forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update camera to follow robot
            viewer.cam.lookat[:] = base_pos
            
            # Render
            viewer.render()
            
            # Update frame count
            frame_count += 1
            
            # Print stats every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                fps = frame_count / (current_time - last_print_time)
                print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:5.1f} | Skeleton: {skeleton.Name:20s} | Frame: {frame.FrameId:6d} | Joints: {len(skeleton.MotorAngle):2d}/29")
                frame_count = 0
                last_print_time = current_time
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Shutting down...")
        print("="*60)
    
    finally:
        # Cleanup
        LuMoSDKClient.Close()
        viewer.close()
        total_time = time.time() - start_time
        print(f"Session duration: {total_time:.1f}s")
        print("Disconnected from LuMo mocap system")

if __name__ == '__main__':
    live_fk()

