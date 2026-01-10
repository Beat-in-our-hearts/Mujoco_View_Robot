import numpy as np
import time
import mujoco_viewer 
import mujoco
import os
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
import lumosdk.LuMoSDKClient as LuMoSDKClient
from utils.rot_utils import (
    bvh_yup_to_zup_wxyz,
    rotate_yup_to_zup,
)
ip = "192.168.2.30"

LuMoSDKClient.Init()
LuMoSDKClient.Connnect(ip)

HUMAN_HEIGHT = 1.75  # meters
DEBUG = True
DETAILED_LOG = True  # 详细日志开关
frame_count = 0  # 帧计数器

def debug_print(msg):
    if DEBUG:
        print(msg)

retargeter = GMR(
    src_human=f"bvh_fzmotion", 
    tgt_robot="unitree_g1", 
    actual_human_height=HUMAN_HEIGHT,)

mj_xml = os.path.join(os.path.dirname(__file__), '../robots/g1/g1_29dof_rev_1_0.xml')
mj_model = mujoco.MjModel.from_xml_path(mj_xml)
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

POS_SCALE = 1000.0  # mm to m
N = 100
bone_pos_raw = np.empty((N, 3), dtype=np.float32)
bone_quat_raw = np.empty((N, 4), dtype=np.float32)
names = []

while True:
    frame = LuMoSDKClient.ReceiveData(0) # 0 :阻塞接收 1：非阻塞接收
    if frame is None:
        continue

    frame_count += 1

    # 1. Get skeleton data    
    target_skeleton = next((s for s in frame.skeletons if s.Name == "Skeleton0"), None)
    if target_skeleton is not None:
        # update metadata
        bones = target_skeleton.skeletonBones
        names = [] # reset names list
        
        # 保存原始数据用于日志
        bone_pos_original = np.empty((len(bones), 3), dtype=np.float32)
        bone_quat_original = np.empty((len(bones), 4), dtype=np.float32)
        
        for i, bone in enumerate(bones):
            # 原始坐标，不做转换
            bone_pos_raw[i] = [bone.X / POS_SCALE, bone.Y / POS_SCALE, bone.Z / POS_SCALE]
            bone_quat_raw[i] = [bone.qx, bone.qy, bone.qz, bone.qw]
            bone_pos_original[i] = bone_pos_raw[i]
            bone_quat_original[i] = bone_quat_raw[i]
            names.append(bone.Name)

    # yup to zup conversion
    n = len(names)
    
    # yup to zup conversion - 使用WXYZ格式输出（GMR期望的格式）
    bone_pos_raw[:n], bone_quat_raw[:n] = bvh_yup_to_zup_wxyz(bone_pos_raw[:n], bone_quat_raw[:n])
    
    # 详细日志：每30帧打印一次
    if DETAILED_LOG and frame_count % 30 == 0 and n > 0:
        # 找到Hips骨骼
        hips_idx = names.index("Hips") if "Hips" in names else 0
        print(f"\n{'='*70}")
        print(f"帧 {frame_count} - 骨骼 '{names[hips_idx]}' 数据追踪")
        print(f"{'='*70}")
        
        # 1. 原始数据
        euler_orig = R.from_quat(bone_quat_original[hips_idx]).as_euler('xyz', degrees=True)
        print(f"1. SDK原始数据 (Y-UP坐标系, XYZW格式):")
        print(f"   位置: {bone_pos_original[hips_idx]}")
        print(f"   四元数: {bone_quat_original[hips_idx]}")
        print(f"   欧拉角 (XYZ): [{euler_orig[0]:6.1f}, {euler_orig[1]:6.1f}, {euler_orig[2]:6.1f}]")
        
        # 2. 转换后数据（注意：现在是WXYZ格式）
        quat_wxyz = bone_quat_raw[hips_idx]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # 转回XYZW用于显示
        euler_converted = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
        print(f"\n2. 转换后数据 (Z-UP坐标系, WXYZ格式):")
        print(f"   位置: {bone_pos_raw[hips_idx]}")
        print(f"   四元数 (WXYZ): {quat_wxyz}")
        print(f"   欧拉角 (XYZ): [{euler_converted[0]:6.1f}, {euler_converted[1]:6.1f}, {euler_converted[2]:6.1f}]")
        
        # 判断主要旋转轴
        max_idx_orig = np.argmax(np.abs(euler_orig))
        max_idx_conv = np.argmax(np.abs(euler_converted))
        axis_names = ['X', 'Y', 'Z']
        print(f"\n   原始主要旋转轴: {axis_names[max_idx_orig]}轴 ({euler_orig[max_idx_orig]:.1f}度)")
        print(f"   转换后主要旋转轴: {axis_names[max_idx_conv]}轴 ({euler_converted[max_idx_conv]:.1f}度)")
    
    frame_data = {
        name: [bone_pos_raw[i], bone_quat_raw[i]]
        for i, name in enumerate(names)
    }
    
    # 2. GMR retargeting
    qpos = retargeter.retarget(frame_data)
    
    # 详细日志：GMR输出
    if DETAILED_LOG and frame_count % 30 == 0:
        print(f"\n3. GMR retarget输出:")
        print(f"   qpos shape: {qpos.shape}")
        print(f"   qpos前10个值: {qpos[:10]}")
        # 前3个通常是位置，第4-7个是四元数
        if len(qpos) >= 7:
            base_quat = qpos[3:7]
            base_euler = R.from_quat(base_quat).as_euler('xyz', degrees=True)
            print(f"   基座四元数 (qpos[3:7]): {base_quat}")
            print(f"   基座欧拉角: [{base_euler[0]:6.1f}, {base_euler[1]:6.1f}, {base_euler[2]:6.1f}]")
    
    # 3. Render in Mujoco Viewer
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    viewer.render()
    
    # time.sleep(100)
    
LuMoSDKClient.Close()
