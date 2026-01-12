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
)
ip = "192.168.2.30"

LuMoSDKClient.Init()
LuMoSDKClient.Connnect(ip)

HUMAN_HEIGHT = 1.7  # meters
DEBUG = False  # 开启调试输出
DETAILED_LOG = False  # 开启详细日志
frame_count = 0  # 帧计数器
HEADLESS = False
SMOOTH_ENABLE = False

def debug_print(msg):
    if DEBUG:
        print(msg)

retargeter = GMR(
    src_human=f"bvh_fzmotion", 
    tgt_robot="unitree_g1", 
    actual_human_height=HUMAN_HEIGHT,
    solver="daqp",  # 使用DAQP求解器
    damping=5.0,  # 进一步增加阻尼提高稳定性
    verbose=False,
)

# 平衡性能和稳定性
retargeter.max_iter = 1  # 减少迭代，避免过拟合
print(f"IK配置: solver={retargeter.solver}, damping={retargeter.damping}, max_iter={retargeter.max_iter}")

mj_xml = os.path.join(os.path.dirname(__file__), '../robots/g1/g1_29dof_rev_1_0.xml')
mj_model = mujoco.MjModel.from_xml_path(mj_xml)
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

# 运动平滑参数
SMOOTH_ALPHA = 0.9  # 超强平滑（会有延迟）
qpos_prev = None  # 上一帧的qpos
qpos_smoothed = None  # 平滑后的qpos

POS_SCALE = 1000.0  # mm to m
N = 100
bone_pos_raw = np.empty((N, 3), dtype=np.float32)
bone_quat_raw = np.empty((N, 4), dtype=np.float32)
names = []
cnt = 0
cur_time = time.time()

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
    qpos, _, _ = retargeter.retarget(frame_data)
    
    if SMOOTH_ENABLE:
        # 运动平滑：使用指数移动平均减少抖动
        if qpos_smoothed is None:
            qpos_smoothed = qpos.copy()
        else:
            # 平滑公式: smoothed = alpha * current + (1-alpha) * previous
            qpos_smoothed = SMOOTH_ALPHA * qpos + (1 - SMOOTH_ALPHA) * qpos_smoothed
    
        # 检测异常值：如果qpos变化过大，可能是IK发散
        if qpos_prev is not None:
            qpos_diff = np.abs(qpos - qpos_prev)
            max_diff = np.max(qpos_diff)
            # 关节角度变化限制：极度保守10度/帧
            if max_diff > np.deg2rad(10):
                if DEBUG and frame_count % 30 == 0:
                    print(f"[警告] 检测到大幅度跳变 (最大变化: {np.rad2deg(max_diff):.1f}°)，限制变化幅度")
                # 限制变化幅度而非完全丢弃
                qpos = np.clip(qpos, qpos_prev - np.deg2rad(10), qpos_prev + np.deg2rad(10))
        if DETAILED_LOG and frame_count % 30 == 0:
            print(f"\n{'='*70}")
            print(f"7. GMR retarget最终输出 - 帧 {frame_count}")
            print(f"{'='*70}")
            print(f"   qpos shape: {qpos.shape}")
            print(f"   qpos前10个值: {qpos[:10]}")
            # 前3个通常是位置，第4-7个是四元数
            if len(qpos) >= 7:
                base_pos = qpos[:3]
                base_quat = qpos[3:7]
                base_euler = R.from_quat(base_quat).as_euler('xyz', degrees=True)
                print(f"   基座位置 (qpos[0:3]): {base_pos}")
                print(f"   基座四元数 (qpos[3:7]): {base_quat}")
                print(f"   基座欧拉角: [{base_euler[0]:6.1f}, {base_euler[1]:6.1f}, {base_euler[2]:6.1f}]")
            
            # 打印躯干相关的关节角度（根据G1机器人的关节顺序）
            # 通常腰部关节在索引7附近
            if len(qpos) > 10:
                print(f"\\n   关节角度 (qpos[7:]):")
                joint_names = ['waist_yaw', 'waist_roll', 'waist_pitch', 
                            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 
                            'left_knee', 'left_ankle_pitch', 'left_ankle_roll']
                for i, name in enumerate(joint_names):
                    if 7+i < len(qpos):
                        angle_deg = np.rad2deg(qpos[7+i])
                        print(f"     [{7+i}] {name:20s}: {qpos[7+i]:7.3f} rad ({angle_deg:7.1f}°)")
                        if i >= 8:  # 只打印前9个关节
                            break
    else:
        qpos_smoothed = qpos.copy()
    # 3. Render in Mujoco Viewer - 使用平滑后的数据
    if not HEADLESS:
        mj_data.qpos[:] = qpos_smoothed
        mujoco.mj_forward(mj_model, mj_data)
        viewer.render()
    else:
        # print performance analyze
        cnt += 1
        if cnt % 100 == 0:
            cost_time = time.time() - cur_time
            fps = 100/cost_time
            print(f"[INFO] {cnt=} {fps=}")
            cur_time = time.time()
    
    # time.sleep(100)
    
LuMoSDKClient.Close()
