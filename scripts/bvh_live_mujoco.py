import numpy as np
import time
import mujoco_viewer 
import mujoco
import os
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
import lumosdk.LuMoSDKClient as LuMoSDKClient
from utils.rot_utils import (
    bvh_yup_to_zup,
    rotate_yup_to_zup,
)
ip = "192.168.2.30"

LuMoSDKClient.Init()
LuMoSDKClient.Connnect(ip)

HUMAN_HEIGHT = 1.75  # meters
DEBUG = True
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

    # 1. Get skeleton data    
    target_skeleton = next((s for s in frame.skeletons if s.Name == "Skeleton0"), None)
    if target_skeleton is not None:
        # update metadata
        bones = target_skeleton.skeletonBones
        names = [] # reset names list
        for i, bone in enumerate(bones):
            bone_pos_raw[i] = [bone.X / POS_SCALE, -bone.Z / POS_SCALE, bone.Y / POS_SCALE]
            bone_quat_raw[i] = [bone.qx, bone.qy, bone.qz, bone.qw]
            names.append(bone.Name)

    # yup to zup conversion
    n = len(names)
    bone_quat_raw[:n] = rotate_yup_to_zup(bone_quat_raw[:n])
    
    frame_data = {
        name: [bone_pos_raw[i], bone_quat_raw[i]]
        for i, name in enumerate(names)
    }
    debug_print(frame_data)
    
    # 2. GMR retargeting and Mujoco visualization can be added here
    qpos = retargeter.retarget(frame_data)
    debug_print(qpos.shape)
    
    # 3. Render in Mujoco Viewer
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    viewer.render()
    
    # time.sleep(100)
    
LuMoSDKClient.Close()
