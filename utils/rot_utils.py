import numpy as np
from scipy.spatial.transform import Rotation as R

# 全局变量用于调试
_debug_call_count = 0
_debug_enabled = False  # 关闭调试日志提升性能

def bvh_yup_to_zup(global_pos:np.ndarray, global_quat:np.ndarray):
    """
    Convert BVH Y-UP global quaternions and positions to Z-UP
    Args:
        global_quat: (N, 4) array of quaternions in XYZW format (Y-UP)
        global_pos: (N, 3) array of positions (Y-UP)
    """
    global _debug_call_count
    _debug_call_count += 1
    
    # 旋转矩阵: X→X, Y→Z, Z→-Y (调整符号以修复头朝下问题)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_transform = R.from_matrix(rotation_matrix)

    # 位置转换：直接应用旋转矩阵
    global_pos_zup = global_pos @ rotation_matrix.T
    
    # 四元数转换：对于全局坐标系，使用简单的旋转变换
    global_quat_yup = R.from_quat(global_quat)
    global_quat_zup = rotation_transform * global_quat_yup
    
    # 调试日志
    if _debug_enabled and _debug_call_count % 30 == 0 and len(global_quat) > 0:
        print(f"\n[rot_utils.bvh_yup_to_zup] 调用次数: {_debug_call_count}")
        print(f"  旋转矩阵:\n{rotation_matrix}")
        print(f"  输入数量: {len(global_quat)} 个骨骼")
        if len(global_quat) > 0:
            euler_before = R.from_quat(global_quat[0]).as_euler('xyz', degrees=True)
            euler_after = R.from_quat(global_quat_zup.as_quat()[0]).as_euler('xyz', degrees=True)
            print(f"  第一个骨骼转换:")
            print(f"    转换前欧拉角: [{euler_before[0]:6.1f}, {euler_before[1]:6.1f}, {euler_before[2]:6.1f}]")
            print(f"    转换后欧拉角: [{euler_after[0]:6.1f}, {euler_after[1]:6.1f}, {euler_after[2]:6.1f}]")
    
    return global_pos_zup, global_quat_zup.as_quat()  # XYZW format

def xyzw_to_wxyz(quat):
    """
    Convert quaternion from XYZW to WXYZ format
    Args:
        quat: (N, 4) or (4,) array of quaternion in XYZW format
    Returns:
        (N, 4) or (4,) array of quaternion in WXYZ format
    """
    if len(quat.shape) == 1:
        # Single quaternion
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        # Multiple quaternions
        return np.column_stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]])

def bvh_yup_to_zup_wxyz(global_pos:np.ndarray, global_quat:np.ndarray):
    """
    Convert BVH Y-UP global quaternions and positions to Z-UP with WXYZ output
    Args:
        global_quat: (N, 4) array of quaternions in XYZW format (Y-UP)
        global_pos: (N, 3) array of positions (Y-UP)
    Returns:
        global_pos_zup: (N, 3) array of positions (Z-UP)
        global_quat_zup: (N, 4) array of quaternions in WXYZ format (Z-UP)
    """
    pos_zup, quat_xyzw = bvh_yup_to_zup(global_pos, global_quat)
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    return pos_zup, quat_wxyz

# def rotate_yup_to_zup(quat:np.ndarray):
#     """
#     Rotate a quaternion from Y-UP to Z-UP
#     Args:
#         quat: (N, 4) or (4,) array of quaternion in XYZW format (Y-UP)
#     """
#     # 旋转矩阵: X→X, Y→Z, Z→-Y (和bvh_yup_to_zup一致)
#     rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
#     rotation_transform = R.from_matrix(rotation_matrix)
    
#     # 对于全局绝对数据，使用简单的旋转变换
#     quat_yup = R.from_quat(quat)
#     quat_zup = rotation_transform * quat_yup
#     return quat_zup.as_quat()  # XYZW format
    
def bvh_yup_to_zup_fast(global_pos: np.ndarray, global_quat: np.ndarray):
    R_mat = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1, 0]], dtype=np.float32)

    pos_zup = global_pos @ R_mat.T
    rot_transform = R.from_matrix(R_mat)
    quat_zup_wxyz = (rot_transform * R.from_quat(global_quat)).as_quat(scalar_first=True)
    return pos_zup, quat_zup_wxyz