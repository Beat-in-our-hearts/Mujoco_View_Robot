import numpy as np
from scipy.spatial.transform import Rotation as R

def bvh_yup_to_zup(global_pos:np.ndarray, global_quat:np.ndarray):
    """
    Convert BVH Y-UP global quaternions and positions to Z-UP
    Args:
        global_quat: (N, 4) array of quaternions in XYZW format (Y-UP)
        global_pos: (N, 3) array of positions (Y-UP)
    """
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_transform = R.from_matrix(rotation_matrix)

    global_quat_yup = R.from_quat(global_quat)
    global_quat_zup = rotation_transform * global_quat_yup
    
    global_pos_zup = global_pos @ rotation_matrix.T
    
    return global_pos_zup, global_quat_zup.as_quat()  # XYZW format

def rotate_yup_to_zup(quat:np.ndarray):
    """
    Rotate a quaternion from Y-UP to Z-UP
    Args:
        quat: (4,) array of quaternion in XYZW format (Y-UP)
    """
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation = R.from_matrix(rotation_matrix)
    
    axis_rotation = R.from_euler('x', -90, degrees=True)
    rot = R.from_quat(quat)
    rot_new = axis_rotation * rotation * rot * rotation.inv() * axis_rotation.inv()
    return rot_new.as_quat()  # XYZW format
    