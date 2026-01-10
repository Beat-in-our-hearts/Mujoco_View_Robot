import json
from scipy.spatial.transform import Rotation as R


file = "/Users/lzx/Documents/vscode/GMR/general_motion_retargeting/ik_configs/bvh_fzmotion_to_g1.json"
with open(file, 'r') as f:
    data = json.load(f)
    
ik_match_table1 = data['ik_match_table1']
ik_match_table2 = data['ik_match_table2']

rot_transform = R.from_quat([0.5, 0.5, 0.5, -0.5])  # XYZW format

new_ik_match_table1 = {}
for key, value in ik_match_table1.items():
    quaternion = value[4]
    quat_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]  # Convert to XYZW
    rot = R.from_quat(quat_xyzw)
    rot_new = rot_transform.inv() * rot
    quat_xyzw_new = rot_new.as_quat()
    quat_wxyz_new = [quat_xyzw_new[3], quat_xyzw_new[0], quat_xyzw_new[1], quat_xyzw_new[2]]  # Convert back
    new_ik_match_table1[key] = [
        value[0],
        value[1],
        value[2],
        value[3],
        quat_wxyz_new
    ]
    
new_ik_match_table2 = {}
for key, value in ik_match_table2.items():
    quaternion = value[4]
    quat_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]  # Convert to XYZW
    rot = R.from_quat(quat_xyzw)
    rot_new = rot_transform.inv() * rot
    quat_xyzw_new = rot_new.as_quat()
    quat_wxyz_new = [quat_xyzw_new[3], quat_xyzw_new[0], quat_xyzw_new[1], quat_xyzw_new[2]]  # Convert back
    new_ik_match_table2[key] = [
        value[0],
        value[1],
        value[2],
        value[3],
        quat_wxyz_new
    ]
    
data['ik_match_table1'] = new_ik_match_table1
data['ik_match_table2'] = new_ik_match_table2

out_file = file.replace(".json", "_fix.json")
with open(out_file, 'w') as f:
    json.dump(data, f, indent=4)
