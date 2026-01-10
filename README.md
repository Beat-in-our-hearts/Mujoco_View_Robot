# Mujoco View Robot

Real-time forward kinematics visualization for Unitree G1 robot using LuMo motion capture system and Mujoco physics engine.

## Features

- Real-time FK visualization of G1 robot (29 DOF)
- LuMo mocap data integration
- Automatic joint mapping from mocap to Mujoco
- Live rendering with camera tracking

## Installation

### Option 1: Install as package (Recommended)

```bash
cd Mujoco_View_Robot
pip install -e .
```

### Option 2: Install dependencies only

```bash
cd Mujoco_View_Robot
pip install -r requirements.txt
```

### Option 3: Run installation script (macOS/Linux)

```bash
cd Mujoco_View_Robot
chmod +x install.sh
./install.sh
```

## Usage

### Run live FK visualization

```bash
cd scripts
python live_mujoco.py --ip 192.168.2.30
```

Or if installed as package:

```bash
live-fk --ip 192.168.2.30
```

### Command-line options

```bash
python live_mujoco.py --help
```

- `--ip`: IP address of LuMo mocap system (default: 192.168.2.30)
- `--xml_path`: Path to G1 Mujoco XML file (default: ../robots/g1/g1_29dof_rev_1_0.xml)
- `--config`: Path to G1 config file (default: ../config/g1.yaml)

## Configuration

### Joint Mapping

Edit `config/g1.yaml` to customize the mapping between LuMo motor names and Mujoco joint names:

```yaml
Motor_Joint_Map:
  left_hip_pitch_joint: L_LEG_HIP_PITCH
  left_hip_roll_joint: L_LEG_HIP_ROLL
  # ... more mappings
```

### Robot Models

Available models in `robots/g1/`:
- `g1_12dof.xml` - Lower body only (12 DOF)
- `g1_23dof_rev_1_0.xml` - Simplified arms (23 DOF)
- `g1_29dof_rev_1_0.xml` - Full body (29 DOF) ⭐ Default

## Project Structure

```
Mujoco_View_Robot/
├── setup.py              # Package installation configuration
├── requirements.txt      # Python dependencies
├── install.sh           # Installation script
├── README.md            # This file
├── lumosdk/             # LuMo SDK client library
│   ├── __init__.py
│   ├── LuMoSDKClient.py
│   └── LusterFrameStruct_pb2.py
├── scripts/             # Main scripts
│   ├── __init__.py
│   └── live_mujoco.py   # Live FK visualization
├── config/              # Configuration files
│   └── g1.yaml          # G1 joint mapping
└── robots/              # Robot models
    └── g1/              # G1 robot models
        ├── g1_29dof_rev_1_0.xml
        └── meshes/
```

## Requirements

- Python 3.8+
- mujoco >= 3.0.0
- numpy >= 1.20.0
- pyyaml >= 6.0
- pyzmq >= 25.0.0
- protobuf >= 4.0.0

## Troubleshooting

### Module not found error

If you get `ModuleNotFoundError: No module named 'lumosdk'`, install the package:

```bash
pip install -e .
```

### Connection error

Make sure:
1. LuMo mocap system is running
2. IP address is correct
3. Port 6868 is accessible

### No skeleton data

Check:
1. Skeleton is being tracked in LuMo system
2. Robot name matches in mocap software
3. MotorAngle data is being broadcast

## License

See LICENSE file for details.
