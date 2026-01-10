from setuptools import setup, find_packages

setup(
    name="mujoco_view_robot",
    version="1.0.0",
    description="Real-time FK visualization for G1 robot using LuMo mocap and Mujoco",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "mujoco",
        "mujoco-python-viewer",
        "scipy",
        "numpy",
        "pyyaml",
        "pyzmq",
        "protobuf<=3.20.0"
    ]
)
