import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cmm_env'


def get_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ]

    # Recursively collect all model files (SDF, configs, meshes)
    for root, dirs, files in os.walk('models'):
        for file in files:
            source_file = os.path.join(root, file)
            dest_dir = os.path.join('share', package_name, root)
            data_files.append((dest_dir, [source_file]))

    return data_files


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raghav',
    maintainer_email='raghav@todo.todo',
    description='CMM environment package for RL simulation with ROS2 and Gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_box = cmm_env.spawn_box:main',
            'camera_bridge = cmm_env.camera_bridge:main',
            'revolve_camera = cmm_env.revolve_camera:main',
        ],
    },
)
