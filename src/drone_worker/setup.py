from glob import glob
import os
from setuptools import setup

package_name = 'drone_worker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '/policy', package_name + '/utils'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models/quadrotor_v2'), glob('models/quadrotor_v2/*')),
        (os.path.join('share', package_name, 'models/platform'), glob('models/platform/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='diego@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_drone = drone_worker.worker:main'
        ],
    },
)
