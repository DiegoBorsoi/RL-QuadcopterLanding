import os

import rclpy

from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def get_param_file(file_name: str):
    """Get file path for default paramater file."""
    rclpy.logging.get_logger('Launch File').info(file_name)
    return os.path.join(get_package_share_directory('drone_worker'), 'config', file_name)


def generate_launch_description():
    """Launch settings for gazebo"""
    os.environ["GAZEBO_PLUGIN_PATH"] = "/".join(get_package_share_directory("drone_plugins").split('/')[:-4]) + "/build/drone_plugins/"

    pkg_dir = get_package_share_directory('drone_worker')
    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world_file_name_stat = 'quadcopter_platf-stat.world'
    world_stat = os.path.join(pkg_dir, 'worlds', world_file_name_stat)

    world_file_name_mov2D = 'quadcopter_platf-moving_2D.world'
    world_mov2D = os.path.join(pkg_dir, 'worlds', world_file_name_mov2D)

    world_file_name_mov3D = 'quadcopter_platf-moving_3D.world'
    world_mov3D = os.path.join(pkg_dir, 'worlds', world_file_name_mov3D)

    """Launch file for training."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=[get_param_file('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'worker_ns',
            default_value=['dw'],
            description='The Worker node namespace.'
        ),
        DeclareLaunchArgument(
            'input_folder',
            default_value=['saves/'],
            description='Name of input folder for model or other values to be loaded'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value=['PPO'],
            description='Policy worker will use for training.'
        ),
        DeclareLaunchArgument(
            'open_gui',
            default_value=['True'],
            description='Condition to check for having the gui.'
        ),
        DeclareLaunchArgument(
            'moving_platform',
            default_value=['False'],
            description='Condition to check for having the moving platform (default stationary).'
        ),
        DeclareLaunchArgument(
            'dimensions',
            default_value=['3'],
            description='Number of dimensions in which the drone can operate (default 3).'
        ),
        DeclareLaunchArgument(
            'n_test',
            default_value=['10'],
            description='Number of test to execute.'
        ),
        Node(
            package='drone_worker',
            executable='net_tester',
            name='tester_node',
            namespace=LaunchConfiguration('worker_ns'),
            output='screen',
            arguments=[
                LaunchConfiguration('input_folder'),
                LaunchConfiguration('policy_type'),
                LaunchConfiguration('param_file'),
                LaunchConfiguration('dimensions'),
                LaunchConfiguration('n_test'),
            ]
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_stat,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 1']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_stat,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 2'
                                                    ' and ',
                                                    LaunchConfiguration('moving_platform'), ' == False']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_stat,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 3'
                                                    ' and ',
                                                    LaunchConfiguration('moving_platform'), ' == False']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_mov2D,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 2'
                                                    ' and ',
                                                    LaunchConfiguration('moving_platform')]))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_mov3D,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 3'
                                                    ' and ',
                                                    LaunchConfiguration('moving_platform')]))
        ),
        ExecuteProcess(
            cmd=['gzclient', '--verbose'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('open_gui')),
        )
    ])
