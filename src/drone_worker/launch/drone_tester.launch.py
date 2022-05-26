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

    world_file_name_2D_stat = 'quadcopter_platf-2D_stat.world'
    world_2D_stat = os.path.join(pkg_dir, 'worlds', world_file_name_2D_stat)

    world_file_name_2D_moving = 'quadcopter_platf-2D_moving.world'
    world_2D_moving = os.path.join(pkg_dir, 'worlds', world_file_name_2D_moving)

    world_file_name_3D_stat = 'quadcopter_platf-3D_stat.world'
    world_3D_stat = os.path.join(pkg_dir, 'worlds', world_file_name_3D_stat)

    world_file_name_3D_moving = 'quadcopter_platf-3D_moving.world'
    world_3D_moving = os.path.join(pkg_dir, 'worlds', world_file_name_3D_moving)

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
            'platform_type',
            default_value=['0'],
            description='Type of platform: 0 -> stationary (default), 1 -> moves to a rot and stay until reset, 2 -> moves continuosly.'
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
        DeclareLaunchArgument(
            'output_folder',
            default_value=['saves/'],
            description='Name of output folder for position data to be saved'
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
        Node(
            package='drone_worker',
            executable='pos_saver',
            name='pos_saver',
            namespace=LaunchConfiguration('worker_ns'),
            output='screen',
            arguments=[
                LaunchConfiguration('output_folder')
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
                                                    LaunchConfiguration('platform_type'), ' == 0']))
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
                                                    LaunchConfiguration('platform_type'), ' == 0']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_2D_stat,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 2'
                                                    ' and ',
                                                    LaunchConfiguration('platform_type'), ' == 1']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_2D_moving,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 2'
                                                    ' and ',
                                                    LaunchConfiguration('platform_type'), ' == 2']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_3D_stat,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 3'
                                                    ' and ',
                                                    LaunchConfiguration('platform_type'), ' == 1']))
        ),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_3D_moving,
                             '-s', 'libgazebo_ros_init.so',
                             '-s', 'libgazebo_ros_factory.so',
                             #'-s', 'libgazebo_ros_force_system.so',
                             '--pause'],
            output='screen',
            condition=IfCondition(PythonExpression([LaunchConfiguration('dimensions'), ' == 3'
                                                    ' and ',
                                                    LaunchConfiguration('platform_type'), ' == 2']))
        ),
        ExecuteProcess(
            cmd=['gzclient', '--verbose'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('open_gui')),
        )
    ])
