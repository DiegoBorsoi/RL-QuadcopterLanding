#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/common/PID.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/conversions/builtin_interfaces.hpp>
#include <gazebo_ros/conversions/geometry_msgs.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <string>

namespace gazebo_plugins
{
  class DronePluginPrivate
  {
  public:
    /// Callback to be called at every simulation iteration.
    /// \param[in] _info Updated simulation info.
    void OnUpdate(const gazebo::common::UpdateInfo & _info);

    /// Callback when a velocity command is received.
    /// \param[in] _msg Twist command message.
    void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr _msg);

    /// Update odometry.
    /// \param[in] _current_time Current simulation time
    void UpdateOdometry(const gazebo::common::Time & _current_time);

    /// A pointer to the GazeboROS node.
    gazebo_ros::Node::SharedPtr ros_node;

    /// Subscriber to command velocities
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub;

    /// Odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub;

    /// Velocity received on command.
    geometry_msgs::msg::Twist target_cmd_vel;

    // Vertical velocity to hover in place
    float hover_z_vel;

    // Max angular rotation
    float max_rot;

    // Rotation velocity
    float rot_vel;

    // Margin value for which the rotation value is treated as 0
    float rot_zero_margin;

    // Current vertical velocity
    float current_vertical_vel;

    // Max vertical velocity (beyond hover_z_vel)
    float vertical_vel_max;

    // Rate at which the vertical velocity is changed
    float vertical_step;

    // Max translation force
    double max_force;

    // Queue used to store the last translations forces
    std::queue<double> last_x_force;
    std::queue<double> last_y_force;

    /// Keep latest odometry message
    nav_msgs::msg::Odometry odom;

    /// Pointer to world.
    gazebo::physics::WorldPtr world;

    /// Pointer to model.
    gazebo::physics::ModelPtr model;

    // Pointer to body link
    gazebo::physics::LinkPtr link;

    /// Connection to event called at every world iteration.
    gazebo::event::ConnectionPtr update_connection;

    /// Protect variables accessed on callbacks.
    std::mutex lock;

    /// Update period in seconds.
    double update_period;

    /// Publish period in seconds.
    double publish_period;

    /// Last update time.
    gazebo::common::Time last_update_time;

    /// Last publish time.
    gazebo::common::Time last_publish_time;

    /// Odometry frame ID
    std::string odometry_frame;

    /// Robot base frame ID
    std::string robot_base_frame;

    /// True to publish odometry messages.
    bool publish_odom;
  };

  class DronePlugin : public gazebo::ModelPlugin
  {
  public:
    /// Constructor
    DronePlugin();

    /// Destructor
    ~DronePlugin();

  protected:
    // Documentation inherited
    void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override;

    // Documentation inherited
    void Reset() override;

  private:
    /// Private data pointer
    std::unique_ptr<DronePluginPrivate> impl;
  };

  DronePlugin::DronePlugin()
  : impl(std::make_unique<DronePluginPrivate>())
  {
  }

  DronePlugin::~DronePlugin()
  {
  }

  void DronePlugin::Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    impl->model = _model;

    impl->link = impl->model->GetLink("link");

    impl->world = _model->GetWorld();

    // Initialize ROS node
    impl->ros_node = gazebo_ros::Node::Get(_sdf);

    // Get QoS profiles
    const gazebo_ros::QoS & qos = impl->ros_node->get_qos();

    // Odometry
    impl->odometry_frame = _sdf->Get<std::string>("odometry_frame", "odom").first;
    impl->robot_base_frame = _sdf->Get<std::string>("robot_base_frame", "base_footprint").first;

    // Update rate
    auto update_rate = _sdf->Get<double>("update_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl->update_period = 1.0 / update_rate;
    } else {
      impl->update_period = 0.0;
    }
    impl->last_update_time = impl->world->SimTime();
    printf("update_period= %f ---------------------------------------------\n", impl->update_period);

    // Publish rate
    auto publish_rate = _sdf->Get<double>("publish_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl->publish_period = 1.0 / publish_rate;
    } else {
      impl->publish_period = 0.0;
    }
    impl->last_publish_time = impl->world->SimTime();

    // Create subscription for velocity command
    impl->cmd_vel_sub = impl->ros_node->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", qos.get_subscription_qos("cmd_vel", rclcpp::QoS(1)),
      std::bind(&DronePluginPrivate::OnCmdVel, impl.get(), std::placeholders::_1));


    // Advertise odometry topic
    impl->publish_odom = _sdf->Get<bool>("publish_odom", true).first;
    if (impl->publish_odom) {
      impl->odometry_pub = impl->ros_node->create_publisher<nav_msgs::msg::Odometry>(
        "odom", qos.get_publisher_qos("odom", rclcpp::QoS(1)));
    }

    auto covariance_x = _sdf->Get<double>("covariance_x", 0.00001).first;
    auto covariance_y = _sdf->Get<double>("covariance_y", 0.00001).first;
    auto covariance_z = _sdf->Get<double>("covariance_z", 0.00001).first;
    auto covariance_roll = _sdf->Get<double>("covariance_roll", 0.001).first;
    auto covariance_pitch = _sdf->Get<double>("covariance_pitch", 0.001).first;
    auto covariance_yaw = _sdf->Get<double>("covariance_yaw", 0.001).first;

    // Set covariance
    impl->odom.pose.covariance[0] = covariance_x;
    impl->odom.pose.covariance[7] = covariance_y;
    impl->odom.pose.covariance[14] = covariance_z;
    impl->odom.pose.covariance[21] = covariance_roll;
    impl->odom.pose.covariance[28] = covariance_pitch;
    impl->odom.pose.covariance[35] = covariance_yaw;

    impl->odom.twist.covariance[0] = covariance_x;
    impl->odom.twist.covariance[7] = covariance_y;
    impl->odom.twist.covariance[14] = covariance_z;
    impl->odom.twist.covariance[21] = covariance_roll;
    impl->odom.twist.covariance[28] = covariance_pitch;
    impl->odom.twist.covariance[35] = covariance_yaw;

    // Set header
    impl->odom.header.frame_id = impl->odometry_frame;
    impl->odom.child_frame_id = impl->robot_base_frame;
    

    // Set the vertical hover velocity
    impl->hover_z_vel = 0.294f;//9999997615814f;

    impl->max_rot = (M_PI / 6);
    impl->rot_vel = 1.5f;

    impl->rot_zero_margin = 0.005f;

    impl->current_vertical_vel = impl->hover_z_vel;
    impl->vertical_vel_max = 0.75f;
    impl->vertical_step = impl->vertical_vel_max / 10;

    impl->max_force = 200.0;

    for(unsigned i=0; i<3; i++){
      impl->last_x_force.push(0.0);
      impl->last_y_force.push(0.0);
    }

    // Listen to the update event (broadcast every simulation iteration)
    impl->update_connection = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&DronePluginPrivate::OnUpdate, impl.get(), std::placeholders::_1));
  }

  void DronePlugin::Reset()
  {
    impl->last_update_time = impl->world->SimTime();
    impl->target_cmd_vel.linear.x = 0;
    impl->target_cmd_vel.linear.y = 0;
    impl->target_cmd_vel.linear.z = 0;
    impl->target_cmd_vel.angular.x = 0;
    impl->target_cmd_vel.angular.y = 0;
    impl->target_cmd_vel.angular.z = 0;

    impl->current_vertical_vel = impl->hover_z_vel;

    for(unsigned i=0; i<3; i++){
      impl->last_x_force.push(0.0);
      impl->last_y_force.push(0.0);
    }
  }

  void DronePluginPrivate::OnUpdate(const gazebo::common::UpdateInfo & _info)
  {
    double seconds_since_last_update = (_info.simTime - last_update_time).Double();

    std::lock_guard<std::mutex> scoped_lock(lock);

    if (seconds_since_last_update >= update_period) {

      ignition::math::Pose3d pose = model->WorldPose();

      //---------------------------------------------------------------------------------------
      // Set vertical velocity to maintain hover, adjusting to the input-----------------------
      //---------------------------------------------------------------------------------------
      if (target_cmd_vel.linear.z == 1.0f){
        if (current_vertical_vel < hover_z_vel + vertical_vel_max){
          current_vertical_vel += vertical_step;
        }
      }else if (target_cmd_vel.linear.z == -1.0f){
        if (current_vertical_vel > hover_z_vel - vertical_vel_max){
          current_vertical_vel -= vertical_step;
        }
      }else if (target_cmd_vel.linear.z == 0){
        if (current_vertical_vel < hover_z_vel){
          current_vertical_vel += vertical_step;
        }else if (current_vertical_vel > hover_z_vel){
          current_vertical_vel -= vertical_step;
        }
      }

      if (static_cast<float>(pose.Pos().Z()) < 0.1f){
        current_vertical_vel = 0;
      }

      link->SetLinearVel(ignition::math::Vector3d(0, 0, current_vertical_vel));
      //---------------------------------------------------------------------------------------

      // Calculate pitch velocity (y angular velocity)
      float y_ang_vel_ = 0;
      float current_pitch = static_cast<float>(pose.Rot().Pitch());
      current_pitch = (current_pitch > - rot_zero_margin && current_pitch < rot_zero_margin) ? 0 : current_pitch;

      if (target_cmd_vel.linear.x == 1.0f){
        if (current_pitch < max_rot){
          y_ang_vel_ = rot_vel;
        }
      }else if (target_cmd_vel.linear.x == -1.0f) {
        if (current_pitch > - max_rot){
          y_ang_vel_ = - rot_vel;
        }
      }else { // (target_cmd_vel.linear.x == 0)
        if (current_pitch > 0){
          y_ang_vel_ = - rot_vel;
        }
        else if (current_pitch < 0){
          y_ang_vel_ = rot_vel;
        }
      }

      // Calculate roll velocity (x angular velocity)
      float x_ang_vel_ = 0;
      float current_roll = - static_cast<float>(pose.Rot().Roll());
      current_roll = (current_roll > - rot_zero_margin && current_roll < rot_zero_margin) ? 0 : current_roll;

      if (target_cmd_vel.linear.y == 1.0f){
        if (current_roll < max_rot){
          x_ang_vel_ = - rot_vel;
        }
      }else if (target_cmd_vel.linear.y == -1.0f) {
        if (current_roll > - max_rot){
          x_ang_vel_ = rot_vel;
        }
      }else { // (target_cmd_vel.linear.y == 0)
        if (current_roll > 0){
          x_ang_vel_ = rot_vel;
        }
        else if (current_roll < 0){
          x_ang_vel_ = - rot_vel;
        }
      }

      // Calculate yaw velocity (z angular velocity)
      // The yaw of the model is kept at 0
      float z_ang_vel_ = 0;
      float current_yaw = static_cast<float>(pose.Rot().Yaw());
      if (current_yaw > rot_zero_margin){
        z_ang_vel_ = - rot_vel;
      }
      else if (current_yaw < - rot_zero_margin){
        z_ang_vel_ = rot_vel;
      }

      // Set angular velocity
      link->SetAngularVel(
        ignition::math::Vector3d(
          x_ang_vel_, 
          y_ang_vel_, 
          z_ang_vel_));

      // Calculate translation forces
      double x_force = 0.0;
      double y_force = 0.0;

      ignition::math::Vector3d link_rot = link->WorldPose().Rot().Euler();

      x_force = last_x_force.front();
      last_x_force.pop();
      last_x_force.push((sinf(link_rot.Y()) / sinf(max_rot)) * max_force);

      y_force = last_y_force.front();
      last_y_force.pop();
      last_y_force.push((-sinf(link_rot.X()) / sinf(max_rot)) * max_force);

      // Apply translation forces
      link->AddForce(ignition::math::Vector3d(x_force, y_force, 0.0));

      last_update_time = _info.simTime;
    }

    // Publish the odometry if necessary
    if (publish_odom) {
      double seconds_since_last_publish = (_info.simTime - last_publish_time).Double();

      if (seconds_since_last_publish < publish_period) {
        return;
      }

      UpdateOdometry(_info.simTime);

      if (publish_odom) {
        odometry_pub->publish(odom);
      }

      last_publish_time = _info.simTime;
    }
  }

  void DronePluginPrivate::OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr _msg)
  {
    std::lock_guard<std::mutex> scoped_lock(lock);
    target_cmd_vel = *_msg;
  }

  void DronePluginPrivate::UpdateOdometry(const gazebo::common::Time & _current_time)
  {
    auto pose = model->WorldPose();
    odom.pose.pose = gazebo_ros::Convert<geometry_msgs::msg::Pose>(pose);

    // Get velocity in odom frame
    odom.twist.twist.angular.z = model->WorldAngularVel().Z();

    // Convert velocity to child_frame_id(aka base_footprint)
    auto linear = model->WorldLinearVel();
    auto yaw = static_cast<float>(pose.Rot().Yaw());
    odom.twist.twist.linear.x = cosf(yaw) * linear.X() + sinf(yaw) * linear.Y();
    odom.twist.twist.linear.y = cosf(yaw) * linear.Y() - sinf(yaw) * linear.X();

    float z_vel_zero_clamp = 0.0001f;
    float z_vel = linear.Z() - hover_z_vel;
    if (z_vel > -z_vel_zero_clamp && z_vel < z_vel_zero_clamp){
      z_vel = 0;
    }
    odom.twist.twist.linear.z = z_vel;

    // Set timestamp
    odom.header.stamp = gazebo_ros::Convert<builtin_interfaces::msg::Time>(_current_time);
  }

  GZ_REGISTER_MODEL_PLUGIN(DronePlugin)
}  // namespace gazebo_plugins