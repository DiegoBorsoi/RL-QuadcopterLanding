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
#ifdef IGN_PROFILER_ENABLE
#include <ignition/common/Profiler.hh>
#endif
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

    /// Publish odometry transforms
    /// \param[in] _current_time Current simulation time
    void PublishOdometryTf(const gazebo::common::Time & _current_time);

    /// A pointer to the GazeboROS node.
    gazebo_ros::Node::SharedPtr ros_node_;

    /// Subscriber to command velocities
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    /// Odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

    /// To broadcast TF
    std::shared_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;

    /// Velocity received on command.
    geometry_msgs::msg::Twist target_cmd_vel_;

    // Vertical velocity to hover in place
    float hover_z_vel_;

    float max_rot_;

    float rot_step_;

    float rot_zero_margin_;

    float horizontal_vel_;

    float vertical_vel_;

    /// Keep latest odometry message
    nav_msgs::msg::Odometry odom_;

    /// Pointer to world.
    gazebo::physics::WorldPtr world_;

    /// Pointer to model.
    gazebo::physics::ModelPtr model_;

    // Pointer to body link
    gazebo::physics::LinkPtr link_;

    std::vector<gazebo::common::PID> controllers_;

    /// Connection to event called at every world iteration.
    gazebo::event::ConnectionPtr update_connection_;

    /// Protect variables accessed on callbacks.
    std::mutex lock_;

    /// Update period in seconds.
    double update_period_;

    /// Publish period in seconds.
    double publish_period_;

    /// Last update time.
    gazebo::common::Time last_update_time_;

    /// Last publish time.
    gazebo::common::Time last_publish_time_;

    /// Odometry frame ID
    std::string odometry_frame_;

    /// Robot base frame ID
    std::string robot_base_frame_;

    /// True to publish odometry messages.
    bool publish_odom_;

    /// True to publish odom-to-world transforms.
    bool publish_odom_tf_;
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
    std::unique_ptr<DronePluginPrivate> impl_;
  };

  DronePlugin::DronePlugin()
  : impl_(std::make_unique<DronePluginPrivate>())
  {
  }

  DronePlugin::~DronePlugin()
  {
  }

  void DronePlugin::Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    impl_->model_ = _model;

    impl_->link_ = impl_->model_->GetLink("link");

    impl_->world_ = _model->GetWorld();

    // Initialize ROS node
    impl_->ros_node_ = gazebo_ros::Node::Get(_sdf);

    // Get QoS profiles
    const gazebo_ros::QoS & qos = impl_->ros_node_->get_qos();

    // Odometry
    impl_->odometry_frame_ = _sdf->Get<std::string>("odometry_frame", "odom").first;
    impl_->robot_base_frame_ = _sdf->Get<std::string>("robot_base_frame", "base_footprint").first;

    // Update rate
    auto update_rate = _sdf->Get<double>("update_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl_->update_period_ = 1.0 / update_rate;
    } else {
      impl_->update_period_ = 0.0;
    }
    impl_->last_update_time_ = impl_->world_->SimTime();

    // Update rate
    auto publish_rate = _sdf->Get<double>("publish_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl_->publish_period_ = 1.0 / publish_rate;
    } else {
      impl_->publish_period_ = 0.0;
    }
    impl_->last_publish_time_ = impl_->world_->SimTime();

    impl_->cmd_vel_sub_ = impl_->ros_node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", qos.get_subscription_qos("cmd_vel", rclcpp::QoS(1)),
      std::bind(&DronePluginPrivate::OnCmdVel, impl_.get(), std::placeholders::_1));

    RCLCPP_INFO(
      impl_->ros_node_->get_logger(), "Subscribed to [%s]",
      impl_->cmd_vel_sub_->get_topic_name());

    // Advertise odometry topic
    impl_->publish_odom_ = _sdf->Get<bool>("publish_odom", true).first;
    if (impl_->publish_odom_) {
      impl_->odometry_pub_ = impl_->ros_node_->create_publisher<nav_msgs::msg::Odometry>(
        "odom", qos.get_publisher_qos("odom", rclcpp::QoS(1)));

      RCLCPP_INFO(
        impl_->ros_node_->get_logger(), "Advertise odometry on [%s]",
        impl_->odometry_pub_->get_topic_name());
    }

    // Broadcast TF
    impl_->publish_odom_tf_ = _sdf->Get<bool>("publish_odom_tf", true).first;
    if (impl_->publish_odom_tf_) {
      impl_->transform_broadcaster_ =
        std::make_shared<tf2_ros::TransformBroadcaster>(impl_->ros_node_);

      RCLCPP_INFO(
        impl_->ros_node_->get_logger(),
        "Publishing odom transforms between [%s] and [%s]", impl_->odometry_frame_.c_str(),
        impl_->robot_base_frame_.c_str());
    }

    auto covariance_x = _sdf->Get<double>("covariance_x", 0.00001).first;
    auto covariance_y = _sdf->Get<double>("covariance_y", 0.00001).first;
    auto covariance_z = _sdf->Get<double>("covariance_z", 0.00001).first;
    auto covariance_roll = _sdf->Get<double>("covariance_roll", 0.001).first;
    auto covariance_pitch = _sdf->Get<double>("covariance_pitch", 0.001).first;
    auto covariance_yaw = _sdf->Get<double>("covariance_yaw", 0.001).first;

    // Set covariance
    impl_->odom_.pose.covariance[0] = covariance_x;
    impl_->odom_.pose.covariance[7] = covariance_y;
    impl_->odom_.pose.covariance[14] = covariance_z;
    impl_->odom_.pose.covariance[21] = covariance_roll;
    impl_->odom_.pose.covariance[28] = covariance_pitch;
    impl_->odom_.pose.covariance[35] = covariance_yaw;

    impl_->odom_.twist.covariance[0] = covariance_x;
    impl_->odom_.twist.covariance[7] = covariance_y;
    impl_->odom_.twist.covariance[14] = covariance_z;
    impl_->odom_.twist.covariance[21] = covariance_roll;
    impl_->odom_.twist.covariance[28] = covariance_pitch;
    impl_->odom_.twist.covariance[35] = covariance_yaw;

    // Set header
    impl_->odom_.header.frame_id = impl_->odometry_frame_;
    impl_->odom_.child_frame_id = impl_->robot_base_frame_;

    /*
    // PID settings -----------------------------------------------------------------------------------------
    // Hard coded gains. Tune these for your own application!
    double linear_p = 80.0;
    double linear_i = 10.0;
    double linear_d = 10.0;
    double linear_imax = 123456789.0;

    double maxForce = 100;

    for(int i = 0; i < 1; i++){
      impl_->controllers_.push_back(
        gazebo::common::PID(
          linear_p, linear_i, linear_d, 
          linear_imax, -linear_imax, maxForce, -maxForce));
    }
    // ------------------------------------------------------------------------------------------------------
    */

    // Set the hover vertical velocity
    impl_->hover_z_vel_ = 0.294f;

    impl_->max_rot_ = (M_PI / 6);
    impl_->rot_step_ = impl_->max_rot_ / 50;

    impl_->rot_zero_margin_ = 0.005f;

    impl_->horizontal_vel_ = 0.1f;

    impl_->vertical_vel_ = 0.001f;


    // Listen to the update event (broadcast every simulation iteration)
    impl_->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&DronePluginPrivate::OnUpdate, impl_.get(), std::placeholders::_1));
  }

  void DronePlugin::Reset()
  {
    impl_->last_update_time_ = impl_->world_->SimTime();
    impl_->target_cmd_vel_.linear.x = 0;
    impl_->target_cmd_vel_.linear.y = 0;
    impl_->target_cmd_vel_.linear.z = 0;
    impl_->target_cmd_vel_.angular.x = 0;
    impl_->target_cmd_vel_.angular.y = 0;
    impl_->target_cmd_vel_.angular.z = 0;

    //impl_->controllers_.clear();
  }

  void DronePluginPrivate::OnUpdate(const gazebo::common::UpdateInfo & _info)
  {
    double seconds_since_last_update = (_info.simTime - last_update_time_).Double();

    std::lock_guard<std::mutex> scoped_lock(lock_);

  #ifdef IGN_PROFILER_ENABLE
    IGN_PROFILE("DronePluginPrivate::OnUpdate");
    IGN_PROFILE_BEGIN("fill ROS message");
  #endif

    if (seconds_since_last_update >= update_period_) {
      ignition::math::Pose3d pose = model_->WorldPose();
      //auto yaw = static_cast<float>(pose.Rot().Yaw());
      //std::cout << pose.Rot().Pitch() << std::endl;

      float y_ang_vel_ = 0;
      float current_pitch = static_cast<float>(pose.Rot().Pitch());
      current_pitch = (current_pitch > - rot_zero_margin_ && current_pitch < rot_zero_margin_) ? 0 : current_pitch;

      if (target_cmd_vel_.linear.x == 1.0f){
        if (current_pitch < max_rot_){
          y_ang_vel_ = rot_step_;
        }
      }else if (target_cmd_vel_.linear.x == -1.0f) {
        if (current_pitch > - max_rot_){
          y_ang_vel_ = - rot_step_;
        }
      }else { // (target_cmd_vel_.linear.x == 0)
        if (current_pitch > 0){
          y_ang_vel_ = - rot_step_;
        }
        else if (current_pitch < 0){
          y_ang_vel_ = rot_step_;
        }
      }

      float x_ang_vel_ = 0;
      float current_roll = - static_cast<float>(pose.Rot().Roll());
      current_roll = (current_roll > - rot_zero_margin_ && current_roll < rot_zero_margin_) ? 0 : current_roll;

      if (target_cmd_vel_.linear.y == 1.0f){
        if (current_roll < max_rot_){
          x_ang_vel_ = - rot_step_;
        }
      }else if (target_cmd_vel_.linear.y == -1.0f) {
        if (current_roll > - max_rot_){
          x_ang_vel_ = rot_step_;
        }
      }else { // (target_cmd_vel_.linear.y == 0)
        if (current_roll > 0){
          x_ang_vel_ = rot_step_;
        }
        else if (current_roll < 0){
          x_ang_vel_ = - rot_step_;
        }
      }

      float z_ang_vel_ = 0;
      float current_yaw = static_cast<float>(pose.Rot().Yaw());
      if (current_yaw > rot_zero_margin_){
        z_ang_vel_ = - rot_step_;
      }
      else if (current_yaw < - rot_zero_margin_){
        z_ang_vel_ = rot_step_;
      }

      float x_lin_vel_ = sinf(current_pitch) * horizontal_vel_;
      float y_lin_vel_ = sinf(current_roll) * horizontal_vel_;

      float z_lin_vel_ = hover_z_vel_;
      if (target_cmd_vel_.linear.z == 1.0f){
        z_lin_vel_ += vertical_vel_;
      }else if (target_cmd_vel_.linear.z == -1.0f){
        z_lin_vel_ -= vertical_vel_;
      }

      if (static_cast<float>(pose.Pos().Z()) < 0.51f){
        z_lin_vel_ = 0;
      }
      
      model_->SetLinearVel(
        ignition::math::Vector3d(
          x_lin_vel_,
          y_lin_vel_,
          z_lin_vel_));

      model_->SetAngularVel(
        ignition::math::Vector3d(
          x_ang_vel_, 
          y_ang_vel_, 
          z_ang_vel_));
      
      

      /*
      ignition::math::Vector3d wLinearVel = link_->WorldLinearVel();

      link_->AddForce(
        ignition::math::Vector3d(
          0,
          0,
          controllers_[0].Update(wLinearVel.Z() - target_cmd_vel_.linear.z, seconds_since_last_update)));
      */

      last_update_time_ = _info.simTime;
    }

  #ifdef IGN_PROFILER_ENABLE
    IGN_PROFILE_END();
  #endif

    if (publish_odom_ || publish_odom_tf_) {
      double seconds_since_last_publish = (_info.simTime - last_publish_time_).Double();

      if (seconds_since_last_publish < publish_period_) {
        return;
      }

  #ifdef IGN_PROFILER_ENABLE
      IGN_PROFILE_BEGIN("UpdateOdometry");
  #endif
      UpdateOdometry(_info.simTime);
  #ifdef IGN_PROFILER_ENABLE
      IGN_PROFILE_END();
  #endif

      if (publish_odom_) {
  #ifdef IGN_PROFILER_ENABLE
        IGN_PROFILE_BEGIN("publish odometry");
  #endif
        odometry_pub_->publish(odom_);
  #ifdef IGN_PROFILER_ENABLE
        IGN_PROFILE_END();
  #endif
      }

      if (publish_odom_tf_) {
  #ifdef IGN_PROFILER_ENABLE
        IGN_PROFILE_BEGIN("publish odometryTF");
  #endif
        PublishOdometryTf(_info.simTime);
  #ifdef IGN_PROFILER_ENABLE
        IGN_PROFILE_END();
  #endif
      }

      last_publish_time_ = _info.simTime;
    }
  }

  void DronePluginPrivate::OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr _msg)
  {
    std::lock_guard<std::mutex> scoped_lock(lock_);
    target_cmd_vel_ = *_msg;
  }

  void DronePluginPrivate::UpdateOdometry(const gazebo::common::Time & _current_time)
  {
    auto pose = model_->WorldPose();
    odom_.pose.pose = gazebo_ros::Convert<geometry_msgs::msg::Pose>(pose);

    // Get velocity in odom frame
    odom_.twist.twist.angular.z = model_->WorldAngularVel().Z();

    // Convert velocity to child_frame_id(aka base_footprint)
    auto linear = model_->WorldLinearVel();
    auto yaw = static_cast<float>(pose.Rot().Yaw());
    odom_.twist.twist.linear.x = cosf(yaw) * linear.X() + sinf(yaw) * linear.Y();
    odom_.twist.twist.linear.y = cosf(yaw) * linear.Y() - sinf(yaw) * linear.X();

    float z_vel_zero_clamp = 0.0001f;
    float z_vel = linear.Z() - hover_z_vel_;
    if (z_vel > -z_vel_zero_clamp && z_vel < z_vel_zero_clamp){
      z_vel = 0;
    }
    odom_.twist.twist.linear.z = z_vel;

    // Set timestamp
    odom_.header.stamp = gazebo_ros::Convert<builtin_interfaces::msg::Time>(_current_time);
  }

  void DronePluginPrivate::PublishOdometryTf(const gazebo::common::Time & _current_time)
  {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.stamp = gazebo_ros::Convert<builtin_interfaces::msg::Time>(_current_time);
    msg.header.frame_id = odometry_frame_;
    msg.child_frame_id = robot_base_frame_;
    msg.transform = gazebo_ros::Convert<geometry_msgs::msg::Transform>(odom_.pose.pose);

    transform_broadcaster_->sendTransform(msg);
  }

  GZ_REGISTER_MODEL_PLUGIN(DronePlugin)
}  // namespace gazebo_plugins