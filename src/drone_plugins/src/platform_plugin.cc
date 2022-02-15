#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/conversions/builtin_interfaces.hpp>
#include <gazebo_ros/conversions/geometry_msgs.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

namespace gazebo_plugins
{
  class PlatformPluginPrivate
  {
  public:
    /// Callback to be called at every simulation iteration.
    /// \param[in] _info Updated simulation info.
    void OnUpdate(const gazebo::common::UpdateInfo & _info);

    /// Callback when a velocity command is received.
    /// \param[in] _msg Twist command message.
    //void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr _msg);

    /// Update odometry.
    /// \param[in] _current_time Current simulation time
    void UpdateOdometry(const gazebo::common::Time & _current_time);

    /// Publish odometry transforms
    /// \param[in] _current_time Current simulation time
    //void PublishOdometryTf(const gazebo::common::Time & _current_time);

    /// A pointer to the GazeboROS node.
    gazebo_ros::Node::SharedPtr ros_node_;

    /// Subscriber to command velocities
    //rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    /// Odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

    /// To broadcast TF
    //std::shared_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;

    /// Velocity received on command.
    //geometry_msgs::msg::Twist target_cmd_vel_;
/*
    // Vertical velocity to hover in place
    float hover_z_vel_;

    float max_rot_;

    float rot_step_;

    float rot_zero_margin_;

    float horizontal_vel_;

    float vertical_vel_;
*/
    /// Keep latest odometry message
    nav_msgs::msg::Odometry odom_;

    /// Pointer to world.
    gazebo::physics::WorldPtr world_;

    /// Pointer to model.
    gazebo::physics::ModelPtr model_;

    // Pointer to body link
    //gazebo::physics::LinkPtr link_;

    /// Connection to event called at every world iteration.
    gazebo::event::ConnectionPtr update_connection_;

    /// Protect variables accessed on callbacks.
    //std::mutex lock_;

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
    //std::string robot_base_frame_;

    /// True to publish odometry messages.
    bool publish_odom_;

    /// True to publish odom-to-world transforms.
    //bool publish_odom_tf_;
  };

  class PlatformPlugin : public gazebo::ModelPlugin
  {
  public:
    /// Constructor
    PlatformPlugin();

    /// Destructor
    ~PlatformPlugin();

  protected:
    // Documentation inherited
    void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override;

    // Documentation inherited
    void Reset() override;

  private:
    /// Private data pointer
    std::unique_ptr<PlatformPluginPrivate> impl_;
  };

  PlatformPlugin::PlatformPlugin()
  : impl_(std::make_unique<PlatformPluginPrivate>())
  {
  }

  PlatformPlugin::~PlatformPlugin()
  {
  }

  void PlatformPlugin::Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    impl_->model_ = _model;

    //impl_->link_ = impl_->model_->GetLink("link");

    impl_->world_ = _model->GetWorld();

    // Initialize ROS node
    impl_->ros_node_ = gazebo_ros::Node::Get(_sdf);

    // Get QoS profiles
    const gazebo_ros::QoS & qos = impl_->ros_node_->get_qos();

    // Odometry
    impl_->odometry_frame_ = _sdf->Get<std::string>("odometry_frame", "odom").first;
    //impl_->robot_base_frame_ = _sdf->Get<std::string>("robot_base_frame", "base_footprint").first;

    // Update rate
    auto update_rate = _sdf->Get<double>("update_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl_->update_period_ = 1.0 / update_rate;
    } else {
      impl_->update_period_ = 0.0;
    }
    impl_->last_update_time_ = impl_->world_->SimTime();

    // Publish rate
    auto publish_rate = _sdf->Get<double>("publish_rate", 20.0).first;
    if (publish_rate > 0.0) {
      impl_->publish_period_ = 1.0 / publish_rate;
    } else {
      impl_->publish_period_ = 0.0;
    }
    impl_->last_publish_time_ = impl_->world_->SimTime();

    // Advertise odometry topic
    impl_->publish_odom_ = _sdf->Get<bool>("publish_odom", true).first;
    if (impl_->publish_odom_) {
      impl_->odometry_pub_ = impl_->ros_node_->create_publisher<nav_msgs::msg::Odometry>(
        "platform_odom", qos.get_publisher_qos("odom", rclcpp::QoS(1)));

      RCLCPP_INFO(
        impl_->ros_node_->get_logger(), "Advertise platform odometry on [%s]",
        impl_->odometry_pub_->get_topic_name());
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
    //impl_->odom_.child_frame_id = impl_->robot_base_frame_;


    // Listen to the update event (broadcast every simulation iteration)
    impl_->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&PlatformPluginPrivate::OnUpdate, impl_.get(), std::placeholders::_1));
  }

  void PlatformPlugin::Reset()
  {
    impl_->last_update_time_ = impl_->world_->SimTime();
  }

  void PlatformPluginPrivate::OnUpdate(const gazebo::common::UpdateInfo & _info)
  {
    double seconds_since_last_update = (_info.simTime - last_update_time_).Double();

    if (seconds_since_last_update >= update_period_) {
      ignition::math::Pose3d pose = model_->WorldPose();
      
      

      last_update_time_ = _info.simTime;
    }



    if (publish_odom_) {
      double seconds_since_last_publish = (_info.simTime - last_publish_time_).Double();

      if (seconds_since_last_publish < publish_period_) {
        return;
      }

      UpdateOdometry(_info.simTime);

      if (publish_odom_) {
        odometry_pub_->publish(odom_);
      }

      last_publish_time_ = _info.simTime;
    }
  }

  void PlatformPluginPrivate::UpdateOdometry(const gazebo::common::Time & _current_time)
  {
    auto pose = model_->WorldPose();
    odom_.pose.pose = gazebo_ros::Convert<geometry_msgs::msg::Pose>(pose);

    // Set timestamp
    odom_.header.stamp = gazebo_ros::Convert<builtin_interfaces::msg::Time>(_current_time);
  }

  GZ_REGISTER_MODEL_PLUGIN(PlatformPlugin)
}  // namespace gazebo_plugins