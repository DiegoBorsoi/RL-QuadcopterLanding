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
#include <random>

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

    // target roll
    float target_roll;

    // rotation limit and increase/decrease step
    float max_rot;
    float rot_steps;

    // ccondition if we reached the target rotation
    bool reached_target_roll;

    // margin to keep the rotation stable on target
    float rot_zero_margin;

    float initial_x;
    float initial_y;
    float initial_z;

    float pos_margin;

    float ang_vel;
    float lin_vel;
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

    impl_->target_roll = 0;

    impl_->max_rot = (M_PI / 12);
    impl_->rot_steps = 500.0f;

    impl_->rot_zero_margin = impl_->max_rot / (impl_->rot_steps / 2);

    impl_->reached_target_roll = true;

    ignition::math::Pose3d pose = impl_->model_->WorldPose();
    impl_->initial_x = static_cast<float>(pose.Pos().X());
    impl_->initial_y = static_cast<float>(pose.Pos().Y());
    impl_->initial_z = static_cast<float>(pose.Pos().Z());

    impl_->pos_margin = 0.005f;

    impl_->ang_vel = 0.001f;
    impl_->lin_vel = 0.01f;


    // Listen to the update event (broadcast every simulation iteration)
    impl_->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&PlatformPluginPrivate::OnUpdate, impl_.get(), std::placeholders::_1));
  }

  void PlatformPlugin::Reset()
  {
    impl_->last_update_time_ = impl_->world_->SimTime();
    impl_->reached_target_roll = true;
  }

  void PlatformPluginPrivate::OnUpdate(const gazebo::common::UpdateInfo & _info)
  {
    double seconds_since_last_update = (_info.simTime - last_update_time_).Double();

    if (seconds_since_last_update >= update_period_) {
      ignition::math::Pose3d pose = model_->WorldPose();

      float current_roll = static_cast<float>(pose.Rot().Roll()); // on increase go down on y+

      //std::cout << "roll: " << target_roll << " - " << current_roll << std::endl;
      if(current_roll < (target_roll + rot_zero_margin) && current_roll > (target_roll - rot_zero_margin)){
        reached_target_roll = true;
      }
      
      float x_ang_vel = 0;
      // calculate new random rotation if we reach the previous one
      if(reached_target_roll){
        // generate new target rotations
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_real_distribution<float> fdistr(- max_rot, max_rot);

        target_roll = fdistr(gen);
        //std::cout << target_roll << std::endl;

        reached_target_roll = false;
      }else{
        // take a step to reach the target rotation
        if(! reached_target_roll){
          if(current_roll > target_roll){
            x_ang_vel = - abs(target_roll / rot_steps);
          }else if(current_roll < target_roll){
            x_ang_vel = abs(target_roll / rot_steps);
          }
        }
      }

      float y_ang_vel = 0;
      float current_pitch = static_cast<float>(pose.Rot().Pitch());
      if (current_pitch > rot_zero_margin){
        y_ang_vel = - ang_vel;
      }
      else if (current_pitch < - rot_zero_margin){
        y_ang_vel = ang_vel;
      }
      
      float z_ang_vel = 0;
      float current_yaw = static_cast<float>(pose.Rot().Yaw());
      if (current_yaw > rot_zero_margin){
        z_ang_vel = - ang_vel;
      }
      else if (current_yaw < - rot_zero_margin){
        z_ang_vel = ang_vel;
      }
      
      model_->SetAngularVel(
        ignition::math::Vector3d(
          x_ang_vel, 
          y_ang_vel, 
          z_ang_vel));


      // keep the model in the initial position
      float x_lin_vel = 0;
      float current_x = static_cast<float>(pose.Pos().X());
      if (current_x > initial_x + pos_margin){
        x_lin_vel = - lin_vel;
      }
      else if (current_x < initial_x - pos_margin){
        x_lin_vel = lin_vel;
      }

      float y_lin_vel = 0;
      float current_y = static_cast<float>(pose.Pos().Y());
      if (current_y > initial_y + pos_margin){
        y_lin_vel = - lin_vel;
      }
      else if (current_y < initial_y - pos_margin){
        y_lin_vel = lin_vel;
      }

      float z_lin_vel = 0;
      float current_z = static_cast<float>(pose.Pos().Z());
      if (current_z > initial_z + pos_margin){
        z_lin_vel = - lin_vel;
      }
      else if (current_z < initial_z - pos_margin){
        z_lin_vel = lin_vel;
      }

      model_->SetLinearVel(
        ignition::math::Vector3d(
          x_lin_vel, 
          y_lin_vel, 
          z_lin_vel));
      
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