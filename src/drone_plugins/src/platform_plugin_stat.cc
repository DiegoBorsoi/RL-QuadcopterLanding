#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/conversions/builtin_interfaces.hpp>

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

    /// Pointer to world.
    gazebo::physics::WorldPtr world_;

    /// Pointer to model.
    gazebo::physics::ModelPtr model_;

    /// Connection to event called at every world iteration.
    gazebo::event::ConnectionPtr update_connection_;

    /// Update period in seconds.
    double update_period_;

    /// Last update time.
    gazebo::common::Time last_update_time_;

    // rotation limit and increase/decrease step
    float max_rot;
    float rot_steps;

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

    impl_->world_ = _model->GetWorld();

    // Update rate
    auto update_rate = _sdf->Get<double>("update_rate", 20.0).first;
    if (update_rate > 0.0) {
      impl_->update_period_ = 1.0 / update_rate;
    } else {
      impl_->update_period_ = 0.0;
    }
    impl_->last_update_time_ = impl_->world_->SimTime();

    impl_->max_rot = (M_PI / 12);
    impl_->rot_steps = 500.0f;

    impl_->rot_zero_margin = impl_->max_rot / (impl_->rot_steps / 2);

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
  }

  void PlatformPluginPrivate::OnUpdate(const gazebo::common::UpdateInfo & _info)
  {
    double seconds_since_last_update = (_info.simTime - last_update_time_).Double();

    if (seconds_since_last_update >= update_period_) {
      ignition::math::Pose3d pose = model_->WorldPose();

      // keep the model with zero rotation
      float x_ang_vel = 0;
      float current_roll = static_cast<float>(pose.Rot().Roll());
      if (current_roll > rot_zero_margin){
        x_ang_vel = - ang_vel;
      }
      else if (current_roll < - rot_zero_margin){
        x_ang_vel = ang_vel;
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
  }

  GZ_REGISTER_MODEL_PLUGIN(PlatformPlugin)
}  // namespace gazebo_plugins