// Copyright 2018 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gazebo/common/Plugin.hh>
#include <boost/make_shared.hpp>
#include <gazebo/transport/transport.hh>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <gazebo_ros/conversions/builtin_interfaces.hpp>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <string>
#include <algorithm>
#include <limits>
#include <memory>

namespace gazebo_plugins
{

  class GazeboRosRaySensorPrivate
  {
  public:
    /// Node for ROS communication.
    gazebo_ros::Node::SharedPtr ros_node_;

    // Aliases
    using LaserScan = sensor_msgs::msg::LaserScan;
    using LaserScanPub = rclcpp::Publisher<LaserScan>::SharedPtr;

    /// Publisher of output
    LaserScanPub pub_;

    /// TF frame output is published in
    std::string frame_name_;

    void SubscribeGazeboLaserScan();

    /// Publish a sensor_msgs/LaserScan message from a gazebo laser scan
    void PublishLaserScan(ConstLaserScanStampedPtr & _msg);

    /// Gazebo transport topic to subscribe to for laser scan
    std::string sensor_topic_;

    /// Minimum intensity value to publish for laser scan / pointcloud messages
    double min_intensity_{0.0};

    /// Gazebo node used to subscribe to laser scan
    gazebo::transport::NodePtr gazebo_node_;

    /// Gazebo subscribe to parent sensor's laser scan
    gazebo::transport::SubscriberPtr laser_scan_sub_;

  private:
    // Convert gazebo laser_scan to ROS laser_scan
    LaserScan Convert(const gazebo::msgs::LaserScanStamped & in, double min_intensity);
  };

  class GazeboRosRaySensor : public gazebo::SensorPlugin
  {
  public:
    /// \brief Constructor
    GazeboRosRaySensor();

    /// \brief Destructor
    virtual ~GazeboRosRaySensor();

    // Documentation Inherited
    void Load(gazebo::sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

  private:
    std::unique_ptr<GazeboRosRaySensorPrivate> impl_;
  };

  GazeboRosRaySensor::GazeboRosRaySensor()
  : impl_(std::make_unique<GazeboRosRaySensorPrivate>())
  {
  }

  GazeboRosRaySensor::~GazeboRosRaySensor()
  {
    /*// Must release subscriber and then call fini on node to remove it from topic manager.
    impl_->laser_scan_sub_.reset();
    if (impl_->gazebo_node_) {
      impl_->gazebo_node_->Fini();
    }
    impl_->gazebo_node_.reset();*/
  }

  void GazeboRosRaySensor::Load(gazebo::sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
  {
    // Create ros_node configured from sdf
    impl_->ros_node_ = gazebo_ros::Node::Get(_sdf);

    // Get QoS profiles
    const gazebo_ros::QoS & qos = impl_->ros_node_->get_qos();

    // Get QoS profile for the publisher
    rclcpp::QoS pub_qos = qos.get_publisher_qos("~/out", rclcpp::SensorDataQoS().reliable());

    // Get tf frame for output
    impl_->frame_name_ = gazebo_ros::SensorFrameID(*_sensor, *_sdf);

    impl_->pub_ = impl_->ros_node_->create_publisher<sensor_msgs::msg::LaserScan>("~/out", pub_qos);

    if (!_sdf->HasElement("min_intensity")) {
      RCLCPP_DEBUG(
        impl_->ros_node_->get_logger(), "missing <min_intensity>, defaults to %f",
        impl_->min_intensity_);
    } else {
      impl_->min_intensity_ = _sdf->Get<double>("min_intensity");
    }

    // Create gazebo transport node and subscribe to sensor's laser scan
    impl_->gazebo_node_ = boost::make_shared<gazebo::transport::Node>();
    impl_->gazebo_node_->Init(_sensor->WorldName());

    // TODO(ironmig): use lazy publisher to only process laser data when output has a subscriber
    impl_->sensor_topic_ = _sensor->Topic();
    impl_->SubscribeGazeboLaserScan();
  }

  void GazeboRosRaySensorPrivate::SubscribeGazeboLaserScan()
  {
    // Subscribe to gazebo's laserscan
    laser_scan_sub_ = gazebo_node_->Subscribe(sensor_topic_, &GazeboRosRaySensorPrivate::PublishLaserScan, this);
  }

  void GazeboRosRaySensorPrivate::PublishLaserScan(ConstLaserScanStampedPtr & _msg)
  {
    // Convert Laser scan to ROS LaserScan
    auto ls = this->Convert(*_msg, 0.0);

    // Set tf frame
    ls.header.frame_id = frame_name_;
    // Publish output
    boost::get<LaserScanPub>(pub_)->publish(ls);
  }

  sensor_msgs::msg::LaserScan GazeboRosRaySensorPrivate::Convert(const gazebo::msgs::LaserScanStamped & in, double min_intensity)
  {
    auto count = in.scan().count();
    auto vertical_count = in.scan().vertical_count();

    sensor_msgs::msg::LaserScan ls;

    ls.ranges.resize(count * vertical_count);

    ls.header.stamp = gazebo_ros::Convert<builtin_interfaces::msg::Time>(in.time());
    ls.angle_min = in.scan().angle_min();
    ls.angle_max = in.scan().angle_max();
    ls.angle_increment = in.scan().angle_step();
    ls.time_increment = 0;
    ls.scan_time = 0;
    ls.range_min = in.scan().range_min();
    ls.range_max = in.scan().range_max();

    // If there are multiple vertical beams, use the one in the middle
    //size_t start = (vertical_count / 2) * count;

    // Copy ranges into ROS message
    std::copy(
      in.scan().ranges().begin(),
      in.scan().ranges().end(),
      ls.ranges.begin());
/*
    // Copy intensities into ROS message, clipping at min_intensity
    ls.intensities.resize(count);
    std::transform(
      in.scan().intensities().begin() + start,
      in.scan().intensities().begin() + start + count,
      ls.intensities.begin(), [min_intensity](double i) -> double {
        return i > min_intensity ? i : min_intensity;
      });
*/
    return ls;
  }

  // Register this plugin with the simulator
  GZ_REGISTER_SENSOR_PLUGIN(GazeboRosRaySensor)

}  // namespace gazebo_plugins