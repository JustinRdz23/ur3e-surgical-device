#include <chrono>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <moveit_msgs/srv/servo_command_type.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <stdio.h>
#ifndef WIN32
#include <termios.h>
#include <unistd.h>
#else
#include <conio.h>
#endif

// Define used keys
namespace
{
constexpr int8_t KEYCODE_1 = 0x31;
constexpr int8_t KEYCODE_2 = 0x32;
constexpr int8_t KEYCODE_3 = 0x33;
constexpr int8_t KEYCODE_4 = 0x34;
constexpr int8_t KEYCODE_5 = 0x35;
constexpr int8_t KEYCODE_6 = 0x36;
constexpr int8_t KEYCODE_7 = 0x37;
constexpr int8_t KEYCODE_R = 0x72;
constexpr int8_t KEYCODE_J = 0x6A;
constexpr int8_t KEYCODE_T = 0x74;

// qewasd keys
constexpr int8_t KEYCODE_Q = 0x71;
constexpr int8_t KEYCODE_E = 0x65;
constexpr int8_t KEYCODE_W = 0x77;
constexpr int8_t KEYCODE_A = 0x61;
constexpr int8_t KEYCODE_S = 0x73;
constexpr int8_t KEYCODE_D = 0x64;

constexpr int8_t KEYCODE_SPACE = 0x20; 
}  // namespace

namespace
{
const std::string TWIST_TOPIC = "/servo_node/delta_twist_cmds";
const std::string JOINT_TOPIC = "/servo_node/delta_joint_cmds";
const size_t ROS_QUEUE_SIZE = 10;
const std::string PLANNING_FRAME_ID = "base_link";
const std::string EE_FRAME_ID = "tool0";

constexpr double LINEAR_SPEED = 1;
// Safety Timeout window: Stop moving if no terminal repeat signal within 200ms
const rclcpp::Duration DEAD_MAN_TIMEOUT = rclcpp::Duration::from_seconds(0.2);
}  // namespace

class KeyboardReader
{
public:
  KeyboardReader() : file_descriptor_(0)
  {
#ifndef WIN32
    tcgetattr(file_descriptor_, &cooked_);
    struct termios raw;
    memcpy(&raw, &cooked_, sizeof(struct termios));
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VEOL] = 1;
    raw.c_cc[VEOF] = 2;
    tcsetattr(file_descriptor_, TCSANOW, &raw);
#endif
  }
  void readOne(char* c)
  {
#ifndef WIN32
    int rc = read(file_descriptor_, c, 1);
    if (rc < 0)
    {
      throw std::runtime_error("read failed");
    }
#else
    *c = static_cast<char>(_getch());
#endif
  }
  void shutdown()
  {
#ifndef WIN32
    tcsetattr(file_descriptor_, TCSANOW, &cooked_);
#endif
  }

private:
  int file_descriptor_;
#ifndef WIN32
  struct termios cooked_;
#endif
};

class KeyboardServo
{
public:
  KeyboardServo();
  int keyLoop();

private:
  void spin();
  void timerCallback();

  rclcpp::Node::SharedPtr nh_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr joint_pub_;
  rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<moveit_msgs::srv::ServoCommandType::Request> request_;
  double joint_vel_cmd_;
  std::string command_frame_id_;

  double twist_x_ = 0.0;
  double twist_y_ = 0.0;
  double twist_z_ = 0.0;
  std::vector<double> joint_vels_;

  // TIMESTAMPS FOR DEAD-MAN'S TIMEOUT
  rclcpp::Time last_x_time_;
  rclcpp::Time last_y_time_;
  rclcpp::Time last_z_time_;
  std::vector<rclcpp::Time> last_joint_times_;
};

KeyboardServo::KeyboardServo() : joint_vel_cmd_(1.0), command_frame_id_{ "base_link" }
{
  nh_ = rclcpp::Node::make_shared("servo_keyboard_input");

  twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
  joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(JOINT_TOPIC, ROS_QUEUE_SIZE);
  switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");

  joint_vels_.resize(6, 0.0);
  
  // Initialize timestamps
  rclcpp::Time epoch = nh_->now();
  last_x_time_ = epoch;
  last_y_time_ = epoch;
  last_z_time_ = epoch;
  last_joint_times_.resize(6, epoch);

  timer_ = nh_->create_wall_timer(std::chrono::milliseconds(20), std::bind(&KeyboardServo::timerCallback, this));
}

KeyboardReader input;

void quit(int sig)
{
  (void)sig;
  input.shutdown();
  rclcpp::shutdown();
  exit(0);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  KeyboardServo keyboard_servo;

  signal(SIGINT, quit);

  int rc = keyboard_servo.keyLoop();
  input.shutdown();
  rclcpp::shutdown();

  return rc;
}

void KeyboardServo::spin()
{
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(nh_);
  while (rclcpp::ok())
  {
    executor.spin_some();
  }
}

void KeyboardServo::timerCallback()
{
  rclcpp::Time now = nh_->now();

  // Evaluate Dead-Man's Timeouts per Axis
  if (now - last_x_time_ > DEAD_MAN_TIMEOUT) twist_x_ = 0.0;
  if (now - last_y_time_ > DEAD_MAN_TIMEOUT) twist_y_ = 0.0;
  if (now - last_z_time_ > DEAD_MAN_TIMEOUT) twist_z_ = 0.0;

  for (size_t i = 0; i < joint_vels_.size(); ++i)
  {
    if (now - last_joint_times_[i] > DEAD_MAN_TIMEOUT) joint_vels_[i] = 0.0;
  }

  // 1. Handle Simultaneous Twist Outputs
  if (twist_x_ != 0.0 || twist_y_ != 0.0 || twist_z_ != 0.0)
  {
    auto twist_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    twist_msg->header.stamp = now;
    twist_msg->header.frame_id = command_frame_id_;
    twist_msg->twist.linear.x = twist_x_;
    twist_msg->twist.linear.y = twist_y_;
    twist_msg->twist.linear.z = twist_z_;
    twist_pub_->publish(std::move(twist_msg));
  }

  // 2. Handle Simultaneous Joint Outputs
  bool any_joint_active = false;
  for(double v : joint_vels) { if (v != 0.0) any_joint_active = true; }

  if (any_joint_active)
  {
    auto joint_msg = std::make_unique<control_msgs::msg::JointJog>();
    joint_msg->header.stamp = now;
    joint_msg->header.frame_id = PLANNING_FRAME_ID;
    joint_msg->joint_names = { "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint" };
    joint_msg->velocities = joint_vels_;
    joint_pub_->publish(std::move(joint_msg));
  }
}

int KeyboardServo::keyLoop()
{
  char c;
  std::thread{ [this]() { return spin(); } }.detach();

  puts("Reading from keyboard");
  puts("---------------------------");
  puts("Hold keys down to move. Release keys to auto-stop (Dead-man logic).");
  puts("Use w/s (X), a/d (Y), and q/e (Z) keys to Cartesian jog");
  puts("Use 1|2|3|4|5|6|7 keys to joint jog. 'r' to reverse joint direction.");

  for (;;)
  {
    try
    {
      input.readOne(&c);
    }
    catch (const std::runtime_error&)
    {
      perror("read():");
      return -1;
    }

    rclcpp::Time click_time = nh_->now();

    switch (c)
    {
      case KEYCODE_W: twist_x_ = LINEAR_SPEED; last_x_time_ = click_time; break;
      case KEYCODE_S: twist_x_ = -LINEAR_SPEED; last_x_time_ = click_time; break;
      case KEYCODE_A: twist_y_ = -LINEAR_SPEED; last_y_time_ = click_time; break;
      case KEYCODE_D: twist_y_ = LINEAR_SPEED; last_y_time_ = click_time; break;
      case KEYCODE_Q: twist_z_ = -LINEAR_SPEED; last_z_time_ = click_time; break;
      case KEYCODE_E: twist_z_ = LINEAR_SPEED; last_z_time_ = click_time; break;

      case KEYCODE_1: joint_vels_[0] = joint_vel_cmd_; last_joint_times_[0] = click_time; break;
      case KEYCODE_2: joint_vels_[1] = joint_vel_cmd_; last_joint_times_[1] = click_time; break;
      case KEYCODE_3: joint_vels_[2] = joint_vel_cmd_; last_joint_times_[2] = click_time; break;
      case KEYCODE_4: joint_vels_[3] = joint_vel_cmd_; last_joint_times_[3] = click_time; break;
      case KEYCODE_5: joint_vels_[4] = joint_vel_cmd_; last_joint_times_[4] = click_time; break;
      case KEYCODE_6: joint_vels_[5] = joint_vel_cmd_; last_joint_times_[5] = click_time; break;
      
      case KEYCODE_R:
        joint_vel_cmd_ *= -1;
        break;

      case KEYCODE_SPACE:
        twist_x_ = 0.0; twist_y_ = 0.0; twist_z_ = 0.0;
        std::fill(joint_vels_.begin(), joint_vels_.end(), 0.0);
        break;

      case KEYCODE_J:
        request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
        request_->command_type = moveit_msgs::srv::ServoCommandType::Request::JOINT_JOG;
        if (switch_input_->wait_for_service(std::chrono::seconds(1))) {
          auto result = switch_input_->async_send_request(request_);
        }
        break;
      case KEYCODE_T:
        request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
        request_->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
        if (switch_input_->wait_for_service(std::chrono::seconds(1))) {
          auto result = switch_input_->async_send_request(request_);
        }
        break;
    }
  }
  return 0;
}