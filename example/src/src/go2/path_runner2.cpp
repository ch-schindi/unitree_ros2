#define _USE_MATH_DEFINES
#include <rclcpp/rclcpp.hpp>
#include <queue>
#include <cmath>
#include <string>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

struct RobotStep2 {
    std::string name;
    float x_vel;      // Forward/Backward
    float y_vel;      // Sideways
    float yaw_vel;    // Rotation
    double target_distance;
    double target_rotation;
    bool is_turn;
    bool is_wait;
    double wait_duration;
};

class PoseAwarePathRunner : public rclcpp::Node {
public:
    PoseAwarePathRunner() : Node("pose_aware_path_runner") {
        sport_client = std::make_unique<SportClient>(this);

        // Build the Square Path (1.5m at 0.5m/s = 3.0s per side)
        // Adjust yaw_vel and duration for your specific floor friction
        for (int i = 0; i < 4; ++i) {
            path_.push(create_wait_step("Pause", 10.0));
	    path_.push(create_walk_step("Forward" + std::to_string(i), 2.0f, 0.3f));
	    path_.push(create_wait_step("Pause", 10.0));
	    path_.push(create_walk_step("Forward" + std::to_string(i), 2.0f, 0.3f));
	    path_.push(create_wait_step("Pause", 10.0));
            path_.push(create_turn_step("Turn_90" + std::to_string(i), -90.0f, 0.5f));
	    path_.push(create_walk_step("Forward" + std::to_string(i), 0.5f, 0.3f));
	    path_.push(create_turn_step("Turn_90" + std::to_string(i), -90.0f, 0.5f));
	    path_.push(create_wait_step("Pause", 10.0));
	    path_.push(create_walk_step("Forward" + std::to_string(i), 2.0f, 0.3f));
	    path_.push(create_wait_step("Pause", 10.0));
	    path_.push(create_walk_step("Forward" + std::to_string(i), 2.0f, 0.3f));
	    path_.push(create_wait_step("Pause", 10.0));
	    path_.push(create_turn_step("Turn_90" + std::to_string(i), 90.0f, 0.5f));
            path_.push(create_walk_step("Forward" + std::to_string(i), 0.5f, 0.3f));
            path_.push(create_turn_step("Turn_90" + std::to_string(i), 90.0f, 0.5f));
        }

        // Start the control loop
        timer_ = this->create_wall_timer(50ms, std::bind(&PoseAwarePathRunner::execute_path, this));
        start_time_ = this->now();

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/utlidar/robot_pose", 10, std::bind(&PoseAwarePathRunner::pose_callback, this, std::placeholders::_1));


        RCLCPP_INFO(this->get_logger(), "Starting path execution...");
        // RCLCPP_INFO(this->get_logger(), "Standing up...");
        // unitree_api::msg::Request req;
        // sport_client->StandUp(req);
    }

private:

    // Helper to calculate a walking step based on distance
    RobotStep2 create_walk_step(std::string name, float distance, float velocity) {
        float direction = (distance < 0.0f) ? -1.0f : 1.0f;
        return {name, std::abs(velocity) * direction, 0.0f, 0.0f, std::abs(distance), 0.0, false, false, 0.0};
    }

    // Helper to calculate a turning step based on degrees
    RobotStep2 create_turn_step(std::string name, float degrees, float angular_velocity) {
        float radians = degrees * (M_PI / 180.0f);
        float direction = (degrees < 0) ? -1.0f : 1.0f;

        return {name, 0.0f, 0.0f, std::abs(angular_velocity) * direction, 0.0, std::abs(radians), true, false, 0.0};
    }

    RobotStep2 create_wait_step(std::string name, double seconds) {
        return {name, 0.0f, 0.0f, 0.0f, 0.0, 0.0, false, true, seconds};
    }

    static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q) {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    static double normalize_angle(double angle) {
        while (angle > M_PI) {
            angle -= 2.0 * M_PI;
        }
        while (angle < -M_PI) {
            angle += 2.0 * M_PI;
        }
        return angle;
    }

    bool step_reached(const RobotStep2& step) const {
      if (step.is_turn) {
            const double start_yaw = quaternion_to_yaw(step_start_pose_.pose.orientation);
            const double current_yaw = quaternion_to_yaw(latest_pose_.pose.orientation);
            const double delta_yaw = normalize_angle(current_yaw - start_yaw);
            return std::abs(delta_yaw) >= step.target_rotation;
        }

        const double dx = latest_pose_.pose.position.x - step_start_pose_.pose.position.x;
        const double dy = latest_pose_.pose.position.y - step_start_pose_.pose.position.y;
        const double distance = std::sqrt(dx * dx + dy * dy);
        return distance >= step.target_distance;
    }

    void start_step_if_needed() {
        if (step_initialized_ || path_.empty()) {
            return;
        }

        step_start_pose_ = latest_pose_;
	step_start_time_ = this->now();
        step_initialized_ = true;
    }

    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        latest_pose_ = *msg;
    }

    void execute_path() {
        if (path_.empty()) {
            unitree_api::msg::Request req;
            sport_client->StopMove(req);
            RCLCPP_INFO_ONCE(this->get_logger(), "Path complete. Holding position.");
            return;
        }

        if ((this->now() - start_time_).seconds()  < 2.0) return; // Wait 2 seconds before starting

        start_step_if_needed();

        auto& current_step = path_.front();

	if (current_step.is_wait) {
	    unitree_api::msg::Request req;
    	    sport_client->StopMove(req);

    	    if ((this->now() - step_start_time_).seconds() >= current_step.wait_duration) {
            path_.pop();
            step_initialized_ = false;
    	    }
    	    return;
	}

        
        if (step_reached(current_step)) {
            unitree_api::msg::Request req;
            sport_client->StopMove(req);
            path_.pop();
            step_initialized_ = false;
            return;
        }

        unitree_api::msg::Request req;
        sport_client->Move(req, current_step.x_vel, current_step.y_vel, current_step.yaw_vel);
    }

    std::unique_ptr<SportClient> sport_client;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;
    std::queue<RobotStep2> path_;
    geometry_msgs::msg::PoseStamped latest_pose_;
    geometry_msgs::msg::PoseStamped step_start_pose_;
    rclcpp::Time step_start_time_;
    bool step_initialized_ = false;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseAwarePathRunner>());
    rclcpp::shutdown();
    return 0;
}
