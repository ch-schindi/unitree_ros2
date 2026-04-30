#define _USE_MATH_DEFINES
#include <rclcpp/rclcpp.hpp>
#include <queue>
#include <cmath>
#include <string>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

struct RobotStep {
    std::string name;
    float x_vel;      // Forward/Backward
    float y_vel;      // Sideways
    float yaw_vel;    // Rotation
    double duration;  // Seconds
};

class PathRunner : public rclcpp::Node {
public:
    PathRunner() : Node("path_runner") {
        sport_client = std::make_unique<SportClient>(this);
        
        // Build the Square Path (1.5m at 0.5m/s = 3.0s per side)
        // Adjust yaw_vel and duration for your specific floor friction
        for (int i = 0; i < 4; ++i) {
            path_.push(create_walk_step("Forward" + std::to_string(i), 1.0f, 0.2f));
            path_.push(create_turn_step("Turn_90" + std::to_string(i), 90.0f, 0.5f));
        }

        // Start the control loop
        timer_ = this->create_wall_timer(100ms, std::bind(&PathRunner::execute_path, this));
        start_time_ = this->now();
        

        RCLCPP_INFO(this->get_logger(), "Starting path execution...");
        // RCLCPP_INFO(this->get_logger(), "Standing up...");
        // unitree_api::msg::Request req;
        // sport_client->StandUp(req);
    }

private:

    // Helper to calculate a walking step based on distance
    RobotStep create_walk_step(std::string name, float distance, float velocity) {
        double duration = std::abs(distance / velocity);
        return {name, velocity, 0.0f, 0.0f, duration};
    }

    // Helper to calculate a turning step based on degrees
    RobotStep create_turn_step(std::string name, float degrees, float angular_velocity) {
        float radians = degrees * (M_PI / 180.0f);
	float factor = 1.7f;
        double duration = std::abs((radians * factor) / angular_velocity);
        float direction = (degrees < 0) ? -1.0f : 1.0f;
        
        return {name, 0.0f, 0.0f, std::abs(angular_velocity) * direction, duration};
    }

    void execute_path() {
        if (path_.empty()) {
            unitree_api::msg::Request req;
            sport_client->StopMove(req);
            RCLCPP_INFO_ONCE(this->get_logger(), "Path complete. Holding position.");
            return;
        }

        auto& current_step = path_.front();
        double elapsed = (this->now() - start_time_).seconds();

        if (this->now().seconds() - start_time_.seconds() < 2.0) return; // Wait 2 seconds before starting

        if (elapsed < current_step.duration) {
            unitree_api::msg::Request req;
            sport_client->Move(req, current_step.x_vel, current_step.y_vel, current_step.yaw_vel);
        } else {
            RCLCPP_INFO(this->get_logger(), "Completed: %s", current_step.name.c_str());
            path_.pop();
            start_time_ = this->now(); // Reset clock for next step
        }
    }

    std::unique_ptr<SportClient> sport_client;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;
    std::queue<RobotStep> path_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathRunner>());
    rclcpp::shutdown();
    return 0;
}
