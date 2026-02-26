#include <rclcpp/rclcpp.hpp>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h" // Helper from unitree_ros2/example/src/common

class RobotMover : public rclcpp::Node {
public:
    RobotMover() : Node("robot_mover") {
        sport_client = std::make_unique<SportClient>(this);
        
        // 1. Comment out or remove the Timer! 
        // We don't want a loop for a one-time "StandUp" command.
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), 
            std::bind(&RobotMover::timer_callback, this)
        );
        
	/*
        // 2. Call StandUp directly after a short delay (to ensure publishers are ready)
        // Use a "One-Shot" timer: Wait 500ms, then call send_stand_up ONCE
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), 
            std::bind(&RobotMover::send_stand_up, this)
        );*/
    }

private:
    void send_stand_up() {
	// stop timer
	timer_->cancel();
	
        unitree_api::msg::Request req;
        // The helper fills the API ID 1004 and empty parameter for you
        sport_client->StandUp(req);
        RCLCPP_INFO(this->get_logger(), "Sent StandUp command!");
    }

    // 3. Comment out the movement logic
    
    void timer_callback() {
        unitree_api::msg::Request req;
        sport_client->Move(req, 0.1f, 0.0f, 0.0f);
    }
    

    std::unique_ptr<SportClient> sport_client;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotMover>());
    rclcpp::shutdown();
    return 0;
}
