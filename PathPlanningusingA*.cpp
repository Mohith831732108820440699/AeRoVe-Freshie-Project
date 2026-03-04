#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_map>

using std::placeholders::_1;


struct Point { float x, y; };

struct GridPoint {
    int x, y;
    bool operator==(const GridPoint& other) const {
        return x == other.x && y == other.y;
    }
};

struct GridPointHash {
    std::size_t operator()(const GridPoint& p) const {
        return std::hash<int>()(p.x) ^ std::hash<int>()(p.y);
    }
};

struct AStarNode {
    GridPoint pos;
    float g_cost;
    float h_cost;
    float f_cost;
    GridPoint parent;

    bool operator>(const AStarNode& other) const {
        return f_cost > other.f_cost;
    }
};
enum class MissionState {
    GOING_TO_GREEN,
    GOING_TO_YELLOW,
    RETURN_TO_BLUE,
    MISSION_COMPLETE
};

class GlobalPlanner : public rclcpp::Node
{
public:
    GlobalPlanner() : Node("global_planner")
    {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&GlobalPlanner::map_callback, this, _1));

        odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10, std::bind(&GlobalPlanner::odom_callback, this, _1));

        green_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/target_green", 10, std::bind(&GlobalPlanner::green_callback, this, _1));

        yellow_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/target_yellow", 10, std::bind(&GlobalPlanner::yellow_callback, this, _1));

        blue_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/target_blue", 10, std::bind(&GlobalPlanner::blue_callback, this, _1));
        
        
        // --- NEW PUBLISHER AND CONTROL TIMER ---
        trajectory_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 10);
            
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(2000), std::bind(&GlobalPlanner::timer_callback, this));
        // This timer runs 20 times a second (50ms) to keep PX4 happy
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&GlobalPlanner::control_callback, this));

        RCLCPP_INFO(this->get_logger(), "Planner Started!");
    }

private:
    nav_msgs::msg::OccupancyGrid current_map_;
    float current_x_ = 0.0;
    float current_y_ = 0.0;
    bool map_received_ = false;

    float green_x_ = 0.0, green_y_ = 0.0;
    float yellow_x_ = 0.0, yellow_y_ = 0.0;
    float blue_x_ = 0.0, blue_y_ = 0.0;
    bool targets_received_ = false;

    MissionState current_state_ = MissionState::GOING_TO_GREEN;


    std::vector<Point> current_path_;   
    size_t current_wp_index_ = 0;


    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        current_map_ = *msg;
        map_received_ = true;
    }

    void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        current_x_ = msg->position[0]; 
        current_y_ = msg->position[1]; 
    }

    void green_callback(const geometry_msgs::msg::Point::SharedPtr msg)
    {
        green_x_ = msg->x;
        green_y_ = msg->y;
        targets_received_ = true; 
    }

    void yellow_callback(const geometry_msgs::msg::Point::SharedPtr msg)
    {
        yellow_x_ = msg->x;
        yellow_y_ = msg->y;
    }
    
    void blue_callback(const geometry_msgs::msg::Point::SharedPtr msg)
    {
        blue_x_ = msg->x;
        blue_y_ = msg->y;
    }




    void timer_callback()
    {
        if (!map_received_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for map...");
            return;
        }
        if (!targets_received_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for target coordinates from camera...");
            return;
        }

        Point start = {current_x_, current_y_};
        Point goal;

        if (current_state_ == MissionState::GOING_TO_GREEN) {
            goal = {green_x_, green_y_};
            
            float dist = std::hypot(green_x_ - current_x_, green_y_ - current_y_);
            if (dist < 0.5) {
                RCLCPP_INFO(this->get_logger(), "Reached Green! Dropping payload...");
                current_state_ = MissionState::GOING_TO_YELLOW; 
                return; 
            }
        } 
        else if (current_state_ == MissionState::GOING_TO_YELLOW) {
            goal = {yellow_x_, yellow_y_};
            
            float dist = std::hypot(yellow_x_ - current_x_, yellow_y_ - current_y_);
            if (dist < 0.5) {
                RCLCPP_INFO(this->get_logger(), "Reached Yellow! Scanning Aruco...");
                current_state_ = MissionState::RETURN_TO_BLUE;
                return;
            }
        }
        else if (current_state_ == MissionState::RETURN_TO_BLUE) {
            goal = {blue_x_, blue_y_};
            float dist = std::hypot(blue_x_ - current_x_, blue_y_ - current_y_);
            if (dist < 0.5) {
                RCLCPP_INFO(this->get_logger(), "Reached Blue! MISSION__COMPLETED  Respect+++++++  Aura +++++..");
                current_state_ = MissionState::MISSION_COMPLETE;
                return;
            }
        }
        else if (current_state_ == MissionState::MISSION_COMPLETE) {
            return; 
        }

        
        std::vector<Point> new_path = run_a_star(start, goal);

        if (!new_path.empty()) {
            current_path_ = new_path;
            current_wp_index_ = 0; 
        }

    }

    void control_callback()
    {
        if (current_path_.empty() || current_wp_index_ >= current_path_.size()) {
            return; 
        }

        Point target = current_path_[current_wp_index_];

        
        float distance = std::hypot(target.x - current_x_, target.y - current_y_);
        if (distance < 0.2) {
            current_wp_index_++;
            if (current_wp_index_ >= current_path_.size()) {
                current_path_.clear(); 
                return;
            }
            target = current_path_[current_wp_index_];
        }

       
        px4_msgs::msg::TrajectorySetpoint msg;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        msg.position = {target.x, target.y, -5.0}; 
        msg.yaw = std::nanf(""); 
        
        trajectory_pub_->publish(msg);
    }

    GridPoint worldToGrid(float wx, float wy) {
        GridPoint p;
        p.x = static_cast<int>((wx - current_map_.info.origin.position.x) / current_map_.info.resolution);
        p.y = static_cast<int>((wy - current_map_.info.origin.position.y) / current_map_.info.resolution);
        return p;
    }

    Point gridToWorld(GridPoint gp) {
        Point p;
        p.x = (gp.x * current_map_.info.resolution) + current_map_.info.origin.position.x;
        p.y = (gp.y * current_map_.info.resolution) + current_map_.info.origin.position.y;
        return p;
    }

    bool isValid(GridPoint p) {
        if (p.x < 0 || p.x >= (int)current_map_.info.width || 
            p.y < 0 || p.y >= (int)current_map_.info.height) {
            return false;
        }
    
        int index = p.x + (p.y * current_map_.info.width);
        if (current_map_.data[index] > 50) { 
            return false;
        }
        return true;
    }

    std::vector<Point> run_a_star(Point start_world, Point goal_world)
    { 
        std::vector<Point> path_world;
    
        GridPoint start = worldToGrid(start_world.x, start_world.y);
        GridPoint goal = worldToGrid(goal_world.x, goal_world.y);

        if (!isValid(start) || !isValid(goal)) {
            RCLCPP_WARN(this->get_logger(), "Start or Goal is inside an obstacle!");
            return path_world;
        }

        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
        std::unordered_map<GridPoint, float, GridPointHash> g_scores; 
        std::unordered_map<GridPoint, GridPoint, GridPointHash> parents; 

        AStarNode start_node;
        start_node.pos = start;
        start_node.g_cost = 0.0;
        start_node.h_cost = std::hypot(goal.x - start.x, goal.y - start.y); 
        start_node.f_cost = start_node.g_cost + start_node.h_cost;
        start_node.parent = {-1, -1}; 

        open_set.push(start_node);
        g_scores[start] = 0.0;

        while (!open_set.empty()) {
            AStarNode current = open_set.top();
            open_set.pop();

            if (current.pos == goal) {
                GridPoint curr = goal;
                while (!(curr == start)) {
                    path_world.push_back(gridToWorld(curr));
                    curr = parents[curr];
                }
                path_world.push_back(gridToWorld(start)); 
                std::reverse(path_world.begin(), path_world.end()); 
                RCLCPP_INFO(this->get_logger(), "Path found successfully!");
                return path_world;
            }

            int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
            int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

            for (int i = 0; i < 8; i++) {
                GridPoint neighbor = {current.pos.x + dx[i], current.pos.y + dy[i]};

                if (!isValid(neighbor)) continue;

                float move_cost = (dx[i] == 0 || dy[i] == 0) ? 1.0 : 1.414;
                float tentative_g = g_scores[current.pos] + move_cost;

                if (g_scores.find(neighbor) == g_scores.end() || tentative_g < g_scores[neighbor]) {
                    g_scores[neighbor] = tentative_g;
                    parents[neighbor] = current.pos;

                    AStarNode neighbor_node;
                    neighbor_node.pos = neighbor;
                    neighbor_node.g_cost = tentative_g;
                    neighbor_node.h_cost = std::hypot(goal.x - neighbor.x, goal.y - neighbor.y);
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost;
                    
                    open_set.push(neighbor_node);
                }
            }
        }

        RCLCPP_WARN(this->get_logger(), "No path found!");
        return path_world; 
    }
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr green_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr yellow_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr blue_sub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPlanner>());
    rclcpp::shutdown();
    return 0;
}
