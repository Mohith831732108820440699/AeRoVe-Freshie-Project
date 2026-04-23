#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
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
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 16);
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

class GlobalPlanner : public rclcpp::Node {
public:
    GlobalPlanner() : Node("global_planner") {
        // --- Tunable safety parameter ---
        // How far (in metres) the drone must stay away from any obstacle.
        // Increase this value if the drone still clips obstacles;
        // decrease it if the planner fails to find paths in tight spaces.
        this->declare_parameter<float>("inflation_radius_m", 0.6f);
        inflation_radius_m_ = this->get_parameter("inflation_radius_m").as_double();

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/binary_grid", 10, std::bind(&GlobalPlanner::map_callback, this, _1));

        odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10, std::bind(&GlobalPlanner::odom_callback, this, _1));

        targets_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/circle_coordinates", 10, std::bind(&GlobalPlanner::targets_callback, this, _1));

        trajectory_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 10);

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(2000), std::bind(&GlobalPlanner::timer_callback, this));
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&GlobalPlanner::control_callback, this));

        RCLCPP_INFO(this->get_logger(), "Planner Started! Inflation radius: %.2f m", inflation_radius_m_);
    }

private:
    nav_msgs::msg::OccupancyGrid current_map_;
    std::vector<int8_t> inflated_grid_;   // Inflated copy used by A*
    float current_x_ = 0.0;
    float current_y_ = 0.0;
    bool map_received_ = false;

    float goal_x_ = 0.0, goal_y_ = 0.0;
    bool targets_received_ = false;

    float inflation_radius_m_ = 0.6f;    // Safety clearance in metres

    std::vector<Point> current_path_;
    size_t current_wp_index_ = 0;

    
    // Inflate every obstacle cell by inflation_radius_m_ so that A* will
    // treat nearby cells as blocked and keep the drone at a safe distance
    void inflate_map() {
        const int w = static_cast<int>(current_map_.info.width);
        const int h = static_cast<int>(current_map_.info.height);
        const float res = current_map_.info.resolution;

        // Radius in grid cells (rounded up so we never under-inflate)
        const int radius_cells = static_cast<int>(std::ceil(inflation_radius_m_ / res));

        // Start with a clean copy of the raw grid
        inflated_grid_ = current_map_.data;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Only expand actual obstacle cells (value 255 in a binary grid)
                if (current_map_.data[x + y * w] != 0) {
                    // Mark every cell within radius_cells as occupied
                    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                        for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
                            // Use Euclidean distance so we get a circular footprint
                            if (std::hypot(dx, dy) > radius_cells) continue;

                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                inflated_grid_[nx + ny * w] = 100; // Mark as occupied
                            }
                        }
                    }
                }
            }
        }

        RCLCPP_DEBUG(this->get_logger(), "Inflation complete (radius = %d cells)", radius_cells);
    }

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        current_map_ = *msg;
        map_received_ = true;
        inflate_map();  // Rebuild the inflated grid every time the map updates
    }

    void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
        current_x_ = msg->position[0];
        current_y_ = msg->position[1];
    }

    void targets_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        if (msg->poses.size() >= 1) {
            goal_x_ = msg->poses[0].position.x;
            goal_y_ = msg->poses[0].position.y;
            targets_received_ = true;
        }
    }

    void timer_callback() {
        if (!map_received_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for map...");
            return;
        }
        if (!targets_received_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for target coordinates...");
            return;
        }

        Point start = {current_x_, current_y_};
        Point goal  = {goal_x_,    goal_y_};

        std::vector<Point> new_path = run_a_star(start, goal);

        if (!new_path.empty()) {
            current_path_    = new_path;
            current_wp_index_ = 0;

            // Path visualisation
            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp    = this->get_clock()->now();
            path_msg.header.frame_id = "map";

            for (const auto& point : current_path_) {
                geometry_msgs::msg::PoseStamped pose;
                pose.header              = path_msg.header;
                pose.pose.position.x     = point.x;
                pose.pose.position.y     = point.y;
                pose.pose.position.z     = 0.5;
                path_msg.poses.push_back(pose);
            }

            path_pub_->publish(path_msg);
        }
    }

    void control_callback() {
        if (current_path_.empty() || current_wp_index_ >= current_path_.size()) {
            return;
        }

        Point target  = current_path_[current_wp_index_];
        float distance = std::hypot(target.x - current_x_, target.y - current_y_);

        if (distance < 0.2f) {
            current_wp_index_++;
            if (current_wp_index_ >= current_path_.size()) {
                current_path_.clear();
                return;
            }
            target = current_path_[current_wp_index_];
        }

        px4_msgs::msg::TrajectorySetpoint msg;
        msg.timestamp  = this->get_clock()->now().nanoseconds() / 1000;
        msg.position   = {target.x, target.y, -5.0};
        msg.yaw        = std::nanf("");

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

    // isValid now checks the INFLATED grid, not the raw map
    bool isValid(GridPoint p) {
        if (p.x < 0 || p.x >= (int)current_map_.info.width ||
            p.y < 0 || p.y >= (int)current_map_.info.height) {
            return false;
        }
        int index = p.x + (p.y * current_map_.info.width);
        return inflated_grid_[index] == 0; // Free only if not inflated
    }

    std::vector<Point> run_a_star(Point start_world, Point goal_world) {
        std::vector<Point> path_world;
        GridPoint start = worldToGrid(start_world.x, start_world.y);
        GridPoint goal  = worldToGrid(goal_world.x,  goal_world.y);

        // If the goal falls inside the inflation zone, warn but still try —
        // the drone needs to reach the target even if it is close to a wall.
        if (!isValid(start)) {
            RCLCPP_WARN(this->get_logger(), "Start is inside an obstacle or inflation zone!");
            return path_world;
        }
        if (!isValid(goal)) {
            RCLCPP_WARN(this->get_logger(),
                "Goal is inside an obstacle or inflation zone! "
                "Consider moving the goal or reducing inflation_radius_m.");
            return path_world;
        }

        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
        std::unordered_map<GridPoint, float,     GridPointHash> g_scores;
        std::unordered_map<GridPoint, GridPoint, GridPointHash> parents;

        AStarNode start_node;
        start_node.pos    = start;
        start_node.g_cost = 0.0f;
        start_node.h_cost = std::hypot(goal.x - start.x, goal.y - start.y);
        start_node.f_cost = start_node.g_cost + start_node.h_cost;
        start_node.parent = {-1, -1};

        open_set.push(start_node);
        g_scores[start] = 0.0f;

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
                RCLCPP_INFO(this->get_logger(), "Path found successfully! (%zu waypoints)", path_world.size());
                return path_world;
            }

            // Skip stale entries (a node may be pushed multiple times)
            if (g_scores.count(current.pos) && current.g_cost > g_scores[current.pos]) {
                continue;
            }

            const int dx[] = {1, 1, 0, -1, -1, -1,  0,  1};
            const int dy[] = {0, 1, 1,  1,  0, -1, -1, -1};

            for (int i = 0; i < 8; i++) {
                GridPoint neighbor = {current.pos.x + dx[i], current.pos.y + dy[i]};

                if (!isValid(neighbor)) continue;

                float move_cost      = (dx[i] == 0 || dy[i] == 0) ? 1.0f : 1.414f;
                float tentative_g    = g_scores[current.pos] + move_cost;

                if (g_scores.find(neighbor) == g_scores.end() || tentative_g < g_scores[neighbor]) {
                    g_scores[neighbor]  = tentative_g;
                    parents[neighbor]   = current.pos;

                    AStarNode neighbor_node;
                    neighbor_node.pos    = neighbor;
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
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr targets_sub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPlanner>());
    rclcpp::shutdown();
    return 0;
}
