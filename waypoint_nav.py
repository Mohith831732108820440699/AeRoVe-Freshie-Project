import math
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String

WAYPOINT_REACH_THRESHOLD = 0.15
DEFAULT_ALTITUDE         = 5.0
SETPOINT_HZ              = 20.0
ARENA_SPAWN_X            = 5.0
ARENA_SPAWN_Y            = 1.0
KP_XY                    = 0.8
KP_Z                     = 1.0
MAX_VEL_XY               = 7.0
MAX_VEL_Z                = 1.5
DISC_RADIUS              = 0.8

class MissionState(Enum):
    IDLE=auto(); TAKEOFF=auto(); NAVIGATING=auto()
    HOVERING=auto(); LANDING=auto(); DONE=auto()

class WaypointNav(Node):
    def __init__(self):
        super().__init__('waypoint_nav')
        self.create_subscription(Odometry,
            '/model/x3/odometry', self._odom_cb, 10)
        self.create_subscription(Float32MultiArray,
            '/waypoint_list', self._wp_cb, 10)
        self.create_subscription(PoseArray,
            '/circle_coordinates', self._circles_cb, 10)
        self.cmd_pub = self.create_publisher(Twist,
            '/X3/gazebo/command/twist', 10)
        self.event_pub = self.create_publisher(String,
            '/mission_event', 10)

        self.state = MissionState.IDLE
        self.waypoints = []; self.wp_idx = 0
        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.origin_set = False
        self.origin_x = 0.0; self.origin_y = 0.0
        self._last_event = ""

        # Disc positions — updated dynamically from opencv node
        self.green_x  = 2.0; self.green_y  = 6.0
        self.yellow_x = 8.0; self.yellow_y = 8.0
        self.blue_x   = 5.0; self.blue_y   = 1.0

        self.create_timer(1.0/SETPOINT_HZ, self._loop)
        self.get_logger().info('WaypointNav ready — waiting for A* waypoints')

    def _circles_cb(self, msg):
        if len(msg.poses) < 3: return
        if msg.poses[0].position.z >= 0:
            self.blue_x   = msg.poses[0].position.x
            self.blue_y   = msg.poses[0].position.y
        if msg.poses[1].position.z >= 0:
            self.green_x  = msg.poses[1].position.x
            self.green_y  = msg.poses[1].position.y
        if msg.poses[2].position.z >= 0:
            self.yellow_x = msg.poses[2].position.x
            self.yellow_y = msg.poses[2].position.y

    def load_waypoints(self, wps):
        if not wps: return
        self.waypoints = list(wps); self.wp_idx = 0
        if self.state == MissionState.IDLE:
            self.state = MissionState.TAKEOFF
            self.get_logger().info(f'Mission loaded: {len(wps)} waypoints')
        else:
            self.get_logger().info(f'Waypoints updated: {len(wps)} waypoints')
            self.state = MissionState.NAVIGATING

    def land(self):
        self.state = MissionState.LANDING

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        if not self.origin_set:
            self.origin_x = self.x; self.origin_y = self.y
            self.origin_set = True
            self.get_logger().info(f'Origin: ({self.x:.2f},{self.y:.2f},{self.z:.2f})')

    def _wp_cb(self, msg):
        d = msg.data
        wps = [(float(d[i]), float(d[i+1]), float(d[i+2])) for i in range(0, len(d)-2, 3)]
        if wps:
            self.get_logger().info(f'Received {len(wps)} waypoints from A*')
            self.load_waypoints(wps)

    def _arena_to_gz(self, ax, ay, az):
        return ((ax - ARENA_SPAWN_X) + self.origin_x,
                (ay - ARENA_SPAWN_Y) + self.origin_y, az)

    def _dist(self, wp):
        tx, ty, tz = self._arena_to_gz(*wp)
        return math.sqrt((self.x-tx)**2 + (self.y-ty)**2 + (self.z-tz)**2)

    def _vel_toward(self, ax, ay, az):
        tx, ty, tz = self._arena_to_gz(ax, ay, az)
        ex = tx - self.x; ey = ty - self.y; ez = tz - self.z
        vx = max(-MAX_VEL_XY, min(KP_XY*ex, MAX_VEL_XY))
        vy = max(-MAX_VEL_XY, min(KP_XY*ey, MAX_VEL_XY))
        vz = max(-MAX_VEL_Z,  min(KP_Z *ez, MAX_VEL_Z))
        msg = Twist()
        msg.linear = Vector3(x=vx, y=vy, z=vz)
        self.cmd_pub.publish(msg)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _publish_event(self, event):
        if self._last_event != event:
            self._last_event = event
            msg = String(); msg.data = event
            self.event_pub.publish(msg)
            self.get_logger().info(f'Mission event: {event}')

    def _check_disc_arrival(self):
        ax = self.x - self.origin_x + ARENA_SPAWN_X
        ay = self.y - self.origin_y + ARENA_SPAWN_Y
        if math.hypot(ax - self.green_x, ay - self.green_y) < DISC_RADIUS:
            self._publish_event('arrived_green')
        elif math.hypot(ax - self.yellow_x, ay - self.yellow_y) < DISC_RADIUS:
            self._publish_event('arrived_yellow')
        elif (math.hypot(ax - self.blue_x, ay - self.blue_y) < DISC_RADIUS
              and self.state != MissionState.TAKEOFF
              and self.wp_idx > 0):
            self._publish_event('arrived_blue')

    def _loop(self):
        s = self.state
        if s == MissionState.IDLE:
            return

        elif s == MissionState.TAKEOFF:
            target_z = self.waypoints[0][2] if self.waypoints else DEFAULT_ALTITUDE
            self._vel_toward(ARENA_SPAWN_X, ARENA_SPAWN_Y, target_z)
            if abs(self.z - target_z) < WAYPOINT_REACH_THRESHOLD:
                self.get_logger().info(f'Takeoff done z={self.z:.2f}m → NAVIGATING')
                self.state = MissionState.NAVIGATING

        elif s == MissionState.NAVIGATING:
            if self.wp_idx >= len(self.waypoints):
                self.get_logger().info('All waypoints done → HOVERING')
                self._stop(); self.state = MissionState.HOVERING; return
            wp = self.waypoints[self.wp_idx]
            self._vel_toward(*wp)
            self._check_disc_arrival()
            if self._dist(wp) < WAYPOINT_REACH_THRESHOLD:
                self.get_logger().info(f'✓ WP {self.wp_idx} reached: {wp}')
                self.wp_idx += 1

        elif s == MissionState.HOVERING:
            if self.waypoints:
                self._vel_toward(*self.waypoints[-1])
            self._check_disc_arrival()

        elif s == MissionState.LANDING:
            self._vel_toward(
                self.waypoints[-1][0] if self.waypoints else ARENA_SPAWN_X,
                self.waypoints[-1][1] if self.waypoints else ARENA_SPAWN_Y,
                0.3)
            if self.z < 0.5:
                self._stop(); self.state = MissionState.DONE
                self.get_logger().info('Landed → DONE')
                self._publish_event('arrived_blue')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
