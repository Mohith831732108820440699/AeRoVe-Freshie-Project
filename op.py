import rclpy as rcl
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import math as mt
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid

fov = 1.57
cam_height = 20.0
CAM_WORLD_X = 5.0
CAM_WORLD_Y = 5.0

TRANSFORM_FACTOR = cam_height / (640 / (2*mt.tan(fov / 2)))
APPROX_DRONE_RADIUS = 0.45
APPROX_DRONE_RADIUS_IN_PIXELS =0

class Image_processing(Node):
    def __init__(self):
        super().__init__('processed_image_data_node')
        self.subscription = self.create_subscription(
            Image, '/overhead_camera/image', self.image_callback, 10)
        self.circle_center_coordinates_publisher = self.create_publisher(
            PoseArray, '/circle_coordinates', 10)
        self.BINARY_grid_publisher = self.create_publisher(
            OccupancyGrid, '/binary_grid', 10)
        self.bridge = CvBridge()
        self.transformer_pixeltoWorld = TRANSFORM_FACTOR
        self.create_timer(0.05, lambda: cv.waitKey(1))
        self.get_logger().info(f'ImageProcessing ready | scale={TRANSFORM_FACTOR:.4f} m/px')

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blank = np.zeros_like(img)

        def circle_detection(contours):
            if contours:
                c = max(contours, key=cv.contourArea)
                hull = cv.convexHull(c)
                M = cv.moments(hull)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY), hull
            return None, None

        def pixel_to_world(pixel_coords):
            if pixel_coords is None:
                return None
            focal_length = img.shape[1] / (2 * mt.tan(fov / 2))
            K = cam_height / focal_length
            # Offset from camera centre + camera world position = arena world coords
            x_world = -((pixel_coords[1] - img.shape[0] / 2) * K) + CAM_WORLD_X
            y_world = -((pixel_coords[0] - img.shape[1] / 2) * K) + CAM_WORLD_Y
            return (x_world, y_world)

        # ── Disc detection ──────────────────────────────────────────
        lower_blue = np.array([90, 50, 50]);   upper_blue = np.array([130, 255, 255])
        lower_green = np.array([40, 50, 50]);  upper_green = np.array([80, 255, 255])
        lower_yellow = np.array([20, 50, 50]); upper_yellow = np.array([30, 255, 255])

        blue_mask   = cv.inRange(hsv, lower_blue,   upper_blue)
        green_mask  = cv.inRange(hsv, lower_green,  upper_green)
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

        blue_contours,_   = cv.findContours(blue_mask,   cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        green_contours,_  = cv.findContours(green_mask,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        yellow_contours,_ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        blue_pix,   blue_hull   = circle_detection(blue_contours)
        green_pix,  green_hull  = circle_detection(green_contours)
        yellow_pix, yellow_hull = circle_detection(yellow_contours)

        blue_world   = pixel_to_world(blue_pix)
        green_world  = pixel_to_world(green_pix)
        yellow_world = pixel_to_world(yellow_pix)

        # ── Obstacle detection ──────────────────────────────────────
        # Use direct dark threshold in HSV — value < 80 = dark/black objects
        # This is more reliable than Otsu which adapts to scene contrast
        lower_dark = np.array([0,   0,  0])
        upper_dark = np.array([180, 255, 80])
        obstacle_mask = cv.inRange(hsv, lower_dark, upper_dark)

        # Remove disc regions from obstacle mask so discs aren't detected as obstacles
        disc_union = cv.bitwise_or(blue_mask, green_mask)
        disc_union = cv.bitwise_or(disc_union, yellow_mask)
        obstacle_mask = cv.bitwise_and(obstacle_mask, cv.bitwise_not(disc_union))

        # Find obstacle contours and their world positions

        # ── Inflate obstacle mask for path planning ─────────────────
        kernel = np.ones((APPROX_DRONE_RADIUS_IN_PIXELS, APPROX_DRONE_RADIUS_IN_PIXELS), np.uint8)
        inflated = cv.dilate(obstacle_mask, kernel, iterations=1)

        # ── Publish circle coordinates ───────────────────────────────
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        for coord in [blue_world, green_world, yellow_world]:
            p = Pose()
            if coord is not None:
                p.position.x = float(coord[0])
                p.position.y = float(coord[1])
                p.position.z = 0.0
            else:
                p.position.z = -1.0
            pose_array.poses.append(p)
        self.circle_center_coordinates_publisher.publish(pose_array)

        # ── Publish occupancy grid ───────────────────────────────────
        binary_data = OccupancyGrid()
        binary_data.header.stamp = self.get_clock().now().to_msg()
        binary_data.header.frame_id = 'map'
        binary_data.info.resolution = float(TRANSFORM_FACTOR)
        binary_data.info.width = img.shape[1]
        binary_data.info.height = img.shape[0]
        binary_data.info.origin.position.x = -(img.shape[1] / 2) * TRANSFORM_FACTOR
        binary_data.info.origin.position.y = -(img.shape[0] / 2) * TRANSFORM_FACTOR
        grid_data = (inflated.flatten() / 255 * 100).astype(np.int8).tolist()
        binary_data.data = grid_data
        self.BINARY_grid_publisher.publish(binary_data)

        # ── Draw disc contours ───────────────────────────────────────
        for hull, pix, colour, name in [
            (blue_hull,   blue_pix,   (255, 0,   0),   'blue'),
            (green_hull,  green_pix,  (0,   255, 0),   'green'),
            (yellow_hull, yellow_pix, (0,   255, 255), 'yellow'),
        ]:
            if hull is not None:
                cv.drawContours(blank, [hull], -1, colour, 2)
            if pix is not None:
                cv.circle(blank, pix, 5, colour, -1)

        # ── Print world positions ────────────────────────────────────
        if blue_world:
            self.get_logger().info(f'Blue  disc world: ({blue_world[0]:.2f}, {blue_world[1]:.2f})')
        if green_world:
            self.get_logger().info(f'Green disc world: ({green_world[0]:.2f}, {green_world[1]:.2f})')
        if yellow_world:
            self.get_logger().info(f'Yellow disc world: ({yellow_world[0]:.2f}, {yellow_world[1]:.2f})')

        cv.imshow('Discs', blank)
        cv.imshow('Obstacles', inflated)
        cv.imshow('Original', img)

def main(args=None):
    rcl.init(args=args)
    image_node = Image_processing()
    try:
        rcl.spin(image_node)
    except KeyboardInterrupt:
        pass
    finally:
        image_node.destroy_node()
        rcl.shutdown()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
