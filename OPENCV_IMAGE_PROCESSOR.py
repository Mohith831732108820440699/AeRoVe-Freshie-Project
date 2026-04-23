import rclpy as rcl
from rclpy.node import Node    
from sensor_msgs.msg import Image  #for subscribing to image data 
from cv_bridge import CvBridge     #bridge btw ros,opencv eyes,gazebo world
import cv2 as cv
import numpy as np
import math as mt
from geometry_msgs.msg import PoseArray, Pose   #for publishing disc coordi
from nav_msgs.msg import OccupancyGrid    #for publishing 2D binary grid
from std_msgs.msg import Bool             #for ARUCO marker          

fov          = 1.57
cam_height   = 20.0
CAM_WORLD_X  = 5.0
CAM_WORLD_Y  = 5.0

TRANSFORM_FACTOR = cam_height / (640 / (2 * mt.tan(fov / 2)))

APPROX_DRONE_RADIUS_M      = 0.8
APPROX_DRONE_RADIUS_PIXELS = max(1, int(mt.ceil(APPROX_DRONE_RADIUS_M / TRANSFORM_FACTOR)))

#ARUCO marker detection setup
ARUCO_DICT   = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv.aruco.DetectorParameters()
ARUCO_DETECTOR = cv.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
TARGET_MARKER_ID = 0         

class Image_processing(Node):
    def __init__(self):
        super().__init__('processed_image_data_node')
        #subscriber
        self.subscription = self.create_subscription(
            Image, '/overhead_camera/image', self.image_callback, 10)
        #publishers
        self.circle_pub  = self.create_publisher(PoseArray,     '/circle_coordinates', 10)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/binary_grid',        10)
        self.aruco_pub   = self.create_publisher(Bool,          '/aruco_detected',     10)  

        self.bridge = CvBridge()
        self.create_timer(0.05, lambda: cv.waitKey(1))
        self.get_logger().info(
            f'ImageProcessing ready | scale={TRANSFORM_FACTOR:.4f} m/px '
            f'| inflation={APPROX_DRONE_RADIUS_PIXELS} px')

    def image_callback(self, msg):
        img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]
        hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blank = np.zeros_like(img)

        #important functions for circle detection and pixel to world conversion
        def circle_detection(contours):
            #Circle dection using CONVEX HULL 
            if contours:
                c    = max(contours, key=cv.contourArea)
                hull = cv.convexHull(c)
                M    = cv.moments(hull)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY), hull
            return None, None

        def pixel_to_world(pixel_coords):
            if pixel_coords is None:
                return None
            focal_length = w / (2 * mt.tan(fov / 2))
            K = cam_height / focal_length
            x_world = -((pixel_coords[1] - h / 2) * K) + CAM_WORLD_X
            y_world = -((pixel_coords[0] - w / 2) * K) + CAM_WORLD_Y
            return (x_world, y_world)

         #masking
        lower_blue   = np.array([90,  50,  50]); upper_blue   = np.array([130, 255, 255])
        lower_green  = np.array([40,  50,  50]); upper_green  = np.array([80,  255, 255])
        lower_yellow = np.array([20,  50,  50]); upper_yellow = np.array([30,  255, 255])

        blue_mask   = cv.inRange(hsv, lower_blue,   upper_blue)
        green_mask  = cv.inRange(hsv, lower_green,  upper_green)
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
         #contour detection
        bc, _ = cv.findContours(blue_mask,   cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        gc, _ = cv.findContours(green_mask,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        yc, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            #circle detection and pixel to world conversion
        blue_pix,   blue_hull   = circle_detection(bc)
        green_pix,  green_hull  = circle_detection(gc)
        yellow_pix, yellow_hull = circle_detection(yc)

        blue_world   = pixel_to_world(blue_pix)
        green_world  = pixel_to_world(green_pix)
        yellow_world = pixel_to_world(yellow_pix)

         #ARUCO marker detecter  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)

        aruco_found = False
        if ids is not None:
            for marker_id in ids.flatten():
                if marker_id == TARGET_MARKER_ID:
                    aruco_found = True
                    # Draw the detected marker on the debug view
                    cv.aruco.drawDetectedMarkers(blank, corners, ids)
                    self.get_logger().info(f'ArUco marker ID {TARGET_MARKER_ID} detected!')
                    break

        # Always publish the detection state so waypoint_nav gets a continuous signal
        aruco_msg = Bool()
        aruco_msg.data = aruco_found
        self.aruco_pub.publish(aruco_msg)

        
        lower_dark    = np.array([0,   0,   0])
        upper_dark    = np.array([180, 255, 80])
        grid = cv.inRange(hsv, lower_dark, upper_dark)

        disc_union    = yellow_mask+green_mask+blue_mask
        obstacle_GRID =grid-disc_union 

        kernel   = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (2 * APPROX_DRONE_RADIUS_PIXELS + 1,
             2 * APPROX_DRONE_RADIUS_PIXELS + 1))
        inflated = cv.dilate(obstacle_GRID, kernel, iterations=1)

        #Publishing disc coordinates 
        pose_array = PoseArray()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
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
        self.circle_pub.publish(pose_array)

        #Publishing 2D binary obstacle grid
        grid = OccupancyGrid()
        grid.header.stamp            = self.get_clock().now().to_msg()
        grid.header.frame_id         = 'map'
        grid.info.resolution         = float(TRANSFORM_FACTOR)
        grid.info.width              = w
        grid.info.height             = h
        grid.info.origin.position.x  = CAM_WORLD_X - (w / 2) * TRANSFORM_FACTOR
        grid.info.origin.position.y  = CAM_WORLD_Y - (h / 2) * TRANSFORM_FACTOR
        grid.info.origin.position.z  = 0.0
        grid.data = (inflated.flatten() / 255 * 100).astype(np.int8).tolist()
        self.grid_pub.publish(grid)

        #Contour visualization
        for hull, pix, colour in [
            (blue_hull,   blue_pix,   (255,   0,   0)),
            (green_hull,  green_pix,  (  0, 255,   0)),
            (yellow_hull, yellow_pix, (  0, 255, 255)),
        ]:
            if hull is not None:
                cv.drawContours(blank, [hull], -1, colour, 2)
            if pix is not None:
                cv.circle(blank, pix, 5, colour, -1)

        #world coordinates visualization
        for label, world in [('Blue', blue_world), ('Green', green_world), ('Yellow', yellow_world)]:
            if world:
                self.get_logger().info(f'{label} disc world: ({world[0]:.2f}, {world[1]:.2f})')

        cv.imshow('Discs',     blank)
        cv.imshow('Obstacles', inflated)
        cv.imshow('Original',  img)


def main(args=None):
    rcl.init(args=args)
    node = Image_processing()
    try:
        rcl.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rcl.shutdown()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
