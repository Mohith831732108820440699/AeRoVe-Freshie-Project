import rclpy as rcl
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import math as mt
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
fov=1.57
cam_height=20
TRANSFORM_FACTOR=cam_height/(640/(2*mt.tan(fov/2)))
APPROX_DRONE_RADIUS=0.2*mt.sqrt(2)
APPROX_DRONE_RADIUS_IN_PIXELS=int(APPROX_DRONE_RADIUS/TRANSFORM_FACTOR)
class Image_processing(Node):
    def __init__(self):
        super().__init__('processed_image_data_node')
        self.subscription = self.create_subscription(Image,'/overhead_camera',self.image_callback,10)
        self.subscription  
        self.circle_center_coordinates_publisher = self.create_publisher(PoseArray, '/circle_coordinates', 10)
        self.BINARY_grid_publisher = self.create_publisher(OccupancyGrid, '/binary_grid', 10)
        self.bridge = CvBridge()
        self.transformer_pixeltoWorld=TRANSFORM_FACTOR

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        def circle_detection(contours):
            if contours:
                c=max(contours, key=cv.contourArea)
                hull=cv.convexHull(c)
                M=cv.moments(hull)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY), hull
            return None
        def World_coordinates(pixel_coordinates,img,cam_height=20,fov=1.57):
            focal_length=img.shape[1]/(2*mt.tan(fov/2))
            K=cam_height/focal_length
            x_world=(pixel_coordinates[0]-img.shape[1]/2)*K
            y_world=(pixel_coordinates[1]-img.shape[0]/2)*K
            return (x_world,y_world)
            
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gauss_blur=cv.GaussianBlur(gray,(7,7),0)
        adaptive_thresh=cv.adaptiveThreshold(gauss_blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,129,2)
        clean_thresh=cv.medianBlur(adaptive_thresh,5)
        kernal=np.ones((5,5),np.uint8)
        solid_square=cv.morphologyEx(clean_thresh,cv.MORPH_CLOSE,kernal)
        blank = np.zeros_like(img)
        hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        #blue cicle masking
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
        blue_contours,_=cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blue_circle_pix_coordinates, blue_hull = circle_detection(blue_contours)
        blue_actual_coordinates=World_coordinates(blue_circle_pix_coordinates,img)
        
        #green circle masking
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv.inRange(hsv, lower_green, upper_green)
        green_contours,_=cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        green_circle_pix_coordinates, green_hull = circle_detection(green_contours)
        green_actual_coordinates=World_coordinates(green_circle_pix_coordinates,img)
        
        #YELLOW circle masking
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
        yellow_contours,_=cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        yellow_circle_pix_coordinates,yellow_hull = circle_detection(yellow_contours)
        yellow_actual_coordinates=World_coordinates(yellow_circle_pix_coordinates,img)
        
        #creating 2D binary grid
        otsu_THRESH,otsu_tresh=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        main_2DGrid=cv.bitwise_or(blue_mask,green_mask)
        main_2DGrid=cv.bitwise_or(main_2DGrid,yellow_mask)
        main_2DGrid=cv.bitwise_or(main_2DGrid,otsu_tresh)
        kernal=np.ones((APPROX_DRONE_RADIUS_IN_PIXELS,APPROX_DRONE_RADIUS_IN_PIXELS),np.uint8)
        inflated_main_2DGrid=cv.dilate(main_2DGrid,kernal,iterations=1)
       
        #print(img.shape)
        #coordinates publishing    
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'MAP'
        circle_center_coordies=[blue_actual_coordinates, green_actual_coordinates, yellow_actual_coordinates]
        for coord in circle_center_coordies:
            pos = Pose()
            pos.position.x = float(coord[0])
            pos.position.y = float(coord[1])
            pos.position.z = 0.0
            pose_array.poses.append(pos)  
        self.circle_center_coordinates_publisher.publish(pose_array)
        #binary grid publishing
        binary_data= OccupancyGrid()
        binary_data.header.stamp = self.get_clock().now().to_msg()
        binary_data.header.frame_id = 'MAP'        
        binary_data.info.resolution = float(TRANSFORM_FACTOR) 
        binary_data.info.width = img.shape[1]
        binary_data.info.height = img.shape[0]
        binary_data.info.origin.position.x = -(img.shape[1] / 2) * TRANSFORM_FACTOR
        binary_data.info.origin.position.y = -(img.shape[0] / 2) * TRANSFORM_FACTOR
        grid_data = (inflated_main_2DGrid.flatten() / 255 * 100).astype(np.int8).tolist()
        binary_data.data = grid_data
        
        self.BINARY_grid_publisher.publish(binary_data)

        print(f'blue circle actual coordinates: {blue_actual_coordinates}')
        print(f'blue circle pixel coordinates: {blue_circle_pix_coordinates}')
        cv.drawContours(blank, [blue_hull], -1, (0, 255, 0), 2)
        cv.circle(blank, blue_circle_pix_coordinates, 5, (255, 0, 0), -1)

        print(f'green circle actual coordinates: {green_actual_coordinates}')
        print(f'green circle pixel coordinates: {green_circle_pix_coordinates}')
        cv.drawContours(blank, [green_hull], -1, (0, 255, 0), 2)
        cv.circle(blank, green_circle_pix_coordinates, 5, (255, 0, 0), -1)

        print(f'Yellow circle actual coordinates: {yellow_actual_coordinates}')
        print(f'Yellow circle pixel coordinates: {yellow_circle_pix_coordinates}')
        cv.drawContours(blank, [yellow_hull], -1, (0, 255, 0), 2)
        cv.circle(blank, yellow_circle_pix_coordinates, 5, (255, 0, 0), -1)
        cv.imshow('yellow_contours',blank)

        cv.imshow('main_2DGird',inflated_main_2DGrid)      
        cv.imshow('original',img)
    
    cv.waitKey(1)

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
