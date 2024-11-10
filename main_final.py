import cv2
import numpy as np

def get_line_on_original_image(point1, point2, transform):
    M_inv = np.linalg.inv(transform)
    transformed_p1 = np.dot(M_inv, [point1[0], point1[1], 1])
    transformed_p2 = np.dot(M_inv, [point2[0], point2[1], 1])
    
    transformed_p1 = transformed_p1 / transformed_p1[2]
    transformed_p2 = transformed_p2 / transformed_p2[2]
    
    transformed_p1 = [int(i) for i in transformed_p1][:2]
    transformed_p2 = [int(i) for i in transformed_p2][:2]
    return transformed_p1, transformed_p2

def warp(image):
    pts1 = np.float32([[95, 17], [805, 17],
                       [-150, 349], [1100, 349]])
    pts2 = np.float32([[0, 0], [700, 0],
                       [0, 500], [700, 500]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (700, 500))
    return result, matrix
    
    
def main():
    filename = input("Enter filename to get offsides from: ")
    try:
        cap = cv2.VideoCapture(filename)
    except Exception:
        cap = cv2.VideoCapture("RPReplay_Final1731194189.mov") #default file
        
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Color contrator (can ignore)
        warped, transform = warp(frame)
        img_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        kernel_size = 7
        blurred = cv2.GaussianBlur(img_hsv,(kernel_size, kernel_size),0)
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        
        # Define range of green color in HSV
        lower_green = np.array([45, 50, 50])
        upper_green = np.array([75, 255, 255])
        mask = cv2.inRange(blurred, lower_green, upper_green)

        
        # using a findContours() function 
        contours, _ = cv2.findContours( 
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        average_colors = []
        
        rightmostPos = 0
        
        for contour in contours: 
            
            area=cv2.contourArea(contour)
            if area > 110 and area < 800:
            
                x, y, w, h = cv2.boundingRect(contour)
            
                mask = np.zeros_like(gray, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
                cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
                if x+w/2 > rightmostPos:
                    rightmostPos = int(x+w/2)
                    
        transformed_p1, transformed_p2 = get_line_on_original_image((rightmostPos, 0), (rightmostPos, len(warped[0])), transform)
        
        cv2.line(warped, (rightmostPos, 0), (rightmostPos, len(warped[0])), (255, 0, 0), 2)  # Draw rectangle
        cv2.line(frame, transformed_p1, transformed_p2, (255, 0, 0), 2)  # Draw rectangle
 
  
        cv2.imshow('Frame',frame)
        cv2.imshow('Defenders',warped)
        
        print(rightmostPos)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    



    
if __name__ == "__main__":
    main()
