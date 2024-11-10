import cv2
import numpy as np

from sklearn.cluster import KMeans

def seperateAveragesIntoCategories(average_colors):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(average_colors)

    # Get cluster labels (0 or 1)
    labels = kmeans.labels_

    # Separate the points into two categories
    print("Category 1:", labels)
    

# finds the color of the two kits
def findColor(): 
    img = cv2.imread("football.jpg")

    # Color contrator (can ignore)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    kernel_size = 7
    blurred = cv2.GaussianBlur(img_hsv,(kernel_size, kernel_size),0)
    
    # Define range of green color in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(blurred, lower_green, upper_green)

    
    # using a findContours() function     
    contours, _ = cv2.findContours( 
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours: 

        area=cv2.contourArea(contour)
        if area > 200 and area < 10000:
            print(area)
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box


            #cv2.circle(img, (int(x+w/2), int(y+h/2)), 10, (255, 0, 0), 2)

            pixel_color = img[int(y+h/2), int(x+w/2)] 

            if pixel_color[0] > 200 and pixel_color[1] > 200 and pixel_color[2] > 200:  
                cv2.circle(img, (int(x+w/2), int(y+h/2)), 10, (255, 0, 0), 2)
            else:
                cv2.circle(img, (int(x+w/2), int(y+h/2)), 10, (0, 0, 255), 2)

            print(pixel_color)


            if y > 200:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    cv2.imshow("lol", img)
    cv2.waitKey(0)


def main():
    img = cv2.imread("football.jpg")

    # Color contrator (can ignore)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    kernel_size = 7
    blurred = cv2.GaussianBlur(img_hsv,(kernel_size, kernel_size),0)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # Define range of green color in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(blurred, lower_green, upper_green)

    
    # using a findContours() function 
    contours, _ = cv2.findContours( 
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    average_colors = []
    
    for contour in contours: 
        
        area=cv2.contourArea(contour)
        if area > 200 and area < 10000:
        
            x, y, w, h = cv2.boundingRect(contour)
        
            mask = np.zeros_like(gray, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            #cv2.imshow("lol", mask)
            #cv2.waitKey(0)
            
            jersey_color = cv2.mean(img, mask=mask)[:3]
            cv2.drawContours(img, [contour], -1, jersey_color, -1)

            print("average pixel color: ", jersey_color)
            average_colors.append(jersey_color)
        
            print(area)
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
            if y > 200:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    seperateAveragesIntoCategories(average_colors)
    cv2.imshow("lol", img)
    cv2.waitKey(0)
    

def warp(image):
    pts1 = np.float32([[95, 17], [805, 17],
                       [-150, 349], [1100, 349]])
    pts2 = np.float32([[0, 0], [700, 0],
                       [0, 500], [700, 500]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (700, 500))
    return result
    
    
if __name__ == "__main__":
    cap = cv2.VideoCapture('RPReplay_Final1731194189.mov')
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
     
        # Display the resulting frame
        cv2.imshow('Frame',warp(frame))
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()

    #    main()
