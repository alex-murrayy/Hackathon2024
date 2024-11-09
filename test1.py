import cv2
import numpy as np

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
            #print(area)
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

def findSameTeam(team1_colors, team2_colors, pixel_color):
    if (len(team1_colors) == 0):
        team1_colors.append(pixel_color)
    elif (team1_colors != 0):
        if (abs(int(team1_colors[0][0]) - int(pixel_color[0])) <= 10
            and abs(int(team1_colors[0][1]) - int(pixel_color[1])) <= 10
            and abs(int(team1_colors[0][2]) - int(pixel_color[2])) <= 10):
            team1_colors.append(pixel_color)
        else:
            team2_colors.append(pixel_color)


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

    team1_colors = [] 
    team2_colors = [] 
    
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

            #print("average pixel color: ", jersey_color)

            findSameTeam(team1_colors, team2_colors, jersey_color)
        
            #print(area)
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
            if y > 200:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

    for i in range(len(team1_colors)):
        print(team1_colors[i])

    print("TEAM 2")

    for i in range(len(team2_colors)):
        print(team2_colors[i])
         

    cv2.imshow("lol", img)
    cv2.waitKey(0)
    


def lines():
    
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("lol", lines_edges)
    cv2.waitKey(0)
    
    
if __name__ == "__main__":
    main()
