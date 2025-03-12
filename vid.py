import cv2
import numpy as np
from PIL import Image

# Function to apply Canny edge detection
def canny(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    edges = cv2.Canny(blur, 50, 150)  # Detect edges using Canny
    return edges

# Function to define a trapezoidal region of interest (ROI)
def region_of_interest(canny, width, height):
    mask = np.zeros_like(canny)  # Create a black mask
    
    # Define the trapezoidal region (adjusted for better lane detection)
    trapezoid = np.array([[
        (int(width * 0), height),  
        (int(width * 1), height),  
        (int(width * 0.6), int(height * 0.3)),  
        (int(width * 0.4), int(height * 0.3))   
    ]], np.int32)

    cv2.fillPoly(mask, trapezoid, 255)  # Fill the ROI with white
    return cv2.bitwise_and(canny, mask), mask  # Apply the mask to the Canny image

# Function to detect lines using Hough Transform
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=50)  # Increased maxLineGap

# Function to overlay detected lines onto the original frame
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# Function to draw detected lane lines
def display_lines(img, lines, center_line=None):
    line_image = np.zeros_like(img)  # Create an empty black image

    if lines is not None:
        for line in lines:
            if line is None:
                continue
            for point in line:
                if point is None or len(point) != 4:
                    continue
                x1, y1, x2, y2 = point
                try:
                    cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)  # Red lane lines
                except Exception as e:
                    print(f"Error while drawing line: {e}")

    if center_line is not None:
        try:
            x1, y1, x2, y2 = center_line
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)  # Green center line
        except Exception as e:
            print(f"Error while drawing center line: {e}")

    return line_image

# Function to extrapolate lane lines
def make_points(image, line):
    if line is None:
        return None
    slope, intercept = line
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * 0.6)  # Extend to 60% of image height

    if abs(slope) < 1e-3:  # Avoid nearly horizontal lines
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]  # Extended points

# Function to average and merge similar lane lines
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return [], None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if None in (x1, y1, x2, y2):
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit
            if slope < -0.5:  # Adjusted threshold for filtering out noise
                left_fit.append((slope, intercept))  # Left lane
            elif slope > 0.5:
                right_fit.append((slope, intercept))  # Right lane

    if not left_fit or not right_fit:
        return [], None  # Avoid errors if no lines are detected

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = make_points(image, left_fit_avg)
    right_line = make_points(image, right_fit_avg)

    return [left_line, right_line], (left_line, right_line)

# Function to compute the center line between detected lanes
def compute_center_line(left_line, right_line):
    if left_line is None or right_line is None or not left_line or not right_line:
        return None  

    x1_left, y1_left, x2_left, y2_left = left_line[0]
    x1_right, y1_right, x2_right, y2_right = right_line[0]

    x1_center = (x1_left + x1_right) // 2
    y1_center = (y1_left + y1_right) // 2
    x2_center = (x2_left + x2_right) // 2
    y2_center = (y2_left + y2_right) // 2

    return [x1_center, y1_center, x2_center, y2_center]

cap = cv2.VideoCapture("movie.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape  
    canny_image = canny(frame)
    if canny_image is None:
        continue  

    cropped_canny, _ = region_of_interest(canny_image, width, height)
    lines = houghLines(cropped_canny)
    averaged_lines, lane_lines = average_slope_intercept(frame, lines)

    center_line = None
    if lane_lines is not None:
        left_line, right_line = lane_lines
        center_line = compute_center_line(left_line, right_line)
    
    line_image = display_lines(frame, averaged_lines, center_line)
    combo_image = addWeighted(frame, line_image)

    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
