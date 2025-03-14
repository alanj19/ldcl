import cv2
import numpy as np
from PIL import Image  

# Function to apply Canny edge detection
def canny(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  
    edges = cv2.Canny(blur, 50, 150)  
    return edges

# Function to define and visualize a trapezoidal region of interest (ROI)
def region_of_interest(canny, frame, width, height):
    mask = np.zeros_like(canny)  
    trapezoid = np.array([[
        (int(width * 0), height), 
        (int(width * 1), height), 
        (int(width * 0.6), int(height * 0.5)), 
        (int(width * 0.4), int(height * 0.5))  
    ]], np.int32)

    cv2.fillPoly(mask, trapezoid, 255)
    masked_canny = cv2.bitwise_and(canny, mask)

    return masked_canny, trapezoid

# Function to detect lines using Hough Transform
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=50)

# Function to overlay detected lines onto the original frame
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# Function to draw detected lane lines
def display_lines(img, lines, center_line=None):
    line_image = np.zeros_like(img)  

    if lines is not None:
        for line in lines:
            if line is None:
                continue
            for point in line:
                if point is None or len(point) != 4:
                    continue
                x1, y1, x2, y2 = point

                # Make sure the line coordinates are within image bounds
                if any([x1 < 0, x1 > img.shape[1], y1 < 0, y1 > img.shape[0], x2 < 0, x2 > img.shape[1], y2 < 0, y2 > img.shape[0]]):
                    continue

                try:
                    cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)  
                except Exception as e:
                    print(f"Error while drawing line: {e}")

    if center_line is not None:
        try:
            x1, y1, x2, y2 = center_line
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)  
        except Exception as e:
            print(f"Error while drawing center line: {e}")

    return line_image

# Function to derive lane lines
def make_points(image, line):
    if line is None:
        return None
    slope, intercept = line
    y1 = image.shape[0]  
    y2 = int(y1 * 0.6)  

    if abs(slope) < 1e-3:  
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]  

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

            # Handle poorly conditioned polyfit
            if np.isnan(fit[0]) or np.isinf(fit[0]):
                continue

            slope, intercept = fit
            if slope < -0.5:  
                left_fit.append((slope, intercept))  
            elif slope > 0.5:
                right_fit.append((slope, intercept))  

    if not left_fit or not right_fit:
        return [], None  

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

# Open the video file
cap = cv2.VideoCapture("test2.mp4")

# Load the transparent image (ensure it's RGBA for transparency)
overlay_image = Image.open("img1.png").convert("RGBA")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize the overlay image
overlay_width = 100  
overlay_height = 100  
overlay_resized = overlay_image.resize((overlay_width, overlay_height))

# Rotate the overlay image
overlay_rotated = overlay_resized.rotate(90, expand=True)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f"Displaying frame: {frame_number}")

    height, width, _ = frame.shape 
    canny_image = canny(frame)
    if canny_image is None:
        continue 

    # Apply region of interest (with visualization)
    cropped_canny, trapezoid = region_of_interest(canny_image, frame, width, height)
    lines = houghLines(cropped_canny)
    averaged_lines, lane_lines = average_slope_intercept(frame, lines)

    # Compute center line
    center_line = None
    if lane_lines is not None:
        left_line, right_line = lane_lines
        center_line = compute_center_line(left_line, right_line)

    # Overlay detected lane lines
    line_image = display_lines(frame, averaged_lines, center_line)
    combo_image = addWeighted(frame, line_image)

    # Convert OpenCV frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Rotate the overlay image conditionally
    if 2650 <= frame_number <= 2850:
        rotated_overlay = overlay_rotated.rotate(270, expand=True)  # Rotate again
    else:
        rotated_overlay = overlay_rotated  # Keep original rotation

    # Position overlay in the top-right corner
    position = (0, 0)  
    frame_pil.paste(rotated_overlay, position, rotated_overlay)  

    # Convert back to OpenCV format
    frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

    # Draw ROI trapezoid
    cv2.polylines(frame_bgr, [trapezoid], isClosed=True, color=(0, 255, 0), thickness=2)  

    # Show final output
    cv2.imshow("Lane Detection with ROI & Overlay", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

