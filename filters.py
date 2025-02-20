import cv2
import numpy as np

# Filter 1: Grayscale
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filter 2: Super Blur
def apply_super_blur(image):
    return cv2.GaussianBlur(image, (21, 21), 0)

# Filter 3: Sharpen
def apply_sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

# Filter 4: Sepia
def apply_sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    return cv2.transform(image, sepia_filter)

# Filter 5: Invert Colors (Negative)
def apply_negative(image):
    return cv2.bitwise_not(image)

# Filter 6: Emboss
def apply_emboss(image):
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    return cv2.filter2D(image, -1, kernel)

# Filter 7: Cartoon Effect
def apply_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Filter 8: Pixelation
def apply_pixelation(image):
    height, width = image.shape[:2]
    pixelation_level = 10  # Control the size of the pixel blocks
    image = cv2.resize(image, (width // pixelation_level, height // pixelation_level), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image

# Filter 9: Thermal Vision (Color Map)
def apply_thermal(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)

# Filter 10: Oil Painting Effect
def apply_oil_painting(image):
    return cv2.xphoto.createSimpleWB().apply(image)

# Main function to handle webcam feed and apply filters
def main():
    cap = cv2.VideoCapture(0)  # Capture webcam feed
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    current_filter = 'A'  # Default filter
    filter_name = "Grayscale"  # Default filter name
    print("Press the following keys to apply filters:")
    print("A. Grayscale")
    print("B. Super Blur")
    print("C. Sharpen")
    print("D. Sepia")
    print("E. Invert Colors (Negative)")
    print("F. Emboss")
    print("G. Cartoon")
    print("H. Pixelation")
    print("I. Thermal Vision")
    
    print("Q. Quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Apply the selected filter
        if current_filter == 'A':
            filtered_frame = apply_grayscale(frame)
            filter_name = "Grayscale"
        elif current_filter == 'B':
            filtered_frame = apply_super_blur(frame)
            filter_name = "Super Blur"
        elif current_filter == 'C':
            filtered_frame = apply_sharpen(frame)
            filter_name = "Sharpen"
        elif current_filter == 'D':
            filtered_frame = apply_sepia(frame)
            filter_name = "Sepia"
        elif current_filter == 'E':
            filtered_frame = apply_negative(frame)
            filter_name = "Invert Colors (Negative)"
        elif current_filter == 'F':
            filtered_frame = apply_emboss(frame)
            filter_name = "Emboss"
        elif current_filter == 'G':
            filtered_frame = apply_cartoon(frame)
            filter_name = "Cartoon"
        elif current_filter == 'H':
            filtered_frame = apply_pixelation(frame)
            filter_name = "Pixelation"
        elif current_filter == 'I':
            filtered_frame = apply_thermal(frame)
            filter_name = "Thermal Vision"
        
        
        # Show the resulting frame with filter name
        cv2.putText(filtered_frame, f"Filter: {filter_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Filtered Frame", filtered_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Change filter based on key press
        if key == ord('q'):  # Exit
            break
        elif key == ord('a'):
            current_filter = 'A'
        elif key == ord('b'):
            current_filter = 'B'
        elif key == ord('c'):
            current_filter = 'C'
        elif key == ord('d'):
            current_filter = 'D'
        elif key == ord('e'):
            current_filter = 'E'
        elif key == ord('f'):
            current_filter = 'F'
        elif key == ord('g'):
            current_filter = 'G'
        elif key == ord('h'):
            current_filter = 'H'
        elif key == ord('i'):
            current_filter = 'I'
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
