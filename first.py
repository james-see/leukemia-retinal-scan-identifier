import cv2
import numpy as np


import cv2
import numpy as np


def crop_to_square(img):
    y, x, _ = img.shape
    size = min(x, y)
    start_x = (x - size) // 2
    start_y = (y - size) // 2
    return img[start_y : start_y + size, start_x : start_x + size]


def crop_center(img, percent=80):
    y, x, _ = img.shape
    start_x = x * (1 - percent / 100) // 2
    start_y = y * (1 - percent / 100) // 2
    end_x = start_x + x * percent / 100
    end_y = start_y + y * percent / 100
    return img[int(start_y) : int(end_y), int(start_x) : int(end_x)]


def is_unhealthy_retina(image_path, threshold=15000):
    img = cv2.imread(image_path)

    # Crop to square
    img = crop_to_square(img)

    # Scale to 500x500
    img = cv2.resize(img, (500, 500))

    # Crop the image to 80%
    img = crop_center(img, 70)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a threshold to detect dark regions (splotches). 50 is a chosen value to capture dark areas, but this can be adjusted.
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Mask', dark_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    total_detected = np.sum(dark_mask / 255)
    print(total_detected)
    return total_detected > threshold


# Test the function on your images
image_paths = ["pos.jpg", "neg.jpg", "pos2.jpg"]

for path in image_paths:
    result = is_unhealthy_retina(path)
    if result:
        print(f"{path} is unhealthy.")
    else:
        print(f"{path} is healthy.")
