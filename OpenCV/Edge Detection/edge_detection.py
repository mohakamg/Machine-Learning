from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0,255,0)

################################################## Helper Functions #############################################
def show(image):
    # Figure size in Inches
    plt.figure(figsize=(10,10))
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask_clean, image):
    # Make the mask RGB
    rgb_mask = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB) # Covert to greyscale because the original image is in RGB
    # Calvulate the weighted sum of 2 images - Add 2 image arrays numerical pixel value to get the overlay over the image
    # which is basically just a new color
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Make a copy of the image
    image = image.copy()
    # Get the Contour
    im, contours, heirarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_SIMPLE - Gives
                                                                                         # the contours of only the end points.
                                                                                         # For eg: the contours of a rectangle would be the 4 corners.
                                                                                         # RETR_LIST (Retrive_list) - Gives the Entire list of the contours found
    # Isolating the Largest Contours
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Draw the Contour
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, contours,-1, (255, 255, 255),3)
    # Return the biggest Contour
    return biggest_contour, mask

def circle_contour(image, contour):
     # Get the Bounding Ellipse
     image_with_ellipse = image.copy()
     ellipse = cv2.fitEllipse(contour)
     # Add it
     cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)

     return image_with_ellipse

################################################## Helper Functions END #############################################

################################################## Main Fucntion #############################################
def find_strawberry(image):
    print("Orignial Image Matrix: ", image)
    cv2.imwrite('RGB_IMAGE.jpg', image)
    ######################################### Step 1 - Convert to correct color scheme #########################################
        # RGB is red, green, blue
        # BGR is blue, green, red
        # The difference is in RGB order, Red occupies the most significant area of memory and blue the least
        # and in BGR, Red occupies the least significant area of memory and blue the most.
        # So the order of color for image processing matters depending on the
        # color detection - in this case Red for strawberries.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("RGB Image Matrix: ", image)
    cv2.imwrite('Orignal_IMAGE.jpg', image)

    ######################################### Step 2 - Rescale the image #########################################
        # Get the size of the image
    max_dimensions = max(image.shape) # Shape is the window size
    print("Dimensions of Image: ", max_dimensions)
        # Scale the image to the correct dimensions
    scale = 700/max_dimensions # Maximum Windows size that we are gonna use is 700 x 660 px
    print("Scaling by a factor of: ", scale)
    image = cv2.resize(image, None, fx=scale, fy=scale) # Resize the image to a square
    print("New Image Dimensions: ", max(image.shape))

    ######################################### Step 3 - Clean our image #########################################
    image_blur = cv2.GaussianBlur(image, (7,7), 0) # To remove noise from the image we smooth out the image - in our case the yellow seeds/dots
                                    # on the strawberries are noise for us. We want to remove that. We just want clean red Image.
                                    # This also removes other black dots and stuffs that are noise.
                                    # Function - GaussianBlur(Image Object, Kernal Size (700 x 700 pixels), Filter Stength - 0 as, GaussianBlur automatically blurs enough)
    print("Gaussian Blurred Image: ", image_blur)
    cv2.imwrite('Gaussian Blurred Image.jpg', image_blur)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV) # Convert to hsv - Hue Saturation Value, because we want to filter by color.
                                                                 # HSV Format seperates the image intensity (Brightness) Luma from the Color Information Chroma
    print("Gaussian Blurred HSV Image: ", image_blur_hsv)
    cv2.imwrite('Gaussian Blurred HSV Image.jpg', image_blur_hsv)

    ######################################### Step 4 - Define filters #########################################
        # 1 - Filter By the color - not the brightness. Basically a specific range of red's
    min_color = np.array([0,100,80]) # Minimum amount of redness
    max_color = np.array([10,256,256 ]) # Maximum amount of redness
    # min_color = np.array([0,0,0]) # Minimum amount of color shade
    # max_color = np.array([100,75,100]) # Maximum amount of color shade
        # Mask the image pixles in the color range
    mask1 = cv2.inRange(image_blur_hsv, min_color, max_color)
    print("Color Mask: ", mask1)
    cv2.imwrite('Color Mask.jpg', mask1)
        # 2 - Filter By the Brightness
    min_brightness = np.array([170,100,80]) # Minimum amount of Brightness
    max_brightness = np.array([180,256,256]) # Maximum amount of Brightness
        # Mask the image pixles in the brightness range
    mask2 = cv2.inRange(image_blur_hsv, min_brightness, max_brightness)
    print("Brightness Mask: ", mask2)
    cv2.imwrite('Brightness Mask.jpg', mask2)
        # Take these two masks and combine the masks
    mask = mask1 + mask2
    print("Brightness + Color Mask: ", mask)
    cv2.imwrite('Brightness + color Mask.jpg', mask)

    ######################################### Step 5 - Segmentation #########################################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)) # Define the Structure to an ELlipse - because strawnerries look like that
    print("Kernel: ", kernel)
        # Remove Noise
        # NOTICE - These masks will be greyscale
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Closing Operation - Dilation (Close Small holes in the foreground. For eg: Smooth out the black pixles present on the stawberry) followed by Errosion
    print("Closed Mask: ", mask_closed)
    cv2.imwrite('Closed Mask.jpg', mask_closed)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel) # Open Operation - Errosion followed by Dilation
    print("Cleaned Mask: ", mask_clean)
    cv2.imwrite('Cleaned Mask.jpg', mask_clean)

    ######################################### Step 6 -  Find the Biggest Strawberry #########################################
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    print("Mask: ", mask_strawberries)
    cv2.imwrite('Mask.jpg', mask_strawberries)

    ######################################### Step 7 - Overlay the mask that we created on the image #########################################
    overlay = overlay_mask(mask_clean, image)
    print("Overlay: ", overlay)
    cv2.imwrite('Overlay.jpg', overlay)

    ######################################### Step 8 - Circle the bigegst Strawberry #########################################
    circled = circle_contour(overlay, big_strawberry_contour)
    show(circled)

    ######################################### Step 9 - Convert Back to original scheme #########################################
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    return bgr

############################################## Read the image ###########################################
image_name = input("Enter Image Name: ")
image = cv2.imread(image_name)
result = find_strawberry(image)
############################################# Write the new Image ###########################################
cv2.imwrite('detected_strawberry.jpg', result)
