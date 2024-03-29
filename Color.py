import numpy as np

colors = ["BLUE", "YELLOW"]

def color(color):
    if color == "RED":
        mask1_lower_hsv = np.array([0,50,50], dtype=np.uint8)
        mask1_upper_hsv = np.array([10,255,255], dtype=np.uint8)

        mask2_lower_hsv = np.array([170,50,50], dtype=np.uint8)
        mask2_upper_hsv = np.array([180,255,255], dtype=np.uint8)

    return mask1_lower_hsv, mask1_upper_hsv, mask2_lower_hsv, mask2_upper_hsv