
'''
Assignment Prompt:
    RGB -> HSV

    HSV -> RGB

    Both functions should take and return a 3D array.  You can check your answers against OpenCv's function. 
'''



# Supporting Documentation: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv


import numpy as np
import cv2


'''
    Convert the RGB to HSV code so it produces values in OpenCv's specified ranges and array
    structure. Your function should accept a 3D array and return HSV values in OpenCv's ranges. 
    H -> [0,180], S -> [0,255], V -> [0,255]
'''

def rgb_to_hsv(rgb):

    rgb = rgb/255

    r = rgb[0, 0, 0]
    g = rgb[0, 0, 1]
    b = rgb[0, 0, 2]

    h = 0.0 
    s = 0.0
    v = 0.0

    v = np.max(rgb)

    if v != 0:
        s = (v-np.min(rgb))/v
    else:
        s = 0

    if r == b and b == g:
        h = 0
    elif v == r:
        h = 60*(g-b)/(v-np.min(rgb))
    elif v == g:
        h = 120 + 60*(b-r)/(v-np.min(rgb))
    elif v == b:
        h = 240 + 60*(r-g)/(v-np.min(rgb))

    h = round(h / 2)
    s = round(s * 255)
    v = round(v * 255)


    return np.uint8([[[h, s, v]]])



def hsv_to_rgb(hsv):

    '''
        OpenCV uses a range of H [0, 180]. S [0, 255], V [0, 255].
        However, this algorithm uses the range H [0, 360], S [0, 1], V [0, 1].
        Therefore we must normalize the values before we can convert them.
    '''
    h = hsv[0, 0, 0] * 2
    s = hsv[0, 0, 1] / 255
    v = hsv[0, 0, 2] / 255

    c = v * s
    x = c * (1 - abs((h/60) % 2-1))
    m = v - c

    # Depending on the value of h, we will get different values for r, g, and b
    if h >= 0 and h < 60:
        r, g, b = c, x, 0
    elif h >= 60 and h < 120:
        r, g, b = x, c, 0
    elif h >= 120 and h < 180:
        r, g, b = 0, c, x
    elif h >= 180 and h < 240:
        r, g, b = 0, x, c
    elif h >= 240 and h < 300:
        r, g, b = x, 0, c
    elif h >= 300 and h < 360:
        r, g, b = c, 0, x
    
    # Normalize the values to the range [0, 255]
    r, g, b = (r+m)*255, (g+m)*255, (b+m)*255

    return np.uint8([[[r, g, b]]])


rgb = np.uint8([[[200,74,55]]])
hsv = rgb_to_hsv(rgb)
print("Convert RGB to HSV")
print(rgb, " -> ", hsv)

print("Convert HSV to RGB")
print(hsv, " -> ", hsv_to_rgb(hsv))









