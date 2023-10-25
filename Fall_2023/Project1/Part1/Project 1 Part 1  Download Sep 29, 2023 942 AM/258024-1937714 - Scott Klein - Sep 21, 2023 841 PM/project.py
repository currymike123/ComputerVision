import numpy as np 
import matplotlib.pyplot as plt 

#rgb to hsv 
def rgb_to_hsv(rgb):

    #input 3d array, get color values
    r = rgb[0][0][0]
    b = rgb[0][0][1]
    g = rgb[0][0][2]

    #normalize the values between 0 - 1 
    r = r / 255
    b = b / 255
    g = g / 255

    #intialize HSV 
    h = 0.0 #give it 0.0 to initalize as floating point value
    s = 0.0
    v = 0.0 

    #get Vmax
    if(r > b and r > g):   
        v = r 
    elif(b > g):
        v = b  
    else:
        v = g 
    #getVmin
    if(r < b and r < g):
        vMin = r  
    elif(b < g):
        vMin = b 
    else:
        vMin = g


    #set the saturation value 
    if(v>0.0):
        s = (v - vMin)/ v 
    else:
        s = 0.0 #if v = 0.0 its a black image, no saturation

    # Difference from vMax to vMin
    diff = v - vMin

    # Are r,g,b equal?
    if(r == g and g == b):
        h = 0

   # Is the point within +/- 60 degrees of the red axis
    elif(r == v):
        h = 60 * (b - g) / diff
   # Is the point within +/- 60 degrees of the green axis
    elif(b == v):
        h = 120 + 60 * (g - r) / diff
    # IS the point within +/- 60 degrees of the blue axis
    elif(g == v):
        h = 240 + 60 * (r - b) / diff

    #convert to openCV ranges
    h = h / 2 
    s = np.interp(s,[0,1],[0,255])
    v = np.interp(v,[0,1],[0,255])



    hsv = np.uint8([[[h,s,v]]])

    return hsv

#hsv to rgb
def hsv_to_rgb(hsv): 
    #expect 3d hsv array

    #read in values
    h = hsv[0][0][0]
    s = (hsv[0][0][1])
    v = (hsv[0][0][2])

    # print("/n" + str(h))
    # print("/n" + str(s))
    # print("/n" + str(v))
    # print("/n")

    c = v * s 

    #find x value 
    temp = ( (h / 60) % 2 ) - 1
    if(temp < 0):
        temp = -1 * temp
    
    x = c * (1 - temp)

    m = v - c
    
   
    #determine r',g',b' based on h 

    if(h < 60):
        r = c
        g = x 
        b = 0 
    elif(h < 120):
        r = x
        g = c
        b = 0
    elif(h < 180):
        r = 0
        g = c
        b = x
    elif(h < 240):
        r = 0
        g = x
        b = c
    elif(h < 300):
        r = x
        g = 0
        b = c
    elif(h < 360):
        r = c
        g = 0
        b = x

    #determine final rgb value
    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    return np.uint8([[[r,g,b]]])

# Create a rgb value
rgb = np.uint8([[[200, 74, 55]]])

# Call the rgb_to_hsv function
hsv = rgb_to_hsv(rgb)

print(hsv)

# Create a hsv value
hsv = np.uint8([[[4, 185, 200]]])

# Call the hsv_to_rgb function
rgb = hsv_to_rgb(hsv)

print(rgb)

