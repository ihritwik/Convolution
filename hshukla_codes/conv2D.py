import cv2 as cv
import numpy as np
import math

def conv2(input_img,w,pad):
    
    #Check if the input image is RGB or GRAYSCALE. 
    # typeimg = 1 for RGB image and typeimg = 0 for GRAYSCALE image
    if len(size)==3:
        typeimg = 1
    
    else:
        typeimg = 0
    
    #This if is for computing convolution in GRAYSCALE image
    if typeimg == 0:
     
        #Get Padded Image
        pad_fun = findPadImage(pad)
        padded_Image = pad_fun(input_img)
  
        conv = findconvimg(w)
        im2 = conv(padded_Image)
  
        #Display Padded Image
        cv.imshow("Padded Image",padded_Image)
        
        # Now extract the original size image after cropping the pad
        cropped_c1 = im2[1:-1, 1:-1]
    
        pad_ci1 = cropped_c1.astype(np.uint8)
        im2 = im2.astype(np.uint8)
        
        #Display Convoluted Image
        #cv.imshow("Convoluted Image",im2)
        
               
        cv.imshow("CONVOLUTED and CROPPED to original SIZE",pad_ci1)
        
        #To print values of convoluted image for 1024 x 1024 image
        #print("Convoluted Impulse Output \n\n", pad_ci1[510:515, 510:515])
        
        cv.waitKey(0)
        cv.destroyAllWindows()

    #This part is for RGB image. If input image is RGB, then else part executes
    else:
        
        #Get Padded Image
                
        pad_fun = findPadImage(pad)
        k_blue = pad_fun(input_img[:,:,0])
        k_green = pad_fun(input_img[:,:,1])
        k_red = pad_fun(input_img[:,:,2])
       
        pad_image=cv.merge([k_blue,k_green,k_red])
  
        #Display Padded Image
        
        cv.imshow("Padded Image", pad_image)
        con_image = np.zeros((size[0]+2, size[1]+2,3))
        
        conv = findconvimg(w)
        ci_blue = conv(pad_image[:,:,0])
        ci_green = conv(pad_image[:,:,1])
        ci_red = conv(pad_image[:,:,2])
        
        #Crop the 3 channels of convolved image as the size of input image. Padding removed
        cropped_c1_blue = ci_blue[1:-1, 1:-1]
        cropped_c1_green = ci_green[1:-1, 1:-1]
        cropped_c1_red = ci_red[1:-1, 1:-1]
        
        #Change datatype of cropped image to uint8
        final_cropped_img_blue = cropped_c1_blue.astype(np.uint8)
        final_cropped_img_green = cropped_c1_green.astype(np.uint8)
        final_cropped_img_red = cropped_c1_red.astype(np.uint8)
       
        con_image = cv.merge([final_cropped_img_blue,final_cropped_img_green,final_cropped_img_red])
        
      
        cv.imshow("Convoluted Image and cropped",con_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    
#compute and find the padded image

def findPadImage(pad):
    switcher = {
       1: ZeroPad,
       2: WrapAround,
       3: CopyEdge,
       4: ReflectAcross,
         }
    return switcher.get(pad, "Invalid Choice") 


def findconvimg(w):
    switcher = {
            1: BoxFilter,
            2: First_order_derivative,
            3: Prewitt,
            4: Sobel,
            5: Roberts,
            }
    return switcher.get(w, "Invalid Choice")
    
def ZeroPad(input_img1):
    rat = input_img1.shape
    #print(rat)
    #cv.imshow("Inside zeropad  BEFORE",input_img1)
    r = size[0] + 2
    c = size[1] + 2
    new_img = np.zeros((r,c))
    new_img [1:-1, 1:-1] = input_img1 #****************Modifed********************
    flat = new_img.flatten()
    Zpad = flat.astype(np.uint8)
    image = np.reshape(Zpad,new_img.shape)
    #cv.imshow("Inside zeropad  AFTER",image)
    return image

def WrapAround(input_img1):
    r = size[0] + 2
    c = size[1] + 2
    new_img = np.zeros((r,c))
    new_img [1:-1, 1:-1] = input_img1
    flat = new_img.flatten()
    Zpad = flat.astype(np.uint8)
    image = np.reshape(Zpad,new_img.shape)
    
    for i in range (r-1):
        image[i+1, 0] = image[i+1,c-2]
        image[i+1, c-1] = image [i+1, 1]
    for j in range (c):
        image[0, j] = image[r-2,j]
        image[r-1, j] = image [1, j]
    return image

def ReflectAcross(input_img1):
    r = size[0] + 2
    c = size[1] + 2
    new_img = np.zeros((r,c))
    new_img [1:-1, 1:-1] = input_img1
    flat = new_img.flatten()
    Zpad = flat.astype(np.uint8)
    image = np.reshape(Zpad,new_img.shape)
    
    for i in range (r-1):
        image[i+1, 0] = image[i+1,2]
        image[i+1, c-1] = image[i+1, c-3]
    for j in range (c):
        image[0, j] = image[2,j]
        image[r-1, j] = image[r-3, j]
    return image

def CopyEdge(input_img1):
    r = size[0] + 2
    c = size[1] + 2
    new_img = np.zeros((r,c))
    new_img [1:-1, 1:-1] = input_img1
    flat = new_img.flatten()
    Zpad = flat.astype(np.uint8)
    image = np.reshape(Zpad,new_img.shape)
    
    for i in range (r-1):
        image[i+1, 0] = image[i+1,1]
        image[i+1, c-1] = image[i+1, c-2]
    for j in range (c):
        image[0, j] = image[1,j]
        image[r-1, j] = image[r-2, j]
    return image

# Filters start

def BoxFilter(padded_Image1):
    r = size[0] + 2
    c = size[1] + 2
    kernal = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)
    convolve_img = np.zeros((r,c))

    for i in range(r-2):
        for j in range(c-2):
            for p in range (3):
                convolve_img[i+1,j+1] = convolve_img[i+1,j+1] + ((kernal[p,0]*padded_Image1[i+p,j])
                                                           + (kernal[p,1]*padded_Image1[i+p,j+1]) 
                                                            + (kernal[p,2]*padded_Image1[i+p,j+2]))
    return convolve_img

def First_order_derivative(padded_Image1):
    r = size[0] + 2
    c = size[1] + 2

    convolve_img1 = np.zeros((r,c))
    convolve_img2 = np.zeros((r,c))
    convolve_img = np.zeros((r,c))
    #Derivative Kernal
    kernal_x = [(-1), (1)]
    kernal_y = [(-1),(1)]
    #kernal_y2 = [(1),(-1)]
    
    for i in range(r):
        for j in range (c-1):
            convolve_img1[i][j] = convolve_img1[i][j] + ((kernal_x[0]*padded_Image1[i][j])
                                                           + (kernal_x[1]*padded_Image1[i][j+1]))
    for i in range(r-1):
        for j in range (c):
            convolve_img2[i][j] += ((kernal_y[0]*padded_Image1[i][j])
                                                           + (kernal_y[1]*padded_Image1[i+1][j]))                                                  
    for k in range(r):
        for l in range (c):
            convolve_img[k][l] = (math.sqrt(pow((convolve_img1[k][l]),2) + pow((convolve_img2[k][l]),2)))      
    return convolve_img


def Prewitt(padded_Image1):
    r = size[0] + 2
    c = size[1] + 2
    
    #Prewitt Kernal 
    kernal_prewitt_x = [(-1,0,1),(-1,0,1),(-1,0,1)]
    kernal_prewitt_y = [(1,1,1),(0,0,0),(-1,-1,-1)]
    
    convolve_img1 = np.zeros((r,c))
    convolve_img2 = np.zeros((r,c))
    convolve_img = np.zeros((r,c)) 
    
    for i in range(r-2):
        for j in range(c-2):
            for p in range (3):
                convolve_img1[i+1][j+1] = convolve_img1[i+1][j+1] + ((kernal_prewitt_x[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_prewitt_x[p][1]*padded_Image1[i+p][j+1]) 
                                                            + (kernal_prewitt_x[p][2]*padded_Image1[i+p][j+2]));
                convolve_img2[i+1][j+1] = convolve_img2[i+1][j+1] + ((kernal_prewitt_y[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_prewitt_y[p][1]*padded_Image1[i+p][j+1]) 
                                                            + (kernal_prewitt_y[p][2]*padded_Image1[i+p][j+2]));
    for k in range(r):
        for l in range (c):
            convolve_img[k][l] = math.sqrt(pow((convolve_img1[k][l]),2) + pow((convolve_img2[k][l]),2))       
    return convolve_img

def Sobel(padded_Image1):
    r = size[0] + 2
    c = size[1] + 2
    
    #Sobel Kernal
    kernal_sobel_x = [(-1,0,1),(-2,0,2),(-1,0,1)]
    kernal_sobel_y = [(1,2,1),(0,0,0),(-1,-2,-1)]
    
    convolve_img1 = np.zeros((r,c))
    convolve_img2 = np.zeros((r,c))
    convolve_img = np.zeros((r,c)) 
    
    for i in range(r-2):
        for j in range(c-2):
            for p in range (3):
                convolve_img1[i+1][j+1] = convolve_img1[i+1][j+1] + ((kernal_sobel_x[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_sobel_x[p][1]*padded_Image1[i+p][j+1]) 
                                                            + (kernal_sobel_x[p][2]*padded_Image1[i+p][j+2]));
                convolve_img2[i+1][j+1] = convolve_img2[i+1][j+1] + ((kernal_sobel_y[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_sobel_y[p][1]*padded_Image1[i+p][j+1]) 
                                                            + (kernal_sobel_y[p][2]*padded_Image1[i+p][j+2]));
    for k in range(r):
        for l in range (c):
            convolve_img[k][l] = math.sqrt(pow((convolve_img1[k][l]),2) + pow((convolve_img2[k][l]),2))       

    return convolve_img

def Roberts(padded_Image1):
    r = size[0] + 2
    c = size[1] + 2
    
    #Roberts Kernal
    kernal_roberts_x = [(0,1),(-1,0)]
    kernal_roberts_y = [(1,0),(0,-1)]
    
    convolve_img1 = np.zeros((r,c))
    convolve_img2 = np.zeros((r,c))
    convolve_img = np.zeros((r,c)) 
    
    for i in range(r-2):
        for j in range(c-2):
            for p in range (2):
                convolve_img1[i+1][j+1] = convolve_img1[i+1][j+1] + ((kernal_roberts_x[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_roberts_x[p][1]*padded_Image1[i+p][j+1]));
                                                           
                convolve_img2[i+1][j+1] = convolve_img2[i+1][j+1] + ((kernal_roberts_y[p][0]*padded_Image1[i+p][j])
                                                           + (kernal_roberts_y[p][1]*padded_Image1[i+p][j+1]));
    for k in range(r):
        for l in range (c):
            convolve_img[k][l] = math.sqrt(pow((convolve_img1[k][l]),2) + pow((convolve_img2[k][l]),2))    
    return convolve_img
    
    

if __name__ == "__main__":
    
    print("\n")
    print("Hello..This is Convolution of images with different padding.")
    print("\n")
    print("Select Type of Padding to be used. ")
    print(" 1. Zero Padding \n 2. Wrap around \n 3. Copy Edge \n 4. Reflect across edge")
    print("\n")
    pad = int(input("Use number to specify the type\n"))
    
    print("Now, select the type of kernal to be used for CONVOLUTION.\n")
    print(" 1. Box Filter \n 2. First order derivative \n 3. Prewitt \n 4. Sobel \n 5. Roberts  \n")
    w = int(input("Use number to specify the type\n"))
    
    #Check with Lena.png
    input_img = cv.imread("lena.png")
    
    #Check with Wolves.png
    #input_img = cv.imread("wolves.png")
    
    #Now for part (b) to test the code in 1024 X 1024 image 
    #To check this part, uncomment line 41, 310 and 311. Add comment on line 305
    
    #input_img = np.zeros((1024,1024))
    #input_img[512][512] = 255

    size = input_img.shape
    print (size)
    #Calling conv2 function
    conv2(input_img,w,pad)
    
   


    
    
    