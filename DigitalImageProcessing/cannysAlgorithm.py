from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

filename = "cellekjerner.png"
image = imread(filename, as_gray=True)
N, M = image.shape


def conv(imageIn, filter):
    imageOut = np.zeros((N, M))

    #rotate my image 90 deg twice (k=2)
    filter = np.rot90(filter, k=2)

    #a have a squre filter, tha is why I use "filter.shape[0]""
    k = filter.shape[0]

    #np.floor: same as (k-1)/2
    n = int(np.floor(k/2))

    #extend image with ero padding
    imageExtended = np.pad(image, ((n,n),(n,n)), "constant")

    for i in range(N):
        for j in range(M):
            math = imageExtended[i:i+k, j:j+k]
            imageOut[i][j] = np.sum(np.multiply(math, filter))
    return imageOut


def gauss(sigma):
    sum = 0

    #set dimentions for my filter
    dim =  round(1 + 8*sigma)

    #find center
    center = int(( dim - 1) / 2)
    filter = np.zeros((dim, dim))

    #using for-loops to centre my image to get biggest value in the origin
    for x in range(-center, center+1):
        for y in range(-center, center+1):
            g = (1/(2* np.pi * sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            filter[center + x][center + y] = g
            sum += g
    filter = filter /sum
    return filter


def gradient(imageIn):
    #definding hx, hy
    hx = np.array([[0,1,0], [0,0,0], [0,-1,0]])
    hy = np.array([[0,0,0], [1,0,-1], [0,0,0]])

    #using def conv to find new image
    gx = conv(imageIn, hx)
    gy = conv(imageIn, hy)

    magnitude = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx)
    return magnitude, theta

def thinningEdges(imageIn, direction):
    N, M = imageIn.shape
    fOut = np.zeros((N,M), dtype=np.int32)

    #from rad to deg: same as: (* 180. / np.pi)
    direction = np.rad2deg(direction)
    direction[direction < 0] += 180

    for i in range(1,N-1):
        for j in range(1,M-1):
            q = 255
            r = 255

            if (0 <= direction[i][j] < 22.5) or (157.5 <= direction[i][j] <=180):
                q = imageIn[i][j+1]
                r = imageIn[i][j-1]
            elif (22.5 <= direction[i][j] < 67.5):
                q = imageIn[i+1][j-1]
                r = imageIn[i-1][j+1]
            elif (67.5 <= direction[i][j] < 112.5):
                q = imageIn[i+1][j]
                r = imageIn[i-1][j]
            elif (112.5 <= direction[i][j] < 157.5):
                q = imageIn[i-1][j+1]
                r = imageIn[i+1][j-1]
            if (imageIn[i][j] >= q) and (imageIn[i][j] >= r):
                fOut[i][j] = imageIn[i][j]
            else:
                fOut[i][j] = 0
    return fOut

def hysteresis(imageIn, Th, Tl):
    N, M = imageIn.shape
    for i in range(1, N-1):
        for j in range(1, M-1):
            if (imageIn[i][j] == Th):
                if ((imageIn[i+1][j-1] == Tl) or (imageIn[i+1][j] == Tl) or (imageIn[i+1][j+1] == Tl)
                    or (imageIn[i][j-1] == Tl) or (imageIn[i][j+1] == Tl)
                    or (imageIn[i-1][j-1] == Tl) or (imageIn[i-1][j] == Tl) or (imageIn[i-1][j+1] == Tl)):
                    imageIn[i][j] = Tl
                else:
                    imageIn[i][j] = 0
    return imageIn



gaussFilter = gauss(3)
imageConv = conv(image, gaussFilter)
m, t = gradient(imageConv)
thinEdg = thinningEdges(m, t)
finalImage = hysteresis(thinEdg, 10, 20)

plt.figure()
plt.title("Original Image")
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.show()

plt.figure()
plt.title("Finding Edges")
plt.imshow(finalImage, cmap="gray", vmin=0, vmax=255)
plt.show()
