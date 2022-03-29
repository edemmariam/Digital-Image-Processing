from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics

filename = "portrett.png"
f = imread(filename, as_gray=True)
N, M = f.shape
totalPixels = N*M

#finding p(i)
def hist(f):
    h = np.zeros(256)
    p = np.zeros(256)
    for i in range(256):
        h[i] = (f == i).sum()
        p[i] = h[i] / totalPixels
    return p


#finding mean and variance
def mean_sd():
    p = hist(f)
    mean = 0
    variance = 0
    for i in range(256):
        mean += i*p[i]
        variance += (i**2)*p[i]
        sd = math.sqrt(variance-(mean**2))
    return mean, sd

#transforming
def transform():
    mean, sd = mean_sd()
    sdT = 64 #from task
    meanT = 127 #from task
    a = sdT/sd
    b = (meanT) - (a*mean)

    newf = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            newf[i][j] = (a*(f[i][j])) + b #using eqation to find new image
    return newf



def find_coefficients():
    portrettLeftEye = np.array([91, 85, 1])
    portrettRightEye = np.array([71, 121, 1])
    portrettNose = np.array([97, 117, 1])
    portrettLeftMouth= np.array([117, 118, 1])
    portrettRightMouth= np.array([103, 143, 1])

    geoLeftEye = np.array([259,170])
    geoRightEye = np.array([259,342])
    geoNose = np.array([376,257])
    goeLeftMouth= np.array([442,193])
    goeRightMouth= np.array([442,317])

    m = np.array([portrettLeftEye, portrettRightEye, portrettNose, portrettLeftMouth, portrettRightMouth])
    mx = np.array([geoLeftEye[0], geoRightEye[0], geoNose[0], goeLeftMouth[0], goeRightMouth[0]])
    my = np.array([geoLeftEye[1], geoRightEye[1], geoNose[1], goeLeftMouth[1], goeRightMouth[1]])

    mT = np.transpose(m)
    mxT = np.transpose(mx)
    myT = np.transpose(my)

    #Because I do not have square matrix, I have to use least squre problem
    #which solves the equation Ax = b as closely as possible
    #b = (A^T * A)^(-1) * A^T * x
    a = np.linalg.inv(mT@m) @ mT @mxT
    b = np.linalg.inv(mT@m) @ mT @myT

    matrix = np.array([np.transpose(a), np.transpose(b), [0, 0, 1]])
    return matrix



filename2 = "geometrimaske.png"
g = imread(filename2, as_gray=True)
O, P = g.shape

def forwardMapping():
    forwardImg = np.zeros((O,P))
    for i in range(N):
        for j in range(M):
            xyVec = [i, j, 1]
            newxy = matrix @ xyVec
            x = round(newxy[0])
            y = round(newxy[1])
            if x < O and y < P:
                forwardImg[x][y] = newf[i][j]
    return forwardImg



def binaryInterpolation():
    interpolationImg = np.zeros((O,P))
    matrixInv = np.linalg.inv(matrix)
    for i in range(O):
        for j in range(P):
            xyVec = [i, j, 1]
            newxy = matrixInv@xyVec
            x = newxy[0]
            y = newxy[1]

            if 0 <= x <= O and 0 <= y <= P:
                x0 = math.floor(x)
                y0 = math.floor(y)
                x1 = math.ceil(x)
                y1 = math.ceil(y)
                dx = x - x0
                dy = y - y0
                p = newf[x0][y0] + ((newf[x1][y0]-newf[x0][y0]))*dx
                q = newf[x0][y1]+ ((newf[x1][y1]-newf[x0][y1])) * dx
                interpolationImg[i][j] = p+((q-p)*dy)

    return interpolationImg



NNImg = np.zeros((O,P))
def nearsetNeightbor():
    matrixInv = np.linalg.inv(matrix)
    for i in range(O):
        for j in range(P):
            xyVec = [i, j, 1]
            newxy = matrixInv @ xyVec
            x = round(newxy[0])
            y = round(newxy[1])

            #if x in range(O) and y in range(P)):
            if 0 <= x <= O and 0 <= y <= P:
                NNImg[i][j] = newf[x][y]
            else:
                return 0
    return NNImg





newf = transform()
meanNewf = np.mean(newf)
sdNewf = np.std(newf)


matrix = find_coefficients()
forwardImg = forwardMapping()
interpolationImg = binaryInterpolation()
NN = nearsetNeightbor()





plt.figure()
plt.imshow(f,cmap='gray', vmin=0,vmax=255)
plt.title("Orginal")
plt.show()

plt.figure()
plt.imshow(newf,cmap='gray', vmin=0,vmax=255)
plt.title("Changed-grayscale")
plt.show()

plt.figure()
plt.imshow(forwardImg, cmap="gray")
plt.title("Forwards Mapping")
plt.show()

plt.figure()
plt.imshow(interpolationImg, cmap="gray")
plt.title("Backwards: interpolation")
plt.show()

plt.figure()
plt.imshow(NNImg, cmap="gray")
plt.title("Backwards: Nearset Neightbor")
plt.show()
