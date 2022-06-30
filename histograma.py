#!/usr/bin/python
import cv2 as cv
import numpy as np

def check_correctness(i, v):
    u = [0]*5
    for i in v:
        u[i//5] += 1

    return (i//5) == np.argmax(u)

def get_min_index(u, n):
    v = []
    for i in range(n):
        v.append(np.argmin(u))
        u.remove(u[v[-1]])

    return v

def get_max_index(u, n):
    v = []
    for i in range(n):
        v.append(np.argmax(u))
        u.remove(u[v[-1]])

    return v

def classifier(i, histogram):
    correl = []
    chisqr = []
    intersect = []
    bhattacharyya = []

    for j in range(len(histogram)):
        if (i != j):

            correl.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_CORREL))
            chisqr.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_CHISQR))
            intersect.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_INTERSECT))
            bhattacharyya.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_BHATTACHARYYA))

#    print("i: ", i, "bhatta:\n", bhattacharyya)

    return (check_correctness(i, get_max_index(correl, 4)),
            check_correctness(i, get_min_index(chisqr, 4)),
            check_correctness(i, get_max_index(intersect, 4)),
            check_correctness(i, get_min_index(bhattacharyya, 4)))

characters = ["b", "h", "l", "m", "mg"]

images = []

for c in characters:
    for i in range(5):
        images.append(
            cv.split(
                cv.imread(
                    cv.samples.findFile(
                        c+str(i+1)+".bmp"
                    )
                )
            )
        )

histogram = []

for i in images:
    aux = cv.calcHist(i, [0,1,2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256], accumulate=False)
    cv.normalize(aux, aux, norm_type=cv.NORM_MINMAX)
    histogram.append(aux)

#print(histogram)
v = []

for i in range(len(histogram)):
    v.append(classifier(i, histogram))

aux = [0,0,0,0]

for i in v:
    aux[0] += int(i[0])
    aux[1] += int(i[1])
    aux[2] += int(i[2])
    aux[3] += int(i[3])

print(aux)
