import cv2 as cv

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
    histogram.append(
        cv.calcHist(i, [0,1,2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256], accumulate=False)
    )

#print(histogram)

correl = []
chisqr = []
intersect = []
bhattacharyya = []

for i in range(len(histogram)):
    for j in range(len(histogram)):
        if (i != j):

            correl.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_CORREL))
            chisqr.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_CHISQR))
            intersect.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_INTERSECT))
            bhattacharyya.append(cv.compareHist(histogram[i], histogram[j], cv.HISTCMP_BHATTACHARYYA))

print(correl)
print(chisqr)
print(intersect)
print(bhattacharyya)
