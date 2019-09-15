import cv2 
import numpy as np 

beta = -np.pi/3
matrix = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0]])
print(beta)
img = cv2.imread("img.jpg") 
rows, cols, _ = np.shape(img)

print(matrix)
matrix = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0]])
img1 = cv2.warpAffine(img,matrix,(cols,rows))

matrix = np.array([[1., 0., 20.], [0., 1., 40.]])
img2 = cv2.warpAffine(img,matrix,(cols,rows))

matrix = np.array([[2., 0., 0.], [0., 1., 0.]])
img3 = cv2.warpAffine(img,matrix,(cols,rows))

cv2.imwrite("roat.png", img1)
cv2.imwrite("move.png", img2)
cv2.imwrite("strat.png", img3)