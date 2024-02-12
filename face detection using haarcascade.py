import cv2
image = cv2.imread('img.png')

#print(image.shape)  # Printing the shape to know it is color or Gray scale image

image = cv2.resize(image, (900,700))  # Resizing to desired shape
cv2.imshow("People",image)  # To show the image

#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray-scale
#cv2.imshow("People",image_gray)

# Loading the face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detections = face_detector.detectMultiScale(image, scaleFactor=1.08, minNeighbors=7, maxSize=(80,80), minSize=(30,30)) # Using scalefactor and other parameters to avoid False positive
#print(detections)

# For Showing Box
for (x, y, w, h) in face_detections:
    cv2.rectangle(image, (x,y), (x + w, y + h), (0,255,255), 3)
cv2.imshow("People",image)


eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_detections = eye_detector.detectMultiScale(image, scaleFactor=1.25, minNeighbors=5, maxSize=(70,70), minSize=(10,10))# Using scalefactor and other parameters to avoid False positive

for (x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x,y), (x + w, y + h), (0,0,255), 2)
cv2.imshow("People",image)


cv2.waitKey(0)