import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0) # The 0 'zero' here represents webcam if you have external cam use 1 instead

while True:
    ret, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(100,100))

    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press q in keyboard to exit from running
        break

video_capture.release()
cv2.destroyAllWindows()