import cv2
import imutils

cam = cv2.VideoCapture(0)   #primary camera
firstframe = None
area = 500

while True:
    _, img = cam.read()
    if not _:
        print("failed to access camera")
        break
    text = "Normal"
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #convert coloured to grayscale image
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)     #Smoothening(Blur) the image

    if firstframe is None:
        firstframe=gaussianImg
        continue

    imgdiff = cv2.absdiff(firstframe, gaussianImg)
    threshImg = cv2.threshold(imgdiff, 30, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        text = "Moving Object Detected"

    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65,126,72), 2)
    cv2.imshow("camera", img)

    Key = cv2.waitKey(10)
    if Key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()