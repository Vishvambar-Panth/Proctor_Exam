import cv2
from matplotlib import pyplot as plt
videoCaptureObject = cv2.VideoCapture(0)
result = True
if videoCaptureObject.isOpened():
    print('Yes')
    ret,frame = videoCaptureObject.read()
    print(ret)
    cv2.imwrite("NewPicture.jpg",frame)
    result = False

videoCaptureObject.release()
cv2.destroyAllWindows()
