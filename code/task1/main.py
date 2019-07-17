import cv2
import os

capture = cv2.VideoCapture(0)

name=input("名字：")
if not os.path.exists("E:/renlianshibie/facelmages/"+name):
    os.makedirs("E:/renlianshibie/facelmages/"+name)
i=0
while(i<600):
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    cv2.imwrite("E:/renlianshibie/facelmages/"+name+"/"+str(i)+".jpg", frame)
    i+=1

capture.release()
cv2.destroyAllWindows()