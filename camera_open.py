import cv2

cap =cv2.VideoCapture(1)
#cap.set(6,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('myvideo.avi',fourcc, 20.0,(2560,720))
i=0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    out.write(frame)
    cv2.imshow('camera',frame)
    key = cv2.waitKey(1)
    if  key == ord('q') or key == 27:
        break
    if key == ord("w"):
        cv2.imwrite("./%d.png" % i, frame)  # 自己设置拍摄的照片的存储位置
        print("Save images %d succeed!" % i)
        i += 1
cap.release()
out.release()
cv2.destroyAllWindows()



