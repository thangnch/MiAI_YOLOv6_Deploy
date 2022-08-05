from my_yolov6 import my_yolov6
import cv2

yolov6_model = my_yolov6("yolov6s.pt","cpu","data/coco.yaml", 640, True)


# define a video capture object
vid = cv2.VideoCapture(1)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()