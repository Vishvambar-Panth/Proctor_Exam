import cv2
import numpy as np
import math
from person_and_phone import *
from head_pose_estimation import *
# from mouth_opening_detector import *
from gaze_tracking import GazeTracking

yolo = YoloV3()
gaze = GazeTracking()
load_darknet_weights(yolo, 'models/yolov3.weights')

face_model = get_face_detector()
landmark_model = get_landmark_model()


def mouth_open(img3):
    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0]*5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0]*3
    font = cv2.FONT_HERSHEY_SIMPLEX
    rects = find_faces(img3, face_model)
    for rect in rects:
        shape = detect_marks(img3, landmark_model, rect)
        cnt_outer = 0
        cnt_inner = 0
        draw_marks(img3, shape[48:])
        for i, (p1, p2) in enumerate(outer_points):
            if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                cnt_outer += 1
        for i, (p1, p2) in enumerate(inner_points):
            if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                cnt_inner += 1
        if cnt_outer > 3 and cnt_inner > 2:
            print('Mouth open')
            # cv2.putText(img3, 'Mouth open', (30, 30), font,
            #         1, (0, 255, 255), 2)
    return img3


def eye_position(img2):
    gaze.refresh(img2)

    img2 = gaze.annotated_frame()
    text = ""

    if gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    if text!= "":
        print(text)
    # cv2.putText(img2, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    # cv2.putText(img2, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(img2, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    return img2

def person_and_phone_detector(img0):
    img_temp= cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img_temp = cv2.resize(img_temp, (320, 320))
    img_temp = img_temp.astype(np.float32)
    img_temp = np.expand_dims(img_temp, 0)
    img_temp = img_temp / 255
    class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
    boxes, scores, classes, nums = yolo(img_temp)
    count=0
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count +=1
        if int(classes[0][i] == 67):
            print('Mobile Phone detected')
        if int(classes[0][i] == 74):
            print('Book detected')
    if count == 0:
        print('No person detected')
    elif count > 1:
        print('More than one person detected')

    img0 = draw_outputs(img0, (boxes, scores, classes, nums), class_names)
    return img0


def head_estimator(img1):

    size = img1.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    faces = find_faces(img1, face_model)
    for face in faces:
        marks = detect_marks(img1, landmark_model, face)

        image_points = np.array([
                                marks[30],     # Nose tip
                                marks[8],     # Chin
                                marks[36],     # Left eye left corner
                                marks[45],     # Right eye right corne
                                marks[48],     # Left Mouth corner
                                marks[54]      # Right mouth corner
                            ], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img1, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img1, rotation_vector, translation_vector, camera_matrix)

        # cv2.line(img1, p1, p2, (0, 255, 255), 2)
        # cv2.line(img1, tuple(x1), tuple(x2), (255, 255, 0), 2)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90

        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            ang2 = 90

        if ang1 >= 48:
            print('Head down')
            # cv2.putText(img1, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
        elif ang1 <= -48:
            print('Head up')
            # cv2.putText(img1, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

        if ang2 >= 48:
            print('Head right')
            # cv2.putText(img1, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
        elif ang2 <= -48:
            print('Head left')
            # cv2.putText(img1, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

        # cv2.putText(img1, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
        # cv2.putText(img1, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

        return img1

cap = cv2.VideoCapture(0)
while (True):
    ret, img = cap.read()

    #-----------------------------------------------------------------------------#
    if ret:
        img1_fin = head_estimator(img)
        img0_fin = person_and_phone_detector(img)
        img2_fin = eye_position(img)
        img3_fin = mouth_open(img)
    else:
        break

    # cv2.imshow('Tab1', img0_fin)
    # cv2.imshow('Tab2', img1_fin)
    # cv2.imshow('Tab3', img2_fin)
    cv2.imshow('Tab', img3_fin)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
