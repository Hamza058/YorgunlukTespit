import cv2 as cv
import mediapipe as mp
import time
import math

from numpy import greater
import utils

# variables
frame_counter =0

# constants
FONTS =cv.FONT_HERSHEY_COMPLEX

Open_lips=0
Total_yawn=0

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

map_face_mesh = mp.solutions.face_mesh
# camera object
camera = cv.VideoCapture(0)
# landmark detection function

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def yawnRatio(img, landmarks, lips_up,lips_bottom):
    # vertical line
    lip_top = landmarks[lips_up[13]]
    lip_bottom = landmarks[lips_bottom[16]]

    lvDistance = euclaideanDistance(lip_top, lip_bottom)

    return lvDistance

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera
        if not ret:
            break # no more frames break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3 )
            ratio = yawnRatio(frame, mesh_coords, UPPER_LIPS,LOWER_LIPS)
            if Total_yawn==1:
                tic=str(time.perf_counter()).split('.')
                toc=int(tic[0])-4

                if toc > 60:
                    if Total_yawn>5:
                        utils.colorBackgroundText(frame, f'1dk da esneme sayiniz: {Total_yawn}', FONTS, 0.7, (30, 130), 2)
                        utils.colorBackgroundText(frame, '!!! Uykunuz Geliyor !!!', FONTS, 0.7, (250, 70), 2)

            if ratio > 20:
                Open_lips+=1
            else:
                if Open_lips>50:
                    Total_yawn+=1
                    Open_lips=0

            utils.colorBackgroundText(frame, f'Total Yawn: {Total_yawn}', FONTS, 0.7, (30, 100), 2)

            # Changes for Thumbnail of youtube Video
            [cv.circle(frame,mesh_coords[p], 1, utils.GREEN , -1, cv.LINE_AA) for p in LIPS]

        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()