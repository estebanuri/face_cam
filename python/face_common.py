import cv2

def draw_landmarks(image, landmarks, face=None):

    ret = image.copy()

    if face is not None:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(ret, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for i in range(0, landmarks.num_parts):
        coord = (landmarks.part(i).x, landmarks.part(i).y)
        cv2.circle(ret, coord, 2, (0, 255, 0), -1)

    return ret


def get_landmarks(models, image):

    face_detector, shape_predictor = models
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    if len(faces) == 0:
        raise Exception("face not found")

    if len(faces) > 1:
        #raise Exception("multiple faces were found")
        print("waring multiple faces found using the first one")
    # mask = np.zeros((frame.shape[1], frame.shape[0]), dtype='uint8')
    #mask = np.zeros_like(image)

    face = faces[0]
    landmarks = shape_predictor(gray, face)
    if landmarks.num_parts < 3:
        raise Exception("too few landmarks")

    # landmarks = landmarks_to_np(landmarks)
    # #hull = cv2.convexHull(landmarks)
    # ret = delunay(frame, landmarks)

    return face, landmarks

def get_landmarks_points(landmarks):
    landmark_points = []
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, landmarks.num_parts):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmark_points.append((x, y))
    return landmark_points