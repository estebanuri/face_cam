import cv2
import dlib


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_w(img, new_w = 640):
    w, h = img.shape[1], img.shape[0]
    new_h = int((new_w / w) * h)
    ret = cv2.resize(img, (new_w, new_h))
    return ret

def shrink_if_large(img, max=1024):
    w, h = img.shape[1], img.shape[0]
    if w > max or h > max:
        img = resize_w(img, max)

    return img

def imshow(win_name, image):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image)


def load_models(args):
    face_detector = dlib.get_frontal_face_detector()

    # NOTE: dlib 68 landmarks model can be downloaded from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    model_path = args.landmarks_model
    shape_predictor = dlib.shape_predictor(model_path)
    return face_detector, shape_predictor
