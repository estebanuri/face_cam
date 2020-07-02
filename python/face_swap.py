import numpy as np
import cv2
import argparse
from common import load_models, imshow


def parse_args():

    parser = argparse.ArgumentParser(
        description="given two images of faces F1 and F2, swap the faces in those images"
    )

    parser.add_argument(
        '-f1', '--face1',
        required=True,
        help='input face 1 image file'
    )

    parser.add_argument(
        '-f2', '--face2',
        required=True,
        help='input face 2 image file'
    )

    parser.add_argument(
        '-m', '--landmarks_model',
        default="models/shape_predictor_68_face_landmarks.dat",
        help='dlib landmakrs shape predictor model path'
    )

    parser.add_argument(
        '-o1', '--output1',
        required=False,
        help='file output for image1 result'
    )

    parser.add_argument(
        '-o2', '--output2',
        required=False,
        help='file output for image1 result'
    )

    parser.add_argument(
        '-s', '--show',
        default=True,
        help='shows the results on screen and waits a key'
    )

    args = parser.parse_args()
    return args





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

# def draw_triangles(image, triangles):
#
#     ret = image.copy()
#     _, contours = cv2.findContours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for t in triangles:
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#         cv2.line(ret, pt1, pt2, color=(255, 0, 0), thickness=1)
#         cv2.line(ret, pt1, pt3, color=(0, 255, 0), thickness=1)
#         cv2.line(ret, pt2, pt3, color=(0, 0, 255), thickness=1)
#
#     # cnt = triangles.reshape(-1, 6, 2).astype(np.int32)
#     # #contours = np.array(contours).reshape((-1, 1, 2)).astype(np.int32)
#     # cv2.drawContours(ret, cnt, -1, (0, 255, 127), 1)
#     # cv2.imshow("img", ret)
#     # cv2.waitKey()
#
#     return ret

def draw_triangles(image, triplets, points):

    ret = image.copy()
    _, contours = cv2.findContours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for triplet in triplets:
        pt1 = points[triplet[0]]
        pt2 = points[triplet[1]]
        pt3 = points[triplet[2]]
        cv2.line(ret, pt1, pt2, color=(255, 0, 0), thickness=1)
        cv2.line(ret, pt1, pt3, color=(0, 255, 0), thickness=1)
        cv2.line(ret, pt2, pt3, color=(0, 0, 255), thickness=1)

    # cnt = triangles.reshape(-1, 6, 2).astype(np.int32)
    # #contours = np.array(contours).reshape((-1, 1, 2)).astype(np.int32)
    # cv2.drawContours(ret, cnt, -1, (0, 255, 127), 1)
    # cv2.imshow("img", ret)
    # cv2.waitKey()

    return ret


def get_landmarks_points(landmarks):
    landmark_points = []
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, landmarks.num_parts):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmark_points.append((x, y))
    return landmark_points

def delunay(landmarks_points, fast=False):

    np_points = np.array(landmarks_points, np.int32)

    # creates a convex hull that encloses all the landmarks


    # creates a box that encloses the convex hull
    #rect1 = cv2.boundingRect(convexhull)
    rect = cv2.boundingRect(np_points)
    #x1, y1, w, h = rect
    # ts = []
    # create an empty image
    # mask = np.zeros((image.shape[1], image.shape[0]), dtype='uint8')
    # cv2.polylines(image, [convexhull], True, (255, 0, 0), 1)
    # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
    # cv2.imshow("image", image)
    # cv2.waitKey()

    subdiv = cv2.Subdiv2D(rect)
    if fast:
        convexhull = cv2.convexHull(np_points)
        cvhp = list(convexhull)
        cvhp.append(landmarks_points[33])
        cvhp.append(landmarks_points[36])
        cvhp.append(landmarks_points[45])
        cvhp.append(landmarks_points[48])
        cvhp.append(landmarks_points[54])
        cvhp.append(landmarks_points[31])
        cvhp.append(landmarks_points[35])
        cvhp.append(landmarks_points[39])
        cvhp.append(landmarks_points[42])

        subdiv.insert(cvhp)

    else:
        subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()

    triangle_indices = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        idx1 = np.where((np_points == pt1).all(axis=1))[0][0]
        idx2 = np.where((np_points == pt2).all(axis=1))[0][0]
        idx3 = np.where((np_points == pt3).all(axis=1))[0][0]
        # idx1 = np.argwhere(np_points == pt1)[0][0]
        # idx2 = np.argwhere(np_points == pt2)[0][0]
        # idx3 = np.argwhere(np_points == pt3)[0][0]
        index = tuple(sorted([idx1, idx2, idx3]))
        #index = (idx1, idx2, idx3)
        triangle_indices.append(index)

    return triangles, triangle_indices


    #landmarks_points = [(pt.x, pt.y)]
    # mask =
    # points = np.array(landmarks_points, np.int32)
    # convexhull = cv2.convexHull(points)
    # #cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    # cv2.fillConvexPoly(mask, convexhull, 255)
    # face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    # # Delaunay triangulation
    # rect = cv2.boundingRect(convexhull)
    # subdiv = cv2.Subdiv2D(rect)
    # subdiv.insert(landmarks_points)
    # triangles = subdiv.getTriangleList()
    # triangles = np.array(triangles, dtype=np.int32)


def warp_triangle(image1, image2, result1, result2, triplet, lp1, lp2, two_ways=True):

    tr1 = [lp1[triplet[0]], lp1[triplet[1]], lp1[triplet[2]]]
    tr2 = [lp2[triplet[0]], lp2[triplet[1]], lp2[triplet[2]]]

    tr1 = np.array(tr1)
    tr2 = np.array(tr2)

    rect1 = cv2.boundingRect(tr1)
    rect2 = cv2.boundingRect(tr2)

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    tl1 = (x1, y1)
    tl2 = (x2, y2)

    # removes offset to triangles, by subtracting its upper-left bounding box coordinate
    centered_tr1 = tr1 - tl1
    centered_tr2 = tr2 - tl2

    ctr1 = np.float32(centered_tr1)
    ctr2 = np.float32(centered_tr2)

    T21 = cv2.getAffineTransform(ctr2, ctr1)
    cropped_src_2 = image2[y2: y2 + h2, x2: x2 + w2]
    cropped_tgt_1 = result1[y1: y1 + h1, x1: x1 + w1]
    warped_2_1 = cv2.warpAffine(cropped_src_2, T21, dsize=(w1, h1), borderMode=cv2.BORDER_REFLECT101)
    #ht2 = cv2.convertPointsToHomogeneous(centered_tr2)
    #warped_2_1 = np.round(warped_2_1).astype(dtype='uint8')
    #tt1 = np.round(np.dot(T21, centered_tr2))
    # tt1 = cv2.transform(ht2, T21).reshape(-1, 2)
    # diff = np.linalg.norm(tt1 - ctr1)
    # if diff != 0:
    #     print("!differs!", diff)
    mask1 = np.zeros((h1, w1, 3), np.uint8)
    cv2.fillConvexPoly(mask1, centered_tr1, (255, 255, 255))
    patch1 = ((1 - mask1/255.0) * cropped_tgt_1 + mask1/255.0 * warped_2_1).astype(dtype='uint8')
    result1[y1: y1 + h1, x1: x1 + w1] = patch1

    if two_ways:
        T12 = cv2.getAffineTransform(ctr1, ctr2)
        cropped_src_1 = image1[y1: y1 + h1, x1: x1 + w1]
        warped_1_2 = cv2.warpAffine(cropped_src_1, T12, dsize=(w2, h2), borderMode=cv2.BORDER_REFLECT101)
        cropped_tgt_2 = result2[y2: y2 + h2, x2: x2 + w2]
        mask2 = np.zeros((h2, w2, 3), np.uint8)
        cv2.fillConvexPoly(mask2, centered_tr2, (255, 255, 255))
        patch2 = ((1 - mask2 / 255.0) * cropped_tgt_2 + mask2 / 255.0 * warped_1_2).astype(dtype='uint8')
        result2[y2: y2 + h2, x2: x2 + w2] = patch2




def blend(input_image, face_mask, landmark_points, clone_mode=cv2.NORMAL_CLONE):

    convexhull = cv2.convexHull(np.array(landmark_points, np.int32))
    mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype='uint8')
    cv2.fillConvexPoly(mask, convexhull, 255)
    (x1, y1, w1, h1) = cv2.boundingRect(convexhull)
    if clone_mode is None:
        out = cv2.bitwise_and(input_image, input_image, mask=255 - mask) + face_mask
        #out = ((1 - mask/255.0) * input_image + face_mask).astype(dtype='uint8')
    # elif clone_mode == 'poisson':
    #     center1 = (int((x1 + x1 + w1) / 2), int((y1 + y1 + h1) / 2))
    #     out = poisson_edit(face_mask, input_image, mask, center1)
    else:
        center1 = (int((x1 + x1 + w1) / 2), int((y1 + y1 + h1) / 2))
        out = cv2.seamlessClone(face_mask, input_image, mask, center1, clone_mode)
        #out[ipm > 0] = 255
    return out


def precompute_face(models, image):
    face, landmarks = get_landmarks(models, image)
    lps = get_landmarks_points(landmarks)

    pc = {}
    pc['face'] = face
    pc['landmarks'] = landmarks
    pc['landmark_points'] = lps
    return pc


# def precompute_swap(models, image, params, id=2):
#
#     pc = precompute_face(models, image)
#
#     params['precomputed' + str(id)] = pc
#
#     _, triplets = delunay(lps)
#     params['triplets'] = triplets


def build_triplets(pre_computed):
    lps = pre_computed['landmark_points']
    _, triplets = delunay(lps)
    return triplets

def build_params(
        pre_computed1=None,
        pre_computed2=None,
        triplets=None,
        two_ways=True):

    params = {}
    params['precomputed1'] = pre_computed1
    params['precomputed2'] = pre_computed2
    params['triplets'] = triplets
    params['smooth1'] = True
    params['two_ways'] = two_ways

    #clone_mode = None
    #clone_mode = 'poisson'
    clone_mode = cv2.NORMAL_CLONE
    params['clone'] = clone_mode

    return params


def swap_faces(models,
               image1,
               image2,
               params=build_params(),
               debug=False):

    if params['precomputed1'] is not None:
        pc = params['precomputed1']
        face1 = pc['face']
        landmarks1 = pc['landmarks']
        lp1 = pc['landmark_points']
    else:
        face1, landmarks1 = get_landmarks(models, image1)
        lp1 = get_landmarks_points(landmarks1)

    if params['precomputed2'] is not None:
        pc = params['precomputed2']
        face2 = pc['face']
        landmarks2 = pc['landmarks']
        lp2 = pc['landmark_points']
    else:
        face2, landmarks2 = get_landmarks(models, image2)
        lp2 = get_landmarks_points(landmarks2)

    if params['triplets'] is None:
        _, triplets = delunay(lp1)
    else:
        triplets = params['triplets']

    if params['smooth1']:
        if 'hist1' not in params:
            params['hist1'] = []
        params['hist1'].append(lp1)
        if len(params['hist1']) > 2:
            params['hist1'].pop(0)
        h1 = np.array(params['hist1'])
        h1 = np.round(np.median(h1, axis=0)).astype(np.int32)
        lp1 = list(h1)

    #warped1 = image1.copy()
    #warped2 = image2.copy()
    warped1 = np.zeros_like(image1)
    warped2 = np.zeros_like(image2)

    two_ways = params['two_ways']
    for triplet in triplets:

        warp_triangle(
            image1,
            image2,
            warped1,
            warped2,
            triplet,
            lp1,
            lp2,
            two_ways=two_ways
        )

    out1 = blend(image1, warped1, lp1, clone_mode=params['clone'])
    if two_ways:
        out2 = blend(image2, warped2, lp2, clone_mode=params['clone'])
    else:
        out2 = None

    if debug:
        imshow("landmarks 1", draw_landmarks(image1, landmarks1, face1))
        imshow("landmarks 2", draw_landmarks(image2, landmarks2, face2))

        imshow("triangles 1", draw_triangles(image1, triplets, lp1))
        imshow("triangles 2", draw_triangles(image2, triplets, lp2))

        imshow("result 1", warped1)
        imshow("result 2", warped2)

        imshow("out 1", out1)
        imshow("out 2", out2)

        cv2.waitKey()

    return out1, out2


def run(args):

    face1_file = args.face1
    face2_file = args.face2

    #face1_file, face2_file = face2_file, face1_file

    print("loading face 1: {}".format(face1_file))
    face1 = cv2.imread(face1_file)
    if face1 is None:
        print("can't read image ", face1_file)
        return

    print("loading face 2: {}".format(face2_file))
    face2 = cv2.imread(face2_file)
    if face2 is None:
        print("can't read image ", face2_file)
        return

    print("loading models...")
    models = load_models(args)

    result1, result2 = swap_faces(models, face1, face2)

    if args.output1 is not None:
        print("writing output result for image1...")
        cv2.imwrite(args.output1, result1)

    if args.output2 is not None:
        print("writing output result for image2...")
        cv2.imwrite(args.output2, result2)

    if args.show:
        imshow("result 1", result1)
        imshow("result 2", result2)
        cv2.waitKey()

if __name__ == "__main__":
    args = parse_args()
    run(args)