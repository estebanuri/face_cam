import argparse

import cProfile
import io
import os
import pstats
import cv2
import pyfakewebcam

from common import load_models, resize_w, shrink_if_large
from face_swap import swap_faces, build_params, precompute_face, delunay, build_triplets
import ast

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='input video source (device number or file name)'
    )

    parser.add_argument(
        '-is', '--input_size',
        required=False,
        help='desired size for input video (e. g. 640x480)',
    )

    parser.add_argument(
        '-sw', '--pre_scale_w',
        default=640,
        type=int,
        required=False,
        help='scales frame to fixed width before processing (scaling is proportional)',
    )

    parser.add_argument(
        '-f', '--fake_output',
        required=False,
        help='fake video output'
    )

    parser.add_argument(
        '-fs', '--fake_size',
        default="640x480",
        required=False,
        help='fake video output'
    )

    parser.add_argument(
        '-e', '--effect',
        default='swap:images/alberto_fernandez.webp',
        required=False,
        help='effect to apply'
    )

    parser.add_argument(
        '-s', '--show',
        default=True,
        help='shows output frame on screen'
    )

    parser.add_argument(
        '-m', '--landmarks_model',
        default="models/shape_predictor_68_face_landmarks.dat",
        help='dlib landmakrs shape predictor model path'
    )

    parser.add_argument(
        '-p', '--profile',
        default=False,
        help='profiles the code for performance tunning'
    )


    args = parser.parse_args()
    return args


def handle_effect_event(effect, event, params):

    effect_type = effect['effect_type']
    if effect_type == "swap":
        if event == 'select':
            index = params
            precomputed = effect['precomputed']

            if 0 <= index < len(precomputed):
                pc = precomputed[index]
                effect['selected'] = index
                effect['image2'] = pc['image']
                effect['params']['precomputed2'] = pc['data']

            else:
                effect['selected'] = -1
                effect['image2'] = None
                effect['params']['precomputed2'] = None



def build_effect(models, args):

    ret = {}
    effect = args.effect
    effect_type, cfg = effect.split(":")

    ret['effect_type'] = effect_type
    if effect_type == "swap":

        print("reading face for swap", cfg)

        to_precompute = []

        config = ast.literal_eval(cfg)
        if isinstance(config, str):

            if os.path.isfile(config):
                # it is a file: (by now we assume an image)
                name = 'the other face'
                file = cfg
                to_precompute.append((name, file))

        elif isinstance(config, list):

            to_precompute = config
            # for elem in config:
            #     name, file = elem
            #     to_precompute.append((name, file))

        else:
            raise Exception("effect config format not recognized", cfg)

        precomputed = []
        for tpc in to_precompute:
            name, image_file = tpc
            if not os.path.isfile(image_file):
                print("warning: image file {} not found.".format(image_file))
                continue
            image = cv2.imread(image_file)
            image = shrink_if_large(image, max=480)
            data = precompute_face(models, image)
            pc = {
                'name': name,
                'file': image_file,
                'image': image,
                'data': data
            }
            precomputed.append(pc)


        if len(precomputed) == 0:
            raise Exception("cannot build the effect: no faces were found")

        ret['precomputed'] = precomputed

        # construct the triplets from the first face
        pc = precomputed[0]['data']
        triplets = build_triplets(pc)

        params = build_params(triplets=triplets)
        ret['params'] = params

        handle_effect_event(ret, 'select', 0)


    return ret

def apply_effect(models, frame, effect):

    effect_type = effect['effect_type']
    if effect_type == "swap":

        image2 = effect['image2']
        if image2 is not None:
            params = effect['params']
            out1, out2 = swap_faces(models, frame,
                                    image2=image2,
                                    params=params
                                    )
            return out1

    return frame

def run(args):

    input = args.input
    if input.isdigit():
        input = int(input)

    cam = cv2.VideoCapture(input)

    if args.input_size is not None:
        desired_w, desired_h = args.input_size.split("x")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

    current_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video started", current_w, current_h)

    fake_output = None
    fake_h, fake_w = None, None
    if args.fake_output is not None:
        fake_w, fake_h = args.fake_size.split("x")
        fake_w, fake_h = int(fake_w), int(fake_h)
        fake_output = pyfakewebcam.FakeWebcam(args.fake_output, fake_w, fake_h)

    models = load_models(args)

    effect = build_effect(models, args)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)


    if args.profile:
        profile = cProfile.Profile()

    while True:

        ret, orig_frame = cam.read()

        # fps.update()
        # if fps.fps() < 15:
        #     # drop frames
        #     continue

        if args.pre_scale_w is None:
            # no pre-scaling requested
            frame = orig_frame
        elif args.pre_scale_w == current_w:
            # no pre-scaling needed
            frame = orig_frame
        else:
            frame = resize_w(orig_frame, args.pre_scale_w)
            # new_heigth = int(args.pre_scale_w * orig_frame.shape[0]/orig_frame.shape[1])
            # new_size = (args.pre_scale_w, new_heigth)
            # frame = cv2.resize(orig_frame, new_size)

        # if background is None:
        #     background = np.zeros(frame.shape, dtype='uint8')

        #ret_img = apply_face_effect2(models, frame)
        #ret_img = frame.copy()
        if args.profile:
            profile.enable()

        try:

            ret_img = apply_effect(models, frame, effect)

        except Exception as e:
            print(e)
            ret_img = frame

        if args.profile:
            profile.disable()

        cv2.imshow("frame", ret_img)

        out = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)

        if fake_output is not None:
            if out.shape != (fake_h, fake_w):
                out = cv2.resize(out, (fake_w, fake_h))
            fake_output.schedule_frame(out)

        # fake2.schedule_frame(flipped)
        # time.sleep(1/15.0)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        if ord('0') <= k <= ord('9'):
            index = (k - ord('0')) - 1
            handle_effect_event(effect, 'select', index)


        # if k == ord('b'):
        #     background = frame.copy()

    if args.profile:
        profile.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    # cam = cv2.VideoCapture("/home/esteban/src/prj/dl/face-mask-detector/examples/sample_video.mkv")
    run(args)

    # def apply_face_effect2(models, frame):
    #
    #     face_detector, shape_predictor = models
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     faces = face_detector(gray)
    #
    #     # ret = background.copy()
    #     ret = frame.copy()
    #
    #     # mask = np.zeros((frame.shape[1], frame.shape[0]), dtype='uint8')
    #     mask = np.zeros_like(frame)
    #
    #     for face in faces:
    #         x1 = face.left()
    #         y1 = face.top()
    #         x2 = face.right()
    #         y2 = face.bottom()
    #         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #
    #         landmarks = shape_predictor(gray, face)
    #         if landmarks.num_parts > 2:
    #             landmarks = landmarks_to_np(landmarks)
    #             #hull = cv2.convexHull(landmarks)
    #             ret = delunay(frame, landmarks)
    #             #cv2.drawContours(mask, [hull], 0, color=(255, 255, 255), thickness=-1)
    #
    #     # mask = cv2.GaussianBlur(mask, ksize=(15, 15), sigmaX=8, sigmaY=2)
    #     # mask = mask.astype(float) / 255
    #     # mask_inv = 1 - mask
    #     # ret = (frame * mask +
    #     #        background * mask_inv).astype(dtype='uint8')
    #
    #     return ret

    # def apply_face_effect(models, frame, background):
    #
    #     face_detector, shape_predictor = models
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     faces = face_detector(gray)
    #
    #     ret = background.copy()
    #
    #     mask = np.zeros((frame.shape[1], frame.shape[0]), dtype='uint8')
    #     #mask = np.zeros_like(frame)
    #     nose = None
    #     for face in faces:
    #         x1 = face.left()
    #         y1 = face.top()
    #         x2 = face.right()
    #         y2 = face.bottom()
    #         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #
    #         landmarks = shape_predictor(gray, face)
    #         if landmarks.num_parts > 2:
    #
    #             nose = landmarks.part(34)
    #
    #             landmarks = landmarks_to_np(landmarks)
    #             hull = cv2.convexHull(landmarks)
    #             cv2.drawContours(mask, [hull], 0, color=255, thickness=-1)
    #
    #
    #     if nose is not None:
    #         # outimage = cv2.seamlessClone(new_image.astype(numpy.uint8), base_image.astype(numpy.uint8), unitMask,
    #         #                              (masky, maskx), cv2.NORMAL_CLONE)
    #
    #         center = (nose.x, nose.y)
    #         #try:
    #         if 0 <= center[0] < frame.shape[1] and 0 <= center[1] < frame.shape[1]:
    #             try:
    #                 ret = cv2.seamlessClone(frame, background, mask, center, cv2.NORMAL_CLONE)
    #             except:
    #                 pass
    #
    #         #    pass
    #         # ret = cv2.seamlessClone(
    #         #     src=frame,
    #         #     dst=background,
    #         #     mask=mask,
    #         #     p = nose,
    #         #     blend=None,
    #         #     flags=cv2.NORMAL_CLONE)
    #     # mask = cv2.GaussianBlur(mask, ksize=(15, 15), sigmaX=8, sigmaY=2)
    #     # mask = mask.astype(float) / 255
    #     # mask_inv = 1 - mask
    #     # ret = (frame * mask +
    #     #         background * mask_inv).astype(dtype='uint8')
    #
    #     #ret = (alpha * frame * mask + beta * frame * mask_inv).astype(dtype='uint8')
    #     #ret = 255 * (frame * mask/255.0).astype(dtype='uint8')
    #     #ret = (frame * mask).astype(dtype='uint8')
    #     # mask_inv = 255 - mask
    #
    #     # ret = (frame * mask * alpha + frame * mask_inv * beta).astype(dtype='uint8')
    #
    #     # mask_inv = cv2.bitwise_not(mask)
    #     # face_img = cv2.bitwise_and(frame, frame, mask=mask)
    #     # ret = cv2.bitwise_and(ret, ret, mask=mask_inv)
    #     # ret = cv2.add(ret, face_img)
    #     # frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #     # frame = cv2.threshold(frame)
    #     # for n in range(0, landmarks.num_parts):
    #     #     x = landmarks.part(n).x
    #     #     y = landmarks.part(n).y
    #     #     cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    #
    #     return ret

    # def build_effect(args):
    #
    #     effect = args.effect
