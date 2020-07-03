import ast
import os
import cv2
import numpy as np

from face_swap import swap_faces, build_params, precompute_face, build_triplets
from common import shrink_if_large
from floating import floating_face


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

    elif effect_type == "floating":
        if event == 'set_background':
            effect['background'] = params


def build_effect_swap(models, ret, cfg):
    print("reading face for swap", cfg)

    to_precompute = []

    if os.path.isfile(cfg):

        name = 'the face face to swap'
        file = cfg
        to_precompute.append((name, file))

    else:

        config = ast.literal_eval(cfg)

        if isinstance(config, list):

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


def build_effect(models, args):
    ret = {}
    effect = args.effect
    spl = effect.split(":")
    effect_type = spl[0]
    if len(spl) > 1:
        cfg = spl[1]

    ret['effect_type'] = effect_type
    if effect_type == "swap":
        build_effect_swap(models, ret, cfg)
    if effect_type == "floating":
        ret['background'] = None

    return ret


def apply_effect_swap(models, frame, effect):
    image2 = effect['image2']
    if image2 is not None:
        params = effect['params']
        out1, out2 = swap_faces(models, frame,
                                image2=image2,
                                params=params
                                )
        return out1
    return frame


def apply_effect(models, frame, effect):

    effect_type = effect['effect_type']
    if effect_type == "swap":
        return apply_effect_swap(models, frame, effect)
    if effect_type == "floating":
        bg = effect['background']
        if bg is None:
            bg = np.zeros_like(frame)
        return floating_face(models, frame, bg)

    return frame
