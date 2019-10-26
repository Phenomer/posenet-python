#!/usr/bin/python3

import tensorflow as tf
import cv2
import time
import argparse
import math
import random
import statistics
import numpy as np

import posenet
import d435
import udpclient
import dictstat

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_width', type=int, default=424)
parser.add_argument('--cam_height', type=int, default=240)
parser.add_argument('--preview_scale', type=int, default=2)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--host', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=3939)
args = parser.parse_args()

def angle(x1, y1, x2, y2):
    r = math.atan2(y2 - y1, x2 - x1)
    if r < 0:
        r = r + 2 * math.pi
    return math.floor(r * 360 / (2 * math.pi))

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

def face_orient_lr(dat):
    left_dist  = distance(dat['leftEar']['point'][0], dat['leftEar']['point'][1], dat['nose']['point'][0], dat['nose']['point'][1])
    right_dist = distance(dat['nose']['point'][0], dat['nose']['point'][1], dat['leftEar']['point'][0], dat['rightEar']['point'][1])
    return abs(left_dist) - abs(right_dist)

def face_orient_lr2(dat):
    s0 = dat['leftShoulder']['point'][0] - (dat['leftShoulder']['point'][0] - dat['rightShoulder']['point'][0]) / 2
    s1 = dat['leftShoulder']['point'][1] - (dat['leftShoulder']['point'][1] - dat['rightShoulder']['point'][1]) / 2
    return -(angle(dat["nose"]["point"][0], dat["nose"]["point"][1], s0, s1) - 90)

def face_z_angle(dat):
    s0 = dat['leftShoulder']['point'][0] - (dat['leftShoulder']['point'][0] - dat['rightShoulder']['point'][0]) / 2
    s2 = dat['leftShoulder']['point'][2] - (dat['leftShoulder']['point'][2] - dat['rightShoulder']['point'][2]) / 2
    return -(angle(dat["nose"]["point"][0], dat["nose"]["point"][2], s0, s2) - 90)

def body_angle(dat):
    h0 = dat['leftHip']['point'][0] - (dat['leftHip']['point'][0] - dat['rightHip']['point'][0]) / 2
    h1 = dat['leftHip']['point'][1] - (dat['leftHip']['point'][1] - dat['rightHip']['point'][1]) / 2
    s0 = dat['leftShoulder']['point'][0] - (dat['leftShoulder']['point'][0] - dat['rightShoulder']['point'][0]) / 2
    s1 = dat['leftShoulder']['point'][1] - (dat['leftShoulder']['point'][1] - dat['rightShoulder']['point'][1]) / 2
    return angle(s0, s1, h0, h1) - 90

def body_z_angle(dat):
    h0 = dat['leftHip']['point'][1] - (dat['leftHip']['point'][1] - dat['rightHip']['point'][1]) / 2
    h1 = dat['leftHip']['point'][2] - (dat['leftHip']['point'][2] - dat['rightHip']['point'][2]) / 2
    s0 = dat['leftShoulder']['point'][1] - (dat['leftShoulder']['point'][1] - dat['rightShoulder']['point'][1]) / 2
    s1 = dat['leftShoulder']['point'][2] - (dat['leftShoulder']['point'][2] - dat['rightShoulder']['point'][2]) / 2
    return angle(h0, h1, s0, s1) - 170

def shoulder_angle(dat):
    s0 = dat['leftShoulder']['point'][0] - (dat['leftShoulder']['point'][0] - dat['rightShoulder']['point'][0]) / 2
    s1 = dat['leftShoulder']['point'][1] - (dat['leftShoulder']['point'][1] - dat['rightShoulder']['point'][1]) / 2
    return angle(dat["nose"]["point"][0], dat["nose"]["point"][1], s0, s1) - 90

def leftarm_angle(dat):
    return -(angle(dat["leftElbow"]["point"][0], dat["leftElbow"]["point"][1], dat["leftShoulder"]["point"][0], dat["leftShoulder"]["point"][1]) -180)

def rightarm_angle(dat):
    return angle(dat["rightShoulder"]["point"][0], dat["rightShoulder"]["point"][1], dat["rightElbow"]["point"][0], dat["rightElbow"]["point"][1]) - 180

def leftarm_z_angle(dat):
    return -(angle(dat["leftElbow"]["point"][1], dat["leftElbow"]["point"][2], dat["leftShoulder"]["point"][1], dat["leftShoulder"]["point"][2]) - 180)

def rightarm_z_angle(dat):
    return -(angle(dat["rightElbow"]["point"][1], dat["rightElbow"]["point"][2], dat["rightShoulder"]["point"][1], dat["rightShoulder"]["point"][2]) - 180)

def pixel_to_point(intr, x, y, d):
	nx = (x - intr.ppx) / intr.fx
	ny = (y - intr.ppy) / intr.fy
	r2 = nx * nx + ny * ny
	f  = 1 + intr.coeffs[0] * r2 + intr.coeffs[1] * r2 * r2 + intr.coeffs[4] * r2 * r2 * r2
	ux = nx * f + 2 * intr.coeffs[2] * nx * ny + intr.coeffs[3] * (r2 + 2 * nx * nx)
	uy = ny * f + 2 * intr.coeffs[3] * nx * ny + intr.coeffs[2] * (r2 + 2 * ny * ny)
	return [d * ux, d * uy, d]

def clip_frame(depth_frame, color_frame, clipping_distance):
    grey_color = 153
    dimage_3d  = np.dstack((depth_frame,depth_frame,depth_frame))
    bg_removed = np.where((dimage_3d > clipping_distance) | (dimage_3d <= 0), grey_color, color_frame)
    return bg_removed

def data_sender(dat, dstat, socket):
    data = {
        'face_orient_lr':   face_orient_lr2(dat[0]),
        'face_z_angle':     face_z_angle(dat[0]),
        'body_z_angle':     body_z_angle(dat[0]),
        'body_angle':       body_angle(dat[0]),
        'shoulder_angle':   shoulder_angle(dat[0]),
        'leftarm_angle':    leftarm_angle(dat[0]),
        'rightarm_angle':   rightarm_angle(dat[0]),
        'leftarm_z_angle':  leftarm_z_angle(dat[0]),
        'rightarm_z_angle': rightarm_z_angle(dat[0])}
    dstat.append(data)
    stat = dstat.process(statistics.mean)
    stat['eye'] = random.randint(0, 99)
    stat['mouse'] = random.randint(0, 99)
    socket.send(stat)

def rs_main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        d435m  = d435.D435Manager(width=args.cam_width, height=args.cam_height)
        clipping_distance = 2 / d435m.depth_scale # 2m
        client = udpclient.UDPClient(args.host, args.port)
        start  = time.time()
        frame_count = 0
        intr = d435m.depth_intrinsics
        dat = {}
        dstat = dictstat.DictStat(10)

        while True:
            dimg  = d435m.frame()
            clipped_img = clip_frame(dimg['depth'], dimg['color'], clipping_distance)
            input_image, display_image, output_scale = posenet.read_realsense_frame(clipped_img, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            for pi in range(len(pose_scores)):
                if pose_scores[pi] <= 0.5:
                    break
                #print("Pose #%d, score %f" % (pi, pose_scores[pi]))
                if not pi in dat:
                    dat[pi] = {}

                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    try:
                        score = round(s, 3)
                        #coord = [int(round(c[0])), int(round(c[1]))]
                        depth = int(dimg['depth'][int(c[0]),int(c[1])])
                        if depth == 0:
                            continue
                        point = pixel_to_point(intr, c[1], c[0], int(dimg['depth'][int(c[0]),int(c[1])]))
                        dat[pi][posenet.PART_NAMES[ki]] = {
                            "score": score,
                            #"coord": coord,
                            #"depth": depth,
                            "point": point
                        }
                    except IndexError as e:
                        pass
            try:
                if not 0 in dat:
                    pass
                data = {
                    'face_orient_lr':   face_orient_lr2(dat[0]),
                    'face_z_angle':     face_z_angle(dat[0]),
                    'body_z_angle':     body_z_angle(dat[0]),
                    'body_angle':       body_angle(dat[0]),
                    'shoulder_angle':   shoulder_angle(dat[0]),
                    'leftarm_angle':    leftarm_angle(dat[0]),
                    'rightarm_angle':   rightarm_angle(dat[0]),
                    'leftarm_z_angle':  leftarm_z_angle(dat[0]),
                    'rightarm_z_angle': rightarm_z_angle(dat[0])}
                dstat.append(data)
                print("EYE:{eye: >4}, MOUSE:{mouse: >4}, FACE_LR:{face_orient_lr: >4}, FACE_Z:{face_z_angle: >4}, BODY:{body_angle: >4}, BODY_Z:{body_z_angle: >4}, SHO:{shoulder_angle: >4}, LARM:{leftarm_angle: >4}, RARM:{rightarm_angle: >4}, LARMZ:{leftarm_z_angle: >4}, RARMZ:{rightarm_z_angle: >4}".format(**stat), end="\r")
            except KeyError as e:
                print(e)
                pass
            finally:
                stat = dstat.process(statistics.mean)
                stat['eye'] = random.randint(0, 99)
                stat['mouse'] = random.randint(0, 99)
                client.send(stat)

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            resized_overlay_image = cv2.resize(overlay_image,(args.cam_width*args.preview_scale, args.cam_height*args.preview_scale))
            cv2.imshow('posenet', resized_overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    rs_main()
