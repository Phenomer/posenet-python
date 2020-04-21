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
parser.add_argument('--cam_width', type=int, default=424) # 848, 640
parser.add_argument('--cam_height', type=int, default=240) # 480, 480
parser.add_argument('--cam_rate', type=int, default=30)
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

def rs_main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        d435m  = d435.D435Manager(width=args.cam_width, height=args.cam_height, rate=args.cam_rate)
        clipping_distance = 2.5 / d435m.depth_scale # 2m
        client = udpclient.UDPClient(args.host, args.port)
        start  = time.time()
        frame_count = 0
        intr = d435m.depth_intrinsics
        dat = {}
        dstat = dictstat.DictListStat(600)

        while True:
            dimg  = d435m.filtered_frame()
            #dimg['depth'] = cv2.resize(dimg['depth'], (320, 240))
            #dimg['color'] = cv2.resize(dimg['color'], (320, 240))
            #dimg['depth'] = np.rot90(dimg['depth'], 3)
            #dimg['color'] = np.rot90(dimg['color'], 3)
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

            keypoint_score = keypoint_scores[0]
            keypoint_coord = keypoint_coords[0]

            for ki, (s, c) in enumerate(zip(keypoint_score, keypoint_coord[:, :])):
                x = int(c[0])
                y = int(c[1])
                try:
                    score = round(s, 3)
                    depth = int(dimg['depth'][x,y])
                    if score < 0.5:
                        continue
                    if depth == 0:
                        continue
                    point = pixel_to_point(intr, c[1], c[0], int(dimg['depth'][x,y]))
                    dat[posenet.PART_NAMES[ki]] = point
                except IndexError as e:
                    pass
            dstat.append(dat)
            stat = dstat.process(statistics.median)
            print(stat)
            client.send(stat)

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            #resized_overlay_image = cv2.resize(overlay_image,(args.cam_height*args.preview_scale, args.cam_width*args.preview_scale))
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    rs_main()
