import tensorflow as tf
import cv2
import time
import argparse
import threading as mt
import queue

import posenet
import d435
import udpclient

def pixcel_to_point(intr, x, y, d):
	nx = (x - intr.ppx) / intr.fx
	ny = (y - intr.ppy) / intr.fy
	r2 = nx * nx + ny * ny
	f  = 1 + intr.coeffs[0] * r2 + intr.coeffs[1] * r2 * r2 + intr.coeffs[4] * r2 * r2 * r2
	ux = nx * f + 2 * intr.coeffs[2] * nx * ny + intr.coeffs[3] * (r2 + 2 * nx * nx)
	uy = ny * f + 2 * intr.coeffs[3] * nx * ny + intr.coeffs[2] * (r2 + 2 * ny * ny)
	return [d * ux, d * uy, d]

def rs_main(jqueue):
    model        = 101
    cam_width    = 424
    cam_height   = 240
    scale_factor = 0.7125
    host         = "127.0.0.1"
    port         = 3939

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(model, sess)
        output_stride = model_cfg['output_stride']

        d435m  = d435.D435Manager(width=cam_width, height=cam_height)
        client = udpclient.UDPClient(host, port)
        start  = time.time()
        frame_count = 0
        intr = d435m.depth_intrinsics

        dat = {}
        while True:
            dimg  = d435m.frame()
            input_image, display_image, output_scale = posenet.read_realsense_frame(dimg['color'], scale_factor=scale_factor, output_stride=output_stride)

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
                if pose_scores[pi] == 0.:
                    break
                print("Pose #%d, score %f" % (pi, pose_scores[pi]))

                if not pi in dat:
                    dat[pi] = {}

                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        try:
                            score = round(s, 3)
                            #coord = [int(round(c[0])), int(round(c[1]))]
                            depth = int(dimg['depth'][int(c[0]),int(c[1])])
                            if depth == 0:
                                continue
                            point = pixcel_to_point(intr, c[1], c[0], int(dimg['depth'][int(c[0]),int(c[1])]))
                            dat[pi][posenet.PART_NAMES[ki]] = {
                                "score": score,
                                #"coord": coord,
                                #"depth": depth,
                                "point": point
                            }
                        except IndexError:
                            pass

            client.send(dat)
            jqueue.put(dat, True)
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


jqueue = queue.Queue(1)
ps     = mt.Thread(target=rs_main, args=(jqueue,))
ps.start()

from flask import Flask, redirect, url_for
app  = Flask(__name__)
@app.route('/')
def root():
    return redirect(url_for('static', filename="index.html"))

@app.route('/pose.json')
def pose_json():
    try:
        dat = jqueue.get()
        return dat
    except queue.Empty:
        return '{}'
