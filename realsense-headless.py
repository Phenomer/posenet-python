import tensorflow as tf
import cv2
import time
import argparse

import posenet
import d435
import udpclient

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_width', type=int, default=848)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--host', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=3939)
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        d435m  = d435.D435Manager(width=args.cam_width, height=args.cam_height)
        client = udpclient.UDPClient(args.host, args.port)
        start  = time.time()
        frame_count = 0
        while True:
            dimg  = d435m.frame()
            input_image, display_image, output_scale = posenet.read_realsense_frame(dimg['color'], scale_factor=args.scale_factor, output_stride=output_stride)

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

            dat = {}
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print("Pose #%d, score %f" % (pi, pose_scores[pi]))
                dat[pi] = {}
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        try:
                            dat[pi][posenet.PART_NAMES[ki]] = {
                                "score": round(s, 3), "coord": [int(round(c[0])), int(round(c[1]))], "depth": int(dimg['depth'][int(c[0]),int(c[1])])}
                        except IndexError:
                            dat[pi][posenet.PART_NAMES[ki]] = {
                                "score": round(s, 3), "coord": [int(round(c[0])), int(round(c[1]))], "depth": 0}

            client.send(dat)

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
