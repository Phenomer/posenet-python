import tensorflow as tf
import cv2
import time
import argparse

import posenet
import d435

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_width', type=int, default=848)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        d435m = d435.D435Manager(width=args.cam_width, height=args.cam_height)

        start = time.time()
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

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print("Pose #%d, score %f" % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        try:
                            print('Keypoint %s, score = %f, coord = %s depth = %d' % (posenet.PART_NAMES[ki], s, c, dimg['depth'][int(c[0]),int(c[1])]))
                        except IndexError:
                            print('Keypoint %s, score = %f, coord = %s depth = None' % (posenet.PART_NAMES[ki], s, c))

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()