#!/usr/bin/python3

import d435
import msgpack
import numpy as np
import imageio

def pixel_to_point(intr, x, y, d):
	nx = (x - intr.ppx) / intr.fx
	ny = (y - intr.ppy) / intr.fy
	r2 = nx * nx + ny * ny
	f  = 1 + intr.coeffs[0] * r2 + intr.coeffs[1] * r2 * r2 + intr.coeffs[4] * r2 * r2 * r2
	ux = nx * f + 2 * intr.coeffs[2] * nx * ny + intr.coeffs[3] * (r2 + 2 * nx * nx)
	uy = ny * f + 2 * intr.coeffs[3] * nx * ny + intr.coeffs[2] * (r2 + 2 * ny * ny)
	return [d * ux, d * uy, int(d)]

def clip_dump(intr, depth_frame, color_frame, clipping_distance, file):
    plist = []
    for idx, d in np.ndenumerate(depth_frame):
        if d < clipping_distance:
            continue
        vec3 = pixel_to_point(intr, idx[0], idx[1], d)
        color = color_frame[idx[0], idx[1]]
        file.write("{:.3f}\t{:.3f}\t{}\t{}\t{}\t{}\n".format(vec3[0], vec3[1], vec3[2], color[0], color[1], color[2]))


d435m  = d435.D435Manager(width=1280, height=720)
intr   = d435m.depth_intrinsics
clipping_distance = 2 / d435m.depth_scale # 2m
for n in range(0, 60):
    dimg   = d435m.frame()
with open('realsense.tsv', 'w') as f:
    clip_dump(intr, dimg['depth'], dimg['color'], clipping_distance, f)
imageio.imwrite("dump_color.png", dimg['color'], 'png')
imageio.imwrite("dump_depth.png", dimg['depth'], 'png')
