#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np
import sys
import os
if os.path.exists('C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64'):
    sys.path.append('C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64')
import pyrealsense2 as rs

class D435Manager():
    pipeline         = None
    profile          = None
    align            = None
    depth_scale      = None
    depth_intrinsics = None
    color_intrinsics = None

    decimate           = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial            = rs.spatial_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
    spatial.set_option(rs.option.filter_smooth_delta, 8)
    temporal           = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.5)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    hole_filling       = rs.hole_filling_filter()

    def __init__(self, width=640, height=480, serial='828112074708'):
        if D435Manager.pipeline:
            return
        D435Manager.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        D435Manager.profile = D435Manager.pipeline.start(config)
        align_to = rs.stream.color
        D435Manager.align = rs.align(align_to)
        depth_sensor = D435Manager.profile.get_device().first_depth_sensor()
        D435Manager.depth_scale = depth_sensor.get_depth_scale()
        D435Manager.depth_intrinsics = D435Manager.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        D435Manager.color_intrinsics = D435Manager.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        #print(D435Manager.depth_scale)
        #print(D435Manager.color_intrinsics)
        #print(D435Manager.depth_intrinsics)

    def frame(self):
        frames = D435Manager.pipeline.wait_for_frames()
        aligned_frames = D435Manager.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None
        return {
            'depth': np.asanyarray(aligned_depth_frame.get_data()),
            'color': np.asanyarray(color_frame.get_data())
        }

    def filtered_frame(self):
        frames = D435Manager.pipeline.wait_for_frames()
        aligned_frames = D435Manager.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None
        filtered_depth_frame = self.filter(aligned_depth_frame)
        return {
            'depth': np.asanyarray(filtered_depth_frame.get_data()),
            'color': np.asanyarray(color_frame.get_data())
        }

    def filter(self, depth_frame):
        depth_frame = D435Manager.decimate.process(depth_frame)
        depth_frame = D435Manager.depth_to_disparity.process(depth_frame)
        depth_frame = D435Manager.spatial.process(depth_frame)
        depth_frame = D435Manager.temporal.process(depth_frame)
        depth_frame = D435Manager.disparity_to_depth.process(depth_frame)
        depth_frame = D435Manager.hole_filling.process(depth_frame)
        return depth_frame

    def stop(self):
        D435Manager.pipeline.stop