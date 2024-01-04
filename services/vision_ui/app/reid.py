#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import sys
import math
from common.is_aarch_64 import is_aarch64
from common.set_logger import logger
from common.occulusion import Occulusion
from common.detection import Detection
from common.utils import draw_predictions, save_crops_reid, perform_reid_matching
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import common.global_vars as gv
from collections import defaultdict

import numpy as np
import pyds
import cv2
import os
import random
from copy import copy
import ctypes
import shutil
import settings
import streamlit as st
from PIL import Image



def tiler_sink_pad_buffer_probe(pad, info, u_data):
    
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.info("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = defaultdict(int)
        current_frame_object_ids = dict()
        core_detections = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                object_id = obj_meta.object_id

                # Occlusion #########################################
                rect_params = obj_meta.rect_params
                left = int(rect_params.left)
                top = int(rect_params.top)
                right = int(rect_params.left + rect_params.width)
                bottom = int(rect_params.top + rect_params.height)
                x_p, b_p, t_p = left + int(rect_params.width / 2), bottom, top
                if object_id != "-1":
                    current_frame_object_ids[object_id] = (x_p, b_p, t_p)
                # Occlusion #########################################
                

                # logger.info(len(gv.last_frame[str(frame_meta.source_id)][obj_meta.obj_label]))
                if not str(object_id) in gv.last_frame[str(frame_meta.source_id)][obj_meta.obj_label]:

                    gv.last_frame[str(frame_meta.source_id)][obj_meta.obj_label].add(
                        str(object_id)
                    )

                if str(object_id) in gv.reid_switches[str(frame_meta.source_id)]:
                    object_id = gv.reid_switches[str(frame_meta.source_id)][str(object_id)]
            except StopIteration:
                break

            l_user = obj_meta.obj_user_meta_list
            features = None
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break
                
                # OSNET RE-ID
                if obj_meta.class_id == 0:
                    if (
                        user_meta.base_meta.meta_type
                        != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                    ):
                        continue

                    tensor_meta = pyds.NvDsInferTensorMeta.cast(
                        user_meta.user_meta_data
                    )
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(
                        pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float)
                    )
                    features = np.ctypeslib.as_array(ptr, shape=(512,))
                try:
                    l_user = l_user.next
                except StopIteration:
                    break
            
            
            detection = Detection(
                clazz=gv.pgie_classes_str[obj_meta.class_id],
                confidence=obj_meta.confidence,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                object_id=object_id,
                features=features
            )
            core_detections.append(detection)


            obj_counter[obj_meta.obj_label] += 1
            gv.object_id_counter[object_id] += 1
            
            if is_aarch64(): # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                        #  The original array cannot be accessed after this call.

            save_image = True

            

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        stream_index = "stream{0}".format(frame_meta.pad_index)
        gv.perf_data.update_fps(stream_index)
        # logger.info("camera:", str(frame_meta.pad_index), "frame_number is:",  frame_number, current_frame_object_ids)
        if len(current_frame_object_ids) > 0:
            frame_ids = [det.object_id for det in core_detections]
            occulusion = Occulusion(
                current_frame_object_ids,
                gv.prev_frame_info,
                frame_meta.pad_index,
                frame_meta.frame_num,
                frame_ids,
                core_detections,
                gv.object_id_counter,
            )
            core_detections, previous_frame_info = occulusion.find()
            gv.prev_frame_info[str(frame_meta.pad_index)] = previous_frame_info.copy()
        # logger.info("core_detections are:", [detection.object_id for detection in core_detections])
        # DRAW BOUNDING BOXES
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        original_frame = copy(n_frame)
        n_frame = draw_predictions(n_frame, core_detections)
        save_crops_reid(original_frame, core_detections, str(frame_meta.pad_index))
        frame_copy = np.array(n_frame, copy=True, order='C')
        # convert the array into cv2 default color format
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)



        if save_image:
            img_path = "{}/stream_{}/frame_{}.jpg".format(gv.folder_name, frame_meta.pad_index, frame_number)
            cv2.imwrite(img_path, frame_copy)
        gv.saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK





def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    logger.info(f"Decodebin child added: {name} \n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if not is_aarch64() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        Object.set_property("cudadec-memtype", 2)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    logger.info(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(videos=[], folder_name="frames"):

    gv.perf_data = PERF_DATA(len(videos))
    gv.number_sources = len(videos)

    gv.folder_name = folder_name
    for dir_name in os.listdir(gv.folder_name):
        
        if os.path.isdir(os.path.join(gv.folder_name, dir_name)):
            logger.info(f"Deleting {os.path.join(gv.folder_name, dir_name)}")
            shutil.rmtree(os.path.join(gv.folder_name, dir_name))
            

    # os.makedirs(gv.folder_name)
    logger.info(f"Frames will be saved in {gv.folder_name}")
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    logger.info("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    logger.info("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i, name in enumerate(videos):
        os.makedirs(gv.folder_name + "/stream_" + str(i))
        os.makedirs(f"{gv.folder_name}/stream_crops_{str(i)}")
        gv.frame_count["stream_" + str(i)] = 0
        gv.saved_count["stream_" + str(i)] = 0
        logger.info(f"Creating source_bin {i} \n")
        uri_name = name
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    logger.info("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    logger.info("Creating nvvidconv1 \n ")

    sgie_reid = Gst.ElementFactory.make("nvinfer", "reid-nvinference-engine")
    if not sgie_reid:
        sys.stderr.write(" Unable to make Re ID classifier \n")
    sgie_reid.set_property("config-file-path", "configs/reid.txt")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    logger.info("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    logger.info("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    logger.info("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    logger.info("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if is_aarch64():
        logger.info("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        logger.info("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        logger.info("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', gv.number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "configs/detector_configs.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != gv.number_sources):
        logger.warn(f"WARNING: Overriding infer-config batch-size {pgie_batch_size} with number of sources {gv.number_sources} \n")
        pgie.set_property("batch-size", gv.number_sources)
    tiler_rows = int(math.sqrt(gv.number_sources))
    tiler_columns = int(math.ceil((1.0 * gv.number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", gv.TILED_OUTPUT_WIDTH)
    tiler.set_property("height", gv.TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    config = configparser.ConfigParser()
    config.read("configs/tracker_configs.txt")
    config.sections()
    for key in config["tracker"]:
        if key == "tracker-width":
            tracker_width = config.getint("tracker", key)
            tracker.set_property("tracker-width", tracker_width)
        if key == "tracker-height":
            tracker_height = config.getint("tracker", key)
            tracker.set_property("tracker-height", tracker_height)
        # if key == "gpu-id":
        #     tracker_gpu_id = config.getint("tracker", key)
        #     tracker.set_property("gpu-id", configs["tracker_gpu"])
        if key == "ll-lib-file":
            tracker_ll_lib_file = config.get("tracker", key)
            tracker.set_property("ll-lib-file", tracker_ll_lib_file)
        if key == "ll-config-file":
            tracker_ll_config_file = config.get("tracker", key)
            tracker.set_property("ll-config-file", tracker_ll_config_file)
        if key == "enable-batch-process":
            tracker_enable_batch_process = config.getint("tracker", key)
            tracker.set_property(
                "enable_batch_process", tracker_enable_batch_process
            )
        if key == "enable-past-frame":
            tracker_enable_past_frames = config.getint("tracker", key)
            tracker.set_property("enable-past-frame", tracker_enable_past_frames)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    logger.info("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie_reid)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(sink)

    logger.info("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie_reid)
    sgie_reid.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to logger.info fps every 5 sec
        GLib.timeout_add(5000, gv.perf_data.perf_print_callback)

    # List the sources
    logger.info("Now playing...")
    for i, source in enumerate(videos):
        logger.info(f"{i}: {source}")

    logger.info("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    logger.info("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def get_image_paths(folder, exclude='crops'):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        # Exclude directories that contain the exclude keyword
        if exclude and exclude in root:
            continue

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                image_paths.append((os.path.getctime(full_path), full_path))
    
    # Sort by creation time
    image_paths.sort(key=lambda x: x[0])
    return [path for _, path in image_paths]

def stream_video():
    with open(settings.VIDEOS_DICT.get("video_1"), 'rb') as video_file1:
        video_bytes1 = video_file1.read()

    with open(settings.VIDEOS_DICT.get("video_2"), 'rb') as video_file2:
        video_bytes2 = video_file2.read()
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Video 1 Original")
            if video_bytes1:
                st.video(video_bytes1)

        with col2:
            st.markdown("##### Video 2 Original")
            if video_bytes2:
                st.video(video_bytes2)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### video 1 Detections")
            st_frame1 = st.empty()

        with col2:
            st.markdown("##### video 2 Detections")
            st_frame2 = st.empty()
    if 'task_running' not in st.session_state:
        st.session_state['task_running'] = False
    if 'task_completed' not in st.session_state:
        st.session_state['task_completed'] = False

    if st.sidebar.button('Detect Video Objects'):
        base_path = "file://"
        try:
            st.session_state['task_running'] = True
            st.session_state['task_completed'] = False

            if st.session_state['task_running'] and not st.session_state['task_completed']:
                st.info('Background task is running...')
                main(videos=[base_path + str(settings.VIDEOS_DICT.get("video_1")), base_path + str(settings.VIDEOS_DICT.get("video_2"))], folder_name="frames")

            if st.session_state['task_completed']:
                st.success('Background task completed!')
                st.session_state['task_running'] = False  # Reset the running state
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

    if st.sidebar.button('Show'):
        try:
            image_files = get_image_paths(gv.folder_name)
            logger.info(f"Files areeeeeeeeeeeeeeeeeeeeeeee: {image_files}")
            # Display images
            for img_file in image_files:

                image = Image.open(img_file)
                if "stream_0" in img_file:
                    st_frame1.image(image,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                    )
                else:
                    st_frame2.image(image,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                    )
    
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))