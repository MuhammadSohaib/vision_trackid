import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import os
import numpy as np
from common.set_logger import logger
from common.reid_search import search_ids

from .detection import Detection, FrameDetections
import common.global_vars as gv

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def generate_color(object_id: str):
    color = [0, 0, 255]
    if object_id:
        if "cam" in str(object_id):
            object_id = object_id.split("_")[0]
        object_id = int(object_id)
        random.seed(object_id)
        color = [random.randint(0, 255) for _ in range(3)]

    return color


def draw_prediction(
    image: np.ndarray,
    prediction: Detection,
    color: Tuple[int, int, int] = None,
    line_thickness: int = 2,
):
    image_height, image_width, _ = image.shape
    color = color or generate_color(prediction.object_id)

    confidence = prediction.confidence
    left = int(prediction.left)
    top = int(prediction.top)
    right = int(prediction.right)
    bottom = int(prediction.bottom)
    if prediction.attributes:
        obj_name = f"{prediction.clazz} {prediction.attributes['gender']} {prediction.attributes['age']} {confidence:.2f}"
    else:
        obj_name = f"{prediction.clazz} {confidence:.2f}"
    if prediction.object_id:
        obj_name = f"{obj_name} {prediction.object_id}"
    

    tl = line_thickness or round(0.002 * (image_width + image_height) / 2) + 1
    c1 = (left, top)
    c2 = (right, bottom)

    '''Plot Normal Bounding box'''
    '''---------------------------------------'''

    # Draw the bounding box
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # Print the label
    tf = max(tl - 1, 1)
    t_size = cv2.getTextSize(obj_name, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        obj_name,
        (c1[0], c1[1] + t_size[1] + 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )   

    return image


def draw_predictions(
    image: np.ndarray,
    predictions: List[Detection],
    identifier: str = None,
):
    for prediction in predictions:
        image = draw_prediction(image, prediction)
    
    if identifier:
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (20, 30)
        scale = 0.75
        color = (0, 0, 255)
        cv2.putText(image, identifier, pos, font, scale, color, thickness=2)
    return image


def save_tmpfs_frames(
    frame_detections: List[FrameDetections],
    frames,
    l_frame_meta,
    cameras_info,
):
    for frame_detection, frame, frame_meta in zip(
        frame_detections, frames, l_frame_meta
    ):
        source_id = frame_meta.source_id

        camera_info = list(
            filter(lambda x: x.stream_id == source_id, cameras_info)
        )
        if not camera_info:
            logger.error(
                f"Camera info not found for {source_id=}, skipping for tmpfs saving"
            )
            continue
        camera_info = camera_info[0]

        output_path = frame_detection.image_path
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])




def save_crops(
    frame_detections: FrameDetections,
    frame: np.ndarray,
    output_path: str,
    frame_num: str,
):
    for idx, detection in enumerate(frame_detections.detections):
        crop = frame[
            detection.left : detection.right, detection.top : detection.bottom
        ]
        path = str(Path(output_path) / f"{frame_num}_{idx}.jpg")
        cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 10])




def save_crops_reid(image, core_detections, camera_id):
    output_dir = os.path.join(gv.folder_name, f"stream_crops_{camera_id}")
    os.makedirs(output_dir, exist_ok=True)
    modified_detections = []
    for detection in core_detections:
        if detection.clazz == 'person':
            # Crop and save the image
            crop = image[detection.top:detection.bottom, detection.left:detection.right]
            crop_filename = os.path.join(output_dir, f"obj_{str(detection.object_id)}.jpg")
            cv2.imwrite(crop_filename, crop)

            # Perform ReID matching if conditions are met
            if detection.features is not None and detection.features.size > 0:
                gv.reid_features[camera_id][str(detection.object_id)].append(detection.features.tolist())
            if str(detection.object_id) not in gv.reid_switches[camera_id]:
                if gv.MULTI_REID:
                    perform_reid_matching(detection, camera_id)
                modified_detections.append(detection.to_dict())
            else:
                continue
    return modified_detections


def perform_reid_matching(detection, camera_id):
    for i in range(gv.number_sources):
        if int(i) != int(camera_id) and gv.reid_features[str(i)] and detection.features is not None and detection.features.size > 0:
            source_id, found_id = search_ids(
                reid_features=gv.reid_features[str(i)],
                q_id=str(detection.object_id),
                f_camid=camera_id,
                q_camid=str(i),
                q_features=detection.features.tolist(),
                folder_name=gv.folder_name,
                reid_conf=gv.REID_CONF,
            )
            if found_id and found_id != -1:
                gv.reid_switches[camera_id][str(detection.object_id)] = str(found_id)
                gv.drawn_ids.add(found_id)
                if found_id not in gv.drawn_ids:
                    detection.object_id = str(found_id)

                print(f"Cam {camera_id} track_id: {str(detection.object_id)} predicted id: {found_id} in cam {str(i)}")
    return True


        


def draw_bounding_boxes(image, obj_meta, confidence, object_id, frame_meta, o_img):
    gv.trajectory
    random.seed(int(obj_meta.object_id))
    color = [random.randint(0, 255) for _ in range(2)]
    confidence = "{0:.0f}%".format(confidence * 100)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    right = left + int(rect_params.width)
    bottom = top + int(rect_params.height)
    bboxes = [left, top, (left+width), (top+height)]
                                                                 
    
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    c1, c2 = (int(left), int(top)), (int(width), int(height))

    image = cv2.circle(image, (left + int(width/2), top), 5, (255, 0, 0), 2)
    tl = (
        3 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # line/font thickness
    obj_name = str(confidence) + pgie_classes_str[obj_meta.class_id] + '=' + str(object_id)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(obj_name, 0, fontScale=tl / 4, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    image = cv2.rectangle(image, c1,c2, color, -1, cv2.LINE_AA)
    image = cv2.putText(image, obj_name,(c1[0], c1[1] - 2), 0, tl / 4,\
            [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)
    # if pgie_classes_str[obj_meta.class_id] == "person":
    id = int(obj_meta.object_id)
    # object trajectory
    center = ((int(bboxes[0]) + int(bboxes[2])) // 2,(int(bboxes[1]) + int(bboxes[3])) // 2)
    if id not in trajectory:
        trajectory[id] = []
    trajectory[id].append(center)
    for i1 in range(1,len(trajectory[id])):
        if trajectory[id][i1-1] is None or trajectory[id][i1] is None:
            continue
        # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
        thickness = 7
        # try:
            
        #     image = cv2.line(image, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
        # except:
        #     pass
    return image
