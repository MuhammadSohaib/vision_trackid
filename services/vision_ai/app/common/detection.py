from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np



@dataclass
class ClassifierOutput:
    clazz: str
    clazz_id: int
    confidence: float
    purpose: str = ""
    unique_id: Optional[int] = None

    def to_dict(self):
        return {self.clazz: self.confidence}





@dataclass
class Detection:
    clazz: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float
    attributes: Optional[dict] = None
    features: Optional[List] = None
    object_id: Optional[int] = None
    loc_x: Optional[float] = None
    loc_y: Optional[float] = None
    second_stage: Optional[dict] = None

    def to_dict(self):
        # incorrect because of front end or back end current logic
        output = {
            "top": self.left,
            "left": self.top,
            "bottom": self.right,
            "right": self.bottom,
            "clazz": self.clazz,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }
        if self.object_id:
            output["object_id"] = str(self.object_id)
        if self.second_stage:
            output["second_stage"] = self.second_stage

        return output

    def annotate(self) -> Dict:
        """Creates a dictionary with the detection information.
        To be used for saving image annotation.

        Returns:
            Dict: Dictionary with the detection information.
        """
        output = {
            "bbox": {
                "top": self.top,
                "left": self.left,
                "bottom": self.bottom,
                "right": self.right,
            },
            "label": self.clazz,
            "confidence": self.confidence,
        }
        for stage in [self.second_stage, self.third_stage]:
            if not stage:
                continue
            output[stage.purpose] = {
                "label": stage.clazz,
                "confidence": stage.confidence,
            }
        if self.segmentation_mask.any():
            x_values = self.segmentation_mask[:, 0]
            y_values = self.segmentation_mask[:, 1]
            output["segmentation_mask"] = [
                {"x": x, "y": y} for x, y in zip(x_values, y_values)
            ]
        if self.face:
            output["face"] = {
                "label": self.face.face_class,
                "bbox": {
                    "top": self.face.top,
                    "left": self.face.left,
                    "bottom": self.face.bottom,
                    "right": self.face.right,
                },
            }
        if self.license_plate:
            output["license_plate"] = {
                "label": self.license_plate.id_,
                "confidence": self.license_plate.confidence,
                "bbox": {
                    "top": self.license_plate.top,
                    "left": self.license_plate.left,
                    "bottom": self.license_plate.bottom,
                    "right": self.license_plate.right,
                },
            }
        return output


@dataclass
class FrameDetections:
    frame_id: str
    camera_id: str
    detections: List[Detection]
    image_path: str
    analyzed_time: float
    frame_timestamp: float
    frame_width: int
    frame_height: int
    zones: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict:
        """Returns a simple dict representation of the object.
        This is used to dump the object to redis and is compatible with the
        further processing by services.

        Returns:
            Dict: Dictionary with the object information.
        """
        output = {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "detections": [
                detection.to_dict() for detection in self.detections
            ],
            "img_path": self.image_path,
            "analysed_time": self.analyzed_time,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_timestamp": self.frame_timestamp,
        }
        if self.zones:
            output["zones"] = self.zones

        return output

    def annotate(self, img_path: str, camera_name) -> Dict:
        """Returns an annotated version of the object.
        To be used for saving annotations of a frame.

        Args:
            img_path (str): Path to the image on local disk (not tmpfs)
            camera_name (str): Name of the camera of the frame

        Returns:
            Dict: dict that can be saved as json
        """
        output = {
            "image": {
                "file_path": img_path,
                "width": self.frame_width,
                "height": self.frame_height,
                "timestamp": self.frame_timestamp,
                "camera_name": camera_name,
            },
            "annotations": [
                detection.annotate() for detection in self.detections
            ],
        }

        return output
