import numpy as np
from common.set_logger import logger
import cv2
import math
from collections import defaultdict


class Occulusion:
    global_missing_ids = {}

    switch_ids = defaultdict(lambda: defaultdict(str))
    ids_distances = defaultdict(lambda: defaultdict(lambda: dict()))

    def __init__(
        self,
        curr_frame_info,
        prev_frame_info,
        camera_no,
        frame_info,
        frame_ids,
        core_detections,
        object_id_counter,
    ):
        self.curr_ids = set(curr_frame_info.keys())
        self.camera_info = camera_no
        self.threshold = 50
        self.curr_frame_info = curr_frame_info
        self.prev_frame_info = prev_frame_info
        self.frame_info = frame_info
        self.distance_threshold = 250
        self.frame_ids = frame_ids
        self.core_detections = core_detections
        self.object_id_counter = object_id_counter

    def fill_first(self):
        self.prev_frame_info[self.camera_info] = self.curr_frame_info.copy()
        logger.info("First Iteration Added")

    def newly_appeared(self):
        appeared_ids = self.curr_ids - set(
            self.prev_frame_info[self.camera_info].keys()
        )
        return appeared_ids

    def disappeard_ids(self):
        disappeared_ids = (
            set(self.prev_frame_info[self.camera_info].keys()) - self.curr_ids
        )
        return disappeared_ids

    def assign_latest_to_prev(self):
        self.prev_frame_info[self.camera_info] = self.curr_frame_info.copy()

    def find_point_location(self, object_id):
        x, y, t_p = object_id
        location = "center"
        if x < self.threshold or y < self.threshold or x > 1920 - self.threshold or y > 1080 - self.threshold:
            location = "edge"
        return location



    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def is_close(self, point1, point2):
        dist = self.distance(point1[0], point1[1], point2[0], point2[1])
        return dist <= self.distance_threshold

    def add_to_global(self, key, value, frame_no):
        Occulusion.global_missing_ids[key] = (value, frame_no)

    def delete_from_global(self, key):
        Occulusion.global_missing_ids.pop(key)

    
    def rearrange_data(self, input_data):
        result = defaultdict(str)

        for key, value in input_data.items():
            if key < value:
                result[value] = key
            else:
                result[key] = value

        return result

    def find_min_distance_key(self, camera, main_key):
        data = Occulusion.ids_distances[camera]
        if main_key not in data:
            return None

        nested_dict = data[main_key]
        # Find the key with the minimum distance
        min_key = min(nested_dict, key=nested_dict.get)
        return min_key


    def replace_first(self):
        _curr_ids = self.curr_ids.copy()
        for id in _curr_ids:
            if id in Occulusion.switch_ids[str(self.camera_info)]:
                switch_id = Occulusion.switch_ids[str(self.camera_info)][id]
                self.curr_ids.add(switch_id)
                self.curr_ids.discard(id)
                value = self.curr_frame_info[id]
                self.curr_frame_info[switch_id] = value
                self.curr_frame_info.pop(id)

    def ids_missed_in_last_n_frames(self, frames, current_frame):
        missed_ids = []
        for id in Occulusion.global_missing_ids:
            loc, f_no = Occulusion.global_missing_ids[id]
            if (current_frame - f_no) <= frames:
                location = self.find_point_location(loc)
                # logger.info(f"Found missed_id {id} location is: {location}")
                missed_ids.append(id)
        # logger.info(f"Found missed_ids ids are {missed_ids}")
        return missed_ids

    def compare_distance_with_all(self, missed_ids, object_id):
        x1, z1 = object_id
        distances = {}
        for id in missed_ids:
            if id not in Occulusion.global_missing_ids:
                continue
            loc, f_no = Occulusion.global_missing_ids[id]
            x2, y2, z2 = loc
            distances[id] = self.distance(x1, z1, x2, z2)
        if distances:
            match_id = min(distances, key=lambda x: distances[x])
            best_distance = distances[match_id]
        else:
            best_distance, match_id = 2000, None
        # logger.info(f" match id is {match_id}")
        # logger.info(f" missed_ids ids are {missed_ids}")
        return match_id, best_distance

    def swtich_ids_to_detections(self):
        rev_d = defaultdict(list)
        # print(Occulusion.switch_ids[str(self.camera_info)].items())
        # Occulusion.switch_ids[str(self.camera_info)] = self.remove_duplicates(Occulusion.switch_ids[str(self.camera_info)])
        for k, v in Occulusion.switch_ids[str(self.camera_info)].items():
            if k is None or v is None:
                del Occulusion.switch_ids[str(self.camera_info)][k]
            else:
                rev_d[int(v)].append(int(k))
        # create new dictionary preserving only smallest key for duplicate values
        switch_data = defaultdict(str)
        for v, keys in rev_d.items():
            switch_data[min(keys)] = v
        switch_data = self.rearrange_data(switch_data)
        for det in self.core_detections:
            if (
                det.object_id in switch_data
                and int(det.object_id) > int(switch_data[det.object_id])
                and switch_data[det.object_id] not in self.frame_ids
            ):
                # print("Replacing Core detection id", det.object_id, "with", switch_data[det.object_id])
                det.object_id = switch_data[det.object_id]
        return True

    def find(self):
        if self.camera_info not in self.prev_frame_info:
            self.fill_first()

        else:
            self.replace_first()
            appeared_ids = self.newly_appeared()
            disappeared_ids = self.disappeard_ids()
            # logger.info(f"Frame No {self.frame_info} \n")
            if len(disappeared_ids) > 0:
                for i in disappeared_ids:
                    self.add_to_global(
                        i,
                        self.prev_frame_info[self.camera_info][i],
                        self.frame_info,
                    )
                    location = self.find_point_location(
                        self.prev_frame_info[self.camera_info][i]
                    )
                    # logger.info(f'Object ID {i} disappeared from source ID {self.camera_info} at {location} in {self.frame_info}')
            if len(appeared_ids) > 0:
                for i in appeared_ids:
                    if i in Occulusion.global_missing_ids:
                        self.delete_from_global(i)
                    location = self.find_point_location(self.curr_frame_info[i])
                    x1, y1, z1 = self.curr_frame_info[i]
                    _global_missing_ids = Occulusion.global_missing_ids.copy()
                    missed_ids = self.ids_missed_in_last_n_frames(20, self.frame_info)
                    for id in _global_missing_ids:
                        loc, f_no = _global_missing_ids[id]
                        x2, y2, z2 = loc
                        # print(f"compairing {i} with {id}")
                        if (self.frame_info - f_no) < 50 and self.is_close(
                            (x1, z1), (x2, z2)
                        ):
                            # print("True forrrrr", i, id)

                            Occulusion.ids_distances[str(self.camera_info)][i][id] = self.distance(x1, z1, x2, z2)
                            min_distance_key = self.find_min_distance_key(str(self.camera_info), i)
                            Occulusion.switch_ids[str(self.camera_info)][
                                i
                            ] = min_distance_key

                            if id not in Occulusion.global_missing_ids:

                                continue
                            self.delete_from_global(id)
                            if id not in missed_ids:
                                continue
                            missed_ids.remove(id)
                        elif missed_ids:
                            # elif missed_ids and location == "center":
                            # logger.info(f"Newly Appeared ids are: {appeared_ids}")
                            """ logger.info(f"Missed ids are {missed_ids} and global are 
                            {Occulusion.global_missing_ids}")"""
                            match_id, dist = self.compare_distance_with_all(
                                missed_ids, (x1, z1)
                            )
                            if dist <= 200:
                                # logger.info(
                                #     f"Assigning {Occulusion.switch_ids[str(self.camera_info)][i]} with {match_id} "
                                # )
                                Occulusion.switch_ids[str(self.camera_info)][
                                    i
                                ] = match_id
                                self.delete_from_global(match_id)
                            # logger.info(f"We Found a match of {id} with {i} during {self.frame_info}")
                    """logger.info(f'Object ID {i} appeared in source ID {self.camera_info}
                        at {location} in {self.frame_info}')"""
            self.assign_latest_to_prev()
            self.swtich_ids_to_detections()
        return self.core_detections, self.prev_frame_info
