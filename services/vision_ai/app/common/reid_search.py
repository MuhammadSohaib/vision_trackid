from common.set_logger import logger

import numpy as np
import os
import cv2


class NumpySearch:
    def __init__(self, metric: str = "cosine", top_n: int = 5):
        self.metric = metric
        self.top_n = top_n

    def _compute_dist_mat(
        self, query_embeddings: np.array, gallery_embeddings: np.array
    ) -> np.array:

        x_norm = np.linalg.norm(query_embeddings, axis=-1, keepdims=True)
        y_norm = np.linalg.norm(gallery_embeddings, axis=-1, keepdims=True)
        x = np.divide(query_embeddings, x_norm, where=x_norm!=0)
        y = np.divide(gallery_embeddings, y_norm, where=y_norm!=0)
        dist_mat = 1 - np.matmul(x, np.transpose(y))
        return dist_mat

    def search(
        self,
        q_cam_ids: np.array,
        q_p_ids: np.array,
        q_features: np.array,
        g_cam_ids: np.array,
        g_p_ids: np.array,
        g_features: np.array,
    ):
        dist_mat = self._compute_dist_mat(q_features, g_features)

        num_gallery = len(g_features)
        if self.top_n > num_gallery:
            top_n = num_gallery
        else:
            top_n = self.top_n

        top_camera_ids = []
        top_identity_ids = []
        top_distances = []
        for i, (q_cam_id, q_p_id) in enumerate(zip(q_cam_ids, q_p_ids)):
            keep = (g_p_ids != q_p_id) | (g_cam_ids != q_cam_id)
            distances = np.array([dist_mat[i, j] for j in range(num_gallery)])[keep]
            if top_n == num_gallery:
                top_ids = np.argsort(distances)
            else:
                idx = np.argpartition(distances, top_n)[:top_n]
                top_ids = idx[np.argsort(distances[idx])]
                distances = distances[idx]

            # for top_id in top_ids:
            top_camera_ids.append(g_cam_ids[keep][top_ids].tolist())
            top_identity_ids.append(g_p_ids[keep][top_ids].tolist())
            top_distances.append(distances.tolist())

        return top_camera_ids, top_identity_ids, top_distances, dist_mat


def search_ids(
    reid_features: None,
    q_id: str,
    f_camid: str,
    q_camid: str,
    q_features: list,
    folder_name=str,
    reid_conf=float,
):
    source_id, found_id = None, None
    reid_features = dict(reid_features)
    q_features = np.array(np.reshape(q_features, (1, 512)))
    q_camids = np.repeat(f_camid, len(q_features))
    q_pids = np.repeat(q_id, len(q_features))
    

    g_pids_list = []
    g_features_list = []
    for track_id, track in reid_features.items():
        for obj in track:
            g_pids_list.append(str(track_id))
            g_features_list.append(obj)

    g_pids = np.array(g_pids_list)
    g_camids = np.repeat(int(q_camid), len(g_pids))
    g_features = np.array(g_features_list)

    if q_features.shape[1] != g_features.shape[1]:
        raise ValueError("Feature dimensions of query and gallery don't match")

    search_model = NumpySearch()
    top_camera_ids, top_identity_ids, top_distances, dist_mat = search_model.search(
        q_camids, q_pids, q_features, g_camids, g_pids, g_features
    )
 
    min_dist = np.amin(dist_mat)
    if min_dist < reid_conf:
        min_index = np.where(dist_mat == min_dist)
        matched_g_idx = min_index[1][0]  # Get the index of the matched gallery feature
        matched_id = g_pids[matched_g_idx]
        logger.info(f"REID MATCH of CAM_{f_camid} {q_id} WITH CAM_{q_camid} {matched_id} HAVING distance: {min_dist}")

        o_path = f"{folder_name}/stream_crops_{f_camid}/obj_{q_id}.jpg"
        f_path = f"{folder_name}/stream_crops_{q_camid}/obj_{matched_id}.jpg"
        width = 128
        height = 256
        qimg = cv2.imread(o_path)
        logger.info(o_path)
        qimg = cv2.resize(qimg, (width, height))
        gimg = cv2.imread(f_path)
        gimg = cv2.resize(gimg, (width, height))
        vis = np.concatenate((qimg, gimg), axis=1)
        if not os.path.exists(f"{folder_name}/stream_crops_{f_camid}/re_id"):
            os.makedirs(f"{folder_name}/stream_crops_{f_camid}/re_id")
        save_file_path = (
            f"{folder_name}/stream_crops_{f_camid}/"
            f"re_id/matched_{q_id}_cam{q_camid}_{matched_id}.jpg"
        )
        cv2.imwrite(save_file_path, vis)
        logger.info(f"DONE SAVE IMAGE: {save_file_path}")

    else:
        matched_id = -1

    return q_id, matched_id
