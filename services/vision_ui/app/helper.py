# from ultralytics import YOLO
import time
import streamlit as st
import cv2
import requests
# from pytube import YouTube

import settings
from walrus import Database
import itertools
import json
from PIL import Image
import pandas as pd
import os
import glob


db = Database(host="redis", port=6379, db=0)
detections_db = db.Stream("detections")




def read_redis(stream, last_id, block=0):
    """
    Reads data from redis.
    """
    return stream.read(last_id=last_id, block=block)


def style_alternate_rows(df):
    # Apply alternate row styling and style header
    style = df.style.apply(
        lambda x: ["background: #242424" if i % 2 == 0 else "background: #313131" for i in range(len(x))], axis=0
    ).set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#203239'), ('text-align', 'center')]},
         {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    return style



def get_image_paths(folder, exclude='crops'):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        # Exclude directories that contain the exclude keyword
        if exclude and not exclude in root:
            continue

        if "re_id" in dirs:
            root = f"{root}/{dirs[0]}"
            for file in glob.glob(f"{root}/*"):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(file)
    
    # Sort by creation time
    return image_paths


def play_stored_video():

    with open(settings.VIDEOS_DICT.get("video_1"), 'rb') as video_file1:
        video_bytes1 = video_file1.read()

    with open(settings.VIDEOS_DICT.get("video_2"), 'rb') as video_file2:
        video_bytes2 = video_file2.read()
    

    # with st.container():

    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.markdown("### Video 1 Original")
    #         if video_bytes1:
    #             st.video(video_bytes1)

    #     with col2:
    #         st.markdown("### Video 2 Original")
    #         if video_bytes2:
    #             st.video(video_bytes2)
    # st.markdown("--------------------------------------------------------------------------------------")
    with st.container():
        image = Image.open(settings.DEFAULT_IMAGE)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Video 1 Detections")
            st_frame1 = st.empty()
            st_frame1.image(image, caption='', channels="BGR", use_column_width=True)

        with col2:
            st.markdown("### Video 2 Detections")
            st_frame2 = st.empty()
            st_frame2.image(image, caption='', channels="BGR", use_column_width=True)

    
    st.markdown("--------------------------------------------------------------------------------------")
    st.markdown("### Detections Table")
    with st.container():
        table_placeholder = st.empty()
    
    st.sidebar.markdown("----------------------------")
    if st.sidebar.button('Display Stream'):
        st.session_state['show_stream'] = True
        st.session_state['show_reid'] = False
        
    if st.sidebar.button('Show REID'):
        st.session_state['show_reid'] = True
        st.session_state['show_stream'] = False


    # Display Stream
    if st.session_state['show_stream']:
        response = requests.post(f"http://vision_ai:5000/start?reid_conf={1 - st.session_state['reid_conf']}&single_reid={st.session_state['single_reid']}&multi_reid={st.session_state['multi_reid']}")
        last_id = "$"
        fetched_data = []
        for _ in itertools.count():
            try:
                if len(read_redis(detections_db, last_id, 1)):
                    msgid, messages = zip(
                        *[(msg[0], msg[1]) for msg in read_redis(detections_db, last_id, 0)]
                    )
                    data = []
                    last_id = sorted(msgid)[-1]
                    for msg in messages:
                        msg = json.loads(msg[b"batch"])
                        for frame_det in msg:
                            image = Image.open(frame_det["img_path"])
                            if "source_0" in frame_det["camera_id"]:
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
                            for detection in frame_det["detections"]:
                                if not "object_id" in detection:
                                    continue
                                json_data = {"camera_id": frame_det["camera_id"], "frame_no": frame_det["frame_no"],
                                             "detections": len(frame_det["detections"]),
                                             "top": detection["top"], "bottom": detection["bottom"], "left": detection["left"],
                                             "right": detection["right"], "object_id": detection["object_id"], "class": detection["clazz"],
                                             "confidence": detection["confidence"]
                                             }
                                fetched_data.append(json_data)
                                if len(fetched_data) > 10:
                                    fetched_data.pop(0)  # Remove the oldest entry

                                # Convert to DataFrame and display
                                df = pd.DataFrame(fetched_data)
                                styled_df = style_alternate_rows(df)

                                # Display styled DataFrame
                                table_placeholder.table(styled_df)
            except Exception as e:
                print(str(e))
                st.sidebar.error("Error: " + str(e))

        # Your code for displaying the stream
        # ...

    # Show REID
    if st.session_state['show_reid']:
        num_cols = 4

        # Get list of image filenames
        image_files = get_image_paths("frames")
        # Initialize the index for the current image
        current_image_idx = 0
        try:
            # Loop until all images are displayed
            st.markdown("### Source 1 REID")
            while current_image_idx < len(image_files):
                # Create a new row of columns
                cols = st.columns(num_cols)

                # Fill in the row with images
                for idx, col in enumerate(cols):
                    if current_image_idx >= len(image_files):
                        break  # No more images to display

                    # Load and display the image
                    if "stream_crops_0" in image_files[current_image_idx]:
                        image = Image.open(image_files[current_image_idx])
                        col.image(image, use_column_width=True)

                        # Move to the next image
                    current_image_idx += 1
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

        current_image_idx = 0
        try:
            # Loop until all images are displayed
            st.markdown("### Source 2 REID")
            while current_image_idx < len(image_files):
                # Create a new row of columns
                cols = st.columns(num_cols)

                # Fill in the row with images
                for idx, col in enumerate(cols):
                    if current_image_idx >= len(image_files):
                        break  # No more images to display

                    # Load and display the image
                    if "stream_crops_1" in image_files[current_image_idx]:
                        image = Image.open(image_files[current_image_idx])
                        col.image(image, use_column_width=True)

                        # Move to the next image
                    current_image_idx += 1
    
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))