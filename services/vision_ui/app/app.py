# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="VISION TRACKID",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'show_stream' not in st.session_state:
    st.session_state['show_stream'] = False
if 'show_reid' not in st.session_state:
    st.session_state['show_reid'] = False

# Main page heading
st.title("VISION TRACKID")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("style/style.css")



# Sidebar
st.sidebar.header("Models Configs")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

st.session_state['reid_conf'] = confidence


single_reid = st.sidebar.toggle('Single Camera Reid')
multi_reid = st.sidebar.toggle('Multi Camera Reid')

st.session_state['single_reid'] = single_reid
st.session_state['multi_reid'] = multi_reid

# if single_reid:
#     st.write('Feature activated!')

st.sidebar.markdown("----------------------------")
st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)



helper.play_stored_video()

