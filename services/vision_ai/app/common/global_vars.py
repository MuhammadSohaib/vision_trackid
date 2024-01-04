from collections import defaultdict

perf_data = None
frame_count = {}
saved_count = {}

global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 2

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_CART = 1
PGIE_CLASS_ID_PERSON = 0
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["person"]


MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4
REID_CONF = None
SINGLE_REID = None
MULTI_REID = None
number_sources = None
folder_name = None
trajectory = {}
reid_features = defaultdict(lambda: defaultdict(list))
reid_switches = defaultdict(lambda: defaultdict(str))
drawn_ids = set()
last_frame = defaultdict(lambda: defaultdict(set))
prev_frame_info = defaultdict(lambda: dict)
object_id_counter = defaultdict(int)