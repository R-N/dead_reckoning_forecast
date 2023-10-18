import cv2
import numpy as np
import os
from ..util import remake_dir
from pathlib import Path

def get_frames(filename, n_frames=1, begin=0, end=None, size=(224, 224)):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = end or (v_len-1)
    if end >= v_len:
        begin, end = 0, (v_len-1)
    frame_list= np.linspace(begin, end, n_frames, dtype=np.int16)
    frame_list_1 = []
    for fn in range(v_len):
        success, frame = v_cap.read()
        if (fn in frame_list):
            pending = True
        if success is False:
            continue
        if pending:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
            # print(frame.shape)
            frames.append(frame)
            pending = False
            frame_list_1.append(fn)
    v_cap.release()
    try:
        assert len(frames) == n_frames, f"Invalid number of frames: {len(frames)}/{n_frames}/{v_len}, missed {[x for x in frame_list if x not in frame_list_1]} for {filename}"
    except AssertionError as ex:
        print(str(ex))
        return get_frames(filename, n_frames=n_frames, begin=begin, end=end-1, size=size)
    return frames, v_len

def store_frames(frames, frame_path):
    remake_dir(frame_path)
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        img_path = os.path.join(frame_path, f"frame_{ii}.jpg")
        cv2.imwrite(img_path, frame)

def extract_frames(file_path, frame_dir, frame_path=None, n_frames=16, begin=0, end=None, size=(224, 224), skip_exist=True):
    file_name = Path(file_path).stem
    if not frame_path:
        frame_path = os.path.join(frame_dir, file_name)
    if skip_exist and os.path.exists(os.path.join(frame_path, f"frame_{n_frames-1}.jpg")):
        #print(frame_path, "skip_exist")
        return
    frames, vlen = get_frames(file_path, n_frames=n_frames, begin=begin, end=end, size=size)
    print(frame_path, len(frames))
    store_frames(frames, frame_path)

def extract_all_frames(info, **kwargs):
    for file_info in info:
        file_id = file_info["video_name"]
        extract_frames(file_info["file_path"], frame_path=file_info["frame_path"], **kwargs)