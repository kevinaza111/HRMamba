import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class VIPLHRLoader(BaseLoader):
    """The data loader for the VIPL dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an VIPL dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |p1|
                     |  |v1|
                     |      |source1|
                     |          |video.avi|
                     |          |gt_HR.csv|
                     |          |wave.csv|
                     |      |source2/
                     |          |video.avi|
                     |          |gt_HR.csv|
                     |          |wave.csv|
                     |p1|
                     |  |v2|
                     |      |source1|
                     |          |video.avi|
                     |          |gt_HR.csv|
                     |          |wave.csv|
                     |      |source2/
                     |          |video.avi|
                     |          |gt_HR.csv|
                     |          |wave.csv|
                     |...
                     |
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path, selected_vs=['v1'], selected_sources=['source2']):
        """Returns data directories under the path (For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "p*")
        if not data_dirs:
            raise ValueError("Data paths empty!")

        dirs = []

        for data_dir in data_dirs:
            for v_num in selected_vs:
                v_path = os.path.join(data_dir, v_num)

                if os.path.exists(v_path):
                    for source_num in selected_sources:
                        source_path = os.path.join(v_path, source_num)

                        if os.path.exists(source_path):
                            subject_trail_val = os.path.split(data_dir)[-1].replace('p', '')
                            index = int(subject_trail_val)
                            subject = int(subject_trail_val[0:2])
                            dirs.append({"index": index, "path": source_path, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        "invoked by preprocess_dataset for multi_process."
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        frames = self.read_video(os.path.join(data_dirs[i]['path']))

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "wave.csv"))

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        "Reads a video file, returns frames(T, H, W, 3) "
        video_files = [f for f in os.listdir(video_file) if f.endswith('.avi') or f.endswith('.mp4')]
        if not video_files:
            raise ValueError("No video files found in the provided folder.")

        VidObj = cv2.VideoCapture(os.path.join(video_file, video_files[0]))
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):

        bvp = []
        with open(bvp_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
