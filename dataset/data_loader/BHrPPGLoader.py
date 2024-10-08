import glob
import glob
import json
import os
import re
import csv
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class BHrPPGLoader(BaseLoader):
    """The data loader for the BHrPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an BHrPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 0-0/
                     |      |-- 0-0/
                     |      |-- wave.csv
                     |   |-- 0-1/
                     |      |-- 0-1/
                     |      |-- wave.csv
                     |   |-- 0-2/
                     |      |-- 0-2/
                     |      |-- wave.csv
                     |...
                     |   |-- i-j/
                     |      |-- i-j/
                     |      |-- wave.csv
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path, light_condition='1'):
        """Returns data directories based on the specified light condition.

        Args:
            data_path (str): Path of the data directories.
            light_condition (str): Desired light condition - 'low', 'normal', 'high'.

        Returns:
            List of dictionaries containing index, path, and subject information.
        """



        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError("Data paths empty!")

        dirs = []
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            numeric_part = ''
            for char in subject_trail_val[::-1]:
                if char.isdigit():
                    numeric_part = char + numeric_part
                else:
                    break

            if numeric_part:  # Ensure we have a numeric part to convert
                index = int(subject_trail_val[0:2])
                subject = int(subject_trail_val[0:2])

                if light_condition == '0' and numeric_part.endswith('0'):
                    dirs.append({"index": index, "path": data_dir, "subject": subject})
                elif light_condition == '1' and numeric_part.endswith('1'):
                    dirs.append({"index": index, "path": data_dir, "subject": subject})
                elif light_condition == '2' and numeric_part.endswith('2'):
                    dirs.append({"index": index, "path": data_dir, "subject": subject})

        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'], filename, ""))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "wave.csv"))

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
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
