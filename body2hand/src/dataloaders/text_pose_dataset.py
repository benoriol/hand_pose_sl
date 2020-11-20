import torch
from torch.utils.data import Dataset

from glob import glob
import json
import time
from tokenizers import Tokenizer
import random

import h5py

import numpy

BODY_HEAD_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]

def format_keypoints(keypoints, n_dim=2):
    '''
    Changes from [x1, y1, c1, x2, y2, c2, ...] to [[x1, y1, c1], [x2, y2, c2], ...]
    being x, y the 2D coords and c the openpose confidence
    :param keypoints: keypoints with input format
    :param n_dim: dimnensionality of the coordenates
    :return: keypoints with good format
    '''
    n_dim += 1
    return [keypoints[n_dim * i:n_dim * i + n_dim] for i in
                         range(len(keypoints) // n_dim)]


def load_keypoints(input_json):

    if isinstance(input_json, str):
        data = json.load(open(input_json))
    elif isinstance(input_json, dict):
        data = input_json
    else:
        raise Exception("Input type not supported")

    body_kp = format_keypoints(data["people"][0]["pose_keypoints_2d"])
    #Filer out legs
    body_kp = [body_kp[i] for i in BODY_HEAD_KEYPOINTS]

    l_hand_kp = format_keypoints(data["people"][0]["hand_left_keypoints_2d"])
    r_hand_kp = format_keypoints(data["people"][0]["hand_right_keypoints_2d"])

    # Separate confidence and keypoints
    r_hand_kp, r_hand_conf = [x[:2] for x in r_hand_kp], [x[2] for x in r_hand_kp]
    l_hand_kp, l_hand_conf = [x[:2] for x in l_hand_kp], [x[2] for x in l_hand_kp]
    body_kp, body_conf = [x[:2] for x in body_kp], [x[2] for x in body_kp]

    return r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf

def select_jsons(all_jsons, n=100, selection_type=None):
    '''
    :param all_jsons: list with al json files
    :param n: number to be selected
    :param selection_type: algotithm to choose jsons files
    :return: list of jsons, position of first json
    '''

    if len(all_jsons) <= n:
        return all_jsons, 0

    elif selection_type == "first":
        return all_jsons[:n], 0

    elif selection_type == "randomcrop":
        start_n = random.randint(0, len(all_jsons) - n)
        return all_jsons[start_n:start_n+n], start_n



class PoseDataset(Dataset):
    def __init__(self, metadata_file, max_frames, transform):
        super().__init__()
        if isinstance(metadata_file, str):
            self.data = json.load(open(metadata_file))
        elif isinstance(metadata_file, list):
            self.data = metadata_file
        self.max_frames = max_frames

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #print("Start get item")

        # DONE: Get right and left hand
        # TODO: Random crop of sequence instead of clipping
        # TODO: Only load the ones that are going to be used

        metadata = self.data[idx]

        jsons_folder = metadata["frame_jsons_folder"]
        all_jsons = glob(jsons_folder + "/*")
        assert len(all_jsons) > 0
        all_jsons.sort()
        all_jsons = self.select_frames(all_jsons)

        item = self.load_jsons(all_jsons)

        item["text"] = metadata["text"]
        item["n_frames"] = min(metadata["n_frames"], self.max_frames),

        # Pad sequence to max len
        item = self.pad(item)

        # Clip sequence to max len
        # TODO This should not be needed since it is clipped in the all_jsons list
        item = self.clip(item)

        # To tensor
        item = self.to_tensor(item)

        if self.transform:
            item = self.transform(item)

        return item

    def load_jsons(self, all_jsons):
        item = {
            "body_kp": [],
            "body_conf": [],
            "right_hand_kp": [],
            "right_hand_conf": [],
            "left_hand_kp": [],
            "left_hand_conf": [],
            "json_paths": [],
        }

        for json_file in all_jsons:
            r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf = load_keypoints(
                json_file)
            item["body_kp"].append(body_kp)
            item["right_hand_kp"].append(r_hand_kp)
            item["left_hand_kp"].append(l_hand_kp)
            item["body_conf"].append(body_conf)
            item["right_hand_conf"].append(r_hand_conf)
            item["left_hand_conf"].append(l_hand_conf)
            item["json_paths"].append(json_file)
        return item

    def pad(self, item):
        while len(item["body_kp"]) < self.max_frames:
            item["body_kp"].append(item["body_kp"][0])
            item["right_hand_kp"].append(item["right_hand_kp"][0])
            item["left_hand_kp"].append(item["left_hand_kp"][0])
            item["body_conf"].append(item["body_conf"][0])
            item["right_hand_conf"].append(item["right_hand_conf"][0])
            item["left_hand_conf"].append(item["left_hand_conf"][0])
        return item

    def clip(self, item):
        item["body_kp"] = item["body_kp"][:self.max_frames]
        item["right_hand_kp"] = item["right_hand_kp"][:self.max_frames]
        item["left_hand_kp"] = item["left_hand_kp"][:self.max_frames]
        item["body_conf"] = item["body_conf"][:self.max_frames]
        item["right_hand_conf"] = item["right_hand_conf"][:self.max_frames]
        item["left_hand_conf"] = item["left_hand_conf"][:self.max_frames]
        item["json_paths"] = item["json_paths"][:self.max_frames]

        return item

    def select_frames(self, all_jsons):
        all_jsons = all_jsons[:self.max_frames]
        return all_jsons

    def to_tensor(self, item):
        item["body_kp"] = torch.tensor(item["body_kp"]).float()
        item["right_hand_kp"] = torch.tensor(item["right_hand_kp"]).float()
        item["left_hand_kp"] = torch.tensor(item["left_hand_kp"]).float()
        item["body_conf"] = torch.tensor(item["body_conf"]).float()
        item["right_hand_conf"] = torch.tensor(item["right_hand_conf"]).float()
        item["left_hand_conf"] = torch.tensor(item["left_hand_conf"]).float()

        return item

class TextPoseDataset(Dataset):
    def __init__(self, metadata_file, max_frames, transform):
        super().__init__()
        if isinstance(metadata_file, str):
            self.data = json.load(open(metadata_file))
        elif isinstance(metadata_file, list):
            self.data = metadata_file
        self.max_frames = max_frames

        self.transform = transform

        self.tokenizer = Tokenizer.from_file("tokenizer_models/tokenizer.json")

        self.n_tokens = 0
        self.n_utt = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #print("Start get item")

        # DONE: Get right and left hand
        # TODO: Random crop of sequence instead of clipping
        # TODO: Only load the ones that are going to be used

        metadata = self.data[idx]

        jsons_folder = metadata["frame_jsons_folder"]
        all_jsons = glob(jsons_folder + "/*")
        assert len(all_jsons) > 0
        all_jsons.sort()
        all_jsons, n_init = self.select_frames(all_jsons)

        item = self.load_jsons(all_jsons)

        item["text"] = metadata["text"]
        item["n_frames"] = min(metadata["n_frames"], self.max_frames)

        tokens = self.tokenizer.encode(item["text"])
        self.n_tokens += len(tokens)
        self.n_utt += 1

        # Pad sequence to max len
        item = self.pad(item)

        # Clip sequence to max len
        # TODO This should not be needed since it is clipped in the all_jsons list
        item = self.clip(item)

        # To tensor
        item = self.to_tensor(item)

        if self.transform:
            item = self.transform(item)

        return item

    def load_jsons(self, all_jsons):
        item = {
            "body_kp": [],
            "body_conf": [],
            "right_hand_kp": [],
            "right_hand_conf": [],
            "left_hand_kp": [],
            "left_hand_conf": [],
            "json_paths": [],
        }

        for json_file in all_jsons:
            r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf = load_keypoints(
                json_file)
            item["body_kp"].append(body_kp)
            item["right_hand_kp"].append(r_hand_kp)
            item["left_hand_kp"].append(l_hand_kp)
            item["body_conf"].append(body_conf)
            item["right_hand_conf"].append(r_hand_conf)
            item["left_hand_conf"].append(l_hand_conf)
            item["json_paths"].append(json_file)
        return item

    def pad(self, item):
        while len(item["body_kp"]) < self.max_frames:
            item["body_kp"].append(item["body_kp"][0])
            item["right_hand_kp"].append(item["right_hand_kp"][0])
            item["left_hand_kp"].append(item["left_hand_kp"][0])
            item["body_conf"].append(item["body_conf"][0])
            item["right_hand_conf"].append(item["right_hand_conf"][0])
            item["left_hand_conf"].append(item["left_hand_conf"][0])
        return item

    def clip(self, item):
        item["body_kp"] = item["body_kp"][:self.max_frames]
        item["right_hand_kp"] = item["right_hand_kp"][:self.max_frames]
        item["left_hand_kp"] = item["left_hand_kp"][:self.max_frames]
        item["body_conf"] = item["body_conf"][:self.max_frames]
        item["right_hand_conf"] = item["right_hand_conf"][:self.max_frames]
        item["left_hand_conf"] = item["left_hand_conf"][:self.max_frames]
        item["json_paths"] = item["json_paths"][:self.max_frames]

        return item

    def select_frames(self, all_jsons):
        all_jsons = all_jsons[:self.max_frames]
        return all_jsons

    def to_tensor(self, item):
        item["body_kp"] = torch.tensor(item["body_kp"]).float()
        item["right_hand_kp"] = torch.tensor(item["right_hand_kp"]).float()
        item["left_hand_kp"] = torch.tensor(item["left_hand_kp"]).float()
        item["body_conf"] = torch.tensor(item["body_conf"]).float()
        item["right_hand_conf"] = torch.tensor(item["right_hand_conf"]).float()
        item["left_hand_conf"] = torch.tensor(item["left_hand_conf"]).float()

        return item

class FastPoseDataset(Dataset):
    def __init__(self, data, max_frames, transform):
        super().__init__()
        if isinstance(data, str):
            self.data = json.load(open(data))
        elif isinstance(data, list):
            self.data = data

        self.max_frames = max_frames

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #print("Start get item")

        # DONE: Get right and left hand
        # TODO: Random crop of sequence instead of clipping
        # TODO: Only load the ones that are going to be used

        metadata = self.data[idx]

        json_data = metadata["frame_jsons"]

        json_data, start_frame = select_jsons(json_data, self.max_frames)

        item = self.load_jsons(json_data)

        item["text"] = metadata["text"]
        item["n_frames"] = min(metadata["n_frames"], self.max_frames)

        # Pad sequence to max len
        item = self.pad(item)

        # Clip sequence to max len
        # TODO This should not be needed since it is clipped in the all_jsons list
        item = self.clip(item)

        # To tensor
        item = self.to_tensor(item)

        if self.transform:
            item = self.transform(item)

        return item

    def load_jsons(self, all_jsons):
        item = {
            "body_kp": [],
            "body_conf": [],
            "right_hand_kp": [],
            "right_hand_conf": [],
            "left_hand_kp": [],
            "left_hand_conf": [],
            "json_paths": [],
        }

        for json_file in all_jsons:
            r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf, json_path = load_keypoints(
                json_file)
            item["body_kp"].append(body_kp)
            item["right_hand_kp"].append(r_hand_kp)
            item["left_hand_kp"].append(l_hand_kp)
            item["body_conf"].append(body_conf)
            item["right_hand_conf"].append(r_hand_conf)
            item["left_hand_conf"].append(l_hand_conf)
            item["json_paths"].append(json_file)
        return item

    def pad(self, item):
        while len(item["body_kp"]) < self.max_frames:
            item["body_kp"].append(item["body_kp"][0])
            item["right_hand_kp"].append(item["right_hand_kp"][0])
            item["left_hand_kp"].append(item["left_hand_kp"][0])
            item["body_conf"].append(item["body_conf"][0])
            item["right_hand_conf"].append(item["right_hand_conf"][0])
            item["left_hand_conf"].append(item["left_hand_conf"][0])
        return item

    def clip(self, item):
        item["body_kp"] = item["body_kp"][:self.max_frames]
        item["right_hand_kp"] = item["right_hand_kp"][:self.max_frames]
        item["left_hand_kp"] = item["left_hand_kp"][:self.max_frames]
        item["body_conf"] = item["body_conf"][:self.max_frames]
        item["right_hand_conf"] = item["right_hand_conf"][:self.max_frames]
        item["left_hand_conf"] = item["left_hand_conf"][:self.max_frames]
        item["json_paths"] = item["json_paths"][:self.max_frames]

        return item

    def select_frames(self, all_jsons):
        all_jsons = all_jsons[:self.max_frames]
        return all_jsons

    def to_tensor(self, item):
        item["body_kp"] = torch.tensor(item["body_kp"]).float()
        item["right_hand_kp"] = torch.tensor(item["right_hand_kp"]).float()
        item["left_hand_kp"] = torch.tensor(item["left_hand_kp"]).float()
        item["body_conf"] = torch.tensor(item["body_conf"]).float()
        item["right_hand_conf"] = torch.tensor(item["right_hand_conf"]).float()
        item["left_hand_conf"] = torch.tensor(item["left_hand_conf"]).float()

        return item

class FastTextPoseDataset(Dataset):
    def __init__(self, metadata_file, max_frames, transform, selection,
                 use_rand_tokens=False):
        super().__init__()
        if isinstance(metadata_file, str):
            self.data = json.load(open(metadata_file))
        elif isinstance(metadata_file, list):
            self.data = metadata_file
        self.max_frames = max_frames

        self.transform = transform

        self.tokenizer = Tokenizer.from_file("tokenizer_models/tokenizer.json")

        self.n_tokens = 0
        self.n_utt = 0

        self.random_tokens = torch.randint(0, 999, (40,), dtype=torch.long)

        self.use_rand_tokens = use_rand_tokens

        self.selection = selection

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #print("Start get item")

        # DONE: Get right and left hand
        # TODO: Random crop of sequence instead of clipping to first N

        metadata = self.data[idx]

        json_data = metadata["frame_jsons"]

        json_data, start_frame = select_jsons(json_data, self.max_frames, selection_type=self.selection)

        item = self.load_jsons(json_data)


        item["text"] = metadata["text"]
        item["n_frames"] = min(metadata["n_frames"], self.max_frames)

        #
        # self.n_tokens += len(tokens)
        # self.n_utt += 1

        # Pad sequence to max len
        item = self.pad(item)

        # Clip sequence to max len
        # TODO This should not be needed since it is clipped in the all_jsons list
        item = self.clip(item)

        # To tensor
        item = self.to_tensor(item)

        if self.use_rand_tokens:
            item["text_tokens"] = self.random_tokens

        else:
            tokens = self.tokenizer.encode(item["text"]).ids
            tokens = tokens[:40]
            tokens = tokens + [0] * (40 - len(tokens))
            tokens = torch.tensor(tokens, dtype=torch.long)
            item["text_tokens"] = tokens

        if self.transform:
            item = self.transform(item)

        return item

    def load_jsons(self, all_jsons):
        item = {
            "body_kp": [],
            "body_conf": [],
            "right_hand_kp": [],
            "right_hand_conf": [],
            "left_hand_kp": [],
            "left_hand_conf": [],
            "json_paths": [],
        }

        for json_file in all_jsons:
            if "json_data" in json_file.keys():
                r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf = load_keypoints(
                    json_file["json_data"])
            else:
                r_hand_kp, r_hand_conf, l_hand_kp, l_hand_conf, body_kp, body_conf = load_keypoints(
                    json_file)
            item["body_kp"].append(body_kp)
            item["right_hand_kp"].append(r_hand_kp)
            item["left_hand_kp"].append(l_hand_kp)
            item["body_conf"].append(body_conf)
            item["right_hand_conf"].append(r_hand_conf)
            item["left_hand_conf"].append(l_hand_conf)
            if isinstance(json_file, str):
                item["json_paths"].append(json_file)
            # TODO ADAPT this to new json format
            elif "json_path" in json_file.keys():
                item["json_paths"].append(json_file["json_path"])
            else:
                item["json_paths"].append(None)
        return item

    def pad(self, item):
        while len(item["body_kp"]) < self.max_frames:
            item["body_kp"].append(item["body_kp"][0])
            item["right_hand_kp"].append(item["right_hand_kp"][0])
            item["left_hand_kp"].append(item["left_hand_kp"][0])
            item["body_conf"].append(item["body_conf"][0])
            item["right_hand_conf"].append(item["right_hand_conf"][0])
            item["left_hand_conf"].append(item["left_hand_conf"][0])
        return item

    def clip(self, item):
        item["body_kp"] = item["body_kp"][:self.max_frames]
        item["right_hand_kp"] = item["right_hand_kp"][:self.max_frames]
        item["left_hand_kp"] = item["left_hand_kp"][:self.max_frames]
        item["body_conf"] = item["body_conf"][:self.max_frames]
        item["right_hand_conf"] = item["right_hand_conf"][:self.max_frames]
        item["left_hand_conf"] = item["left_hand_conf"][:self.max_frames]
        item["json_paths"] = item["json_paths"][:self.max_frames]

        return item

    def select_frames(self, all_jsons):
        all_jsons = all_jsons[:self.max_frames]
        return all_jsons

    def to_tensor(self, item):
        item["body_kp"] = torch.tensor(item["body_kp"]).float()
        item["right_hand_kp"] = torch.tensor(item["right_hand_kp"]).float()
        item["left_hand_kp"] = torch.tensor(item["left_hand_kp"]).float()
        item["body_conf"] = torch.tensor(item["body_conf"]).float()
        item["right_hand_conf"] = torch.tensor(item["right_hand_conf"]).float()
        item["left_hand_conf"] = torch.tensor(item["left_hand_conf"]).float()

        return item


class TextPoseH5Dataset(Dataset):
    def __init__(self, keypoints_file, text_file, max_frames, transform, selection,
                 use_rand_tokens=False):

        self.keypoints_data = h5py.File(keypoints_file, "r")

        self.utt_texts = {}
        text = open(text_file)
        for line in text:
            line = line.strip().split(" ")
            utt_id = line[0]
            text = line[1:]
            text = " ".join(text)
            self.utt_texts[utt_id] = text

        text_utt_ids = set(self.utt_texts.keys())
        keypoints_utt_ids = list(self.keypoints_data.keys())
        self.utt_ids = list(text_utt_ids.intersection(keypoints_utt_ids))

        print("IDs in text file:\t", len(text_utt_ids))
        print("IDs in keypoints file:\t", len(keypoints_utt_ids))
        print("IDs in both files:\t", len(self.utt_ids))

        self.max_frames = max_frames

        self.transform = transform

        self.tokenizer = Tokenizer.from_file("tokenizer_models/tokenizer.json")

        self.random_tokens = torch.randint(0, 999, (40,), dtype=torch.long)

        self.use_rand_tokens = use_rand_tokens

        self.selection = selection



    def __len__(self):
        return len(self.utt_ids)

    def array2item(self, array):
        N_frames = array.shape[0]
        array = array.reshape((N_frames, 3, -1))
        array = array.transpose(0, 2, 1)
        array_kp = array[:, :, :2]
        array_conf = array[:, :, 2]

        body_keypoints_2d = array_kp[:, :8, :]
        body_conf = array_conf[:, :8]

        left_hand_keypoints = array_kp[:, 8:29, :]
        left_hand_conf = array_conf[:, 8:29]

        right_hand_keypoints = array_kp[:, 29:, :]
        right_hand_conf = array_conf[:, 29:]

        item = {
            "body_kp": body_keypoints_2d,
            "body_conf": body_conf,
            "left_hand_kp": left_hand_keypoints,
            "left_hand_conf": left_hand_conf,
            "right_hand_kp": right_hand_keypoints,
            "right_hand_conf": right_hand_conf
        }

        return item

    def pad(self, item):

        pad_length = self.max_frames - item["body_kp"].shape[0]
        if pad_length > 0:
            for key in item.keys():
                if "kp" in key:
                    item[key] = numpy.pad(item[key], ((0, pad_length), (0,0), (0, 0)))
                elif "conf" in key:
                    item[key] = numpy.pad(item[key], ((0, pad_length), (0,0)))

        return item

    def clip(self, item):
        item["body_kp"] = item["body_kp"][:self.max_frames]
        item["right_hand_kp"] = item["right_hand_kp"][:self.max_frames]
        item["left_hand_kp"] = item["left_hand_kp"][:self.max_frames]
        item["body_conf"] = item["body_conf"][:self.max_frames]
        item["right_hand_conf"] = item["right_hand_conf"][:self.max_frames]
        item["left_hand_conf"] = item["left_hand_conf"][:self.max_frames]
        # item["json_paths"] = item["json_paths"][:self.max_frames]

        return item

    def select_frames(self, all_jsons):
        all_jsons = all_jsons[:self.max_frames]
        return all_jsons

    def to_tensor(self, item):
        item["body_kp"] = torch.tensor(item["body_kp"]).float()
        item["right_hand_kp"] = torch.tensor(item["right_hand_kp"]).float()
        item["left_hand_kp"] = torch.tensor(item["left_hand_kp"]).float()
        item["body_conf"] = torch.tensor(item["body_conf"]).float()
        item["right_hand_conf"] = torch.tensor(item["right_hand_conf"]).float()
        item["left_hand_conf"] = torch.tensor(item["left_hand_conf"]).float()

        return item


    def __getitem__(self, item):

        utt_id = self.utt_ids[item]

        keypoints = numpy.array(self.keypoints_data.get(utt_id))

        item = self.array2item(keypoints)

        item["text"] = self.utt_texts[utt_id]
        n_frames = item["body_kp"].shape[0]
        item["n_frames"] = min(n_frames, self.max_frames)
        item["utt_id"] = utt_id
        #
        # self.n_tokens += len(tokens)
        # self.n_utt += 1
        # Pad sequence to max len
        item = self.pad(item)

        # Clip sequence to max len
        # TODO This should not be needed since it is clipped in the all_jsons list
        item = self.clip(item)

        # To tensor
        item = self.to_tensor(item)
        if self.use_rand_tokens:
            item["text_tokens"] = self.random_tokens

        else:
            tokens = self.tokenizer.encode(item["text"]).ids
            tokens = tokens[:40]
            tokens = tokens + [0] * (40 - len(tokens))
            tokens = torch.tensor(tokens, dtype=torch.long)
            item["text_tokens"] = tokens
        if self.transform:
            item = self.transform(item)

        return item




