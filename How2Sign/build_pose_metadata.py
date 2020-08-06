import json
from glob import glob

import argparse

parser = argparse.ArgumentParser(description="build metadata file for How2Sign")

parser.add_argument('--json-folder', type=str,
                    default="utterance_level/train/rgb_front/features/json",
                    help="Folder with all the folders with OpenPose json outputs")

parser.add_argument('--text-file', type=str,
                    default="utterance_level/train/text/en/raw_text/train.text.id.en",
                    help="Folder with all the folders with OpenPose json outputs")

parser.add_argument('--out-file', type=str,
                    default="utterance_level/train/rgb_front/features/pose_metadata_dev.json",
                    help="path to output metadata file")

def read_text_file(text_file_path):
    f = open(text_file_path)
    utterance_texts = {}
    for line in f:
        line = line.strip()
        utt_id, text = line.split(maxsplit=1)

        utterance_texts[utt_id] = text

    return utterance_texts

if __name__ == '__main__':
    args = parser.parse_args()

    utterance_folders = glob(args.json_folder + "/*")
    assert len(utterance_folders) > 0

    utterance_texts = read_text_file(args.text_file)

    output_data = []
    for utterance_folder in utterance_folders:


        utterance_name = utterance_folder.split("/")[-1]

        print(utterance_name)

        frame_jsons = glob(utterance_folder + "/*")
        frame_jsons.sort()

        utterance_dict = dict()
        utterance_dict["utt_id"] = utterance_name
        utterance_dict["text"] = utterance_texts[utterance_name]
        utterance_dict["n_frames"] = len(frame_jsons)
        #utterance_dict["frame_jsons"] = frame_jsons
        utterance_dict["frame_jsons_folder"] = utterance_folder

        output_data.append(utterance_dict)

    f = open("hola", "w")

    with open(args.out_file, "w") as fp:
        json.dump(output_data, fp , indent=4)

    print("N utterances:\t", len(utterance_folders))
    print("output file:\t", args.out_file)
