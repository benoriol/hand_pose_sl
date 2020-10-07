import json
from glob import glob

import argparse

import tqdm

parser = argparse.ArgumentParser(description="build metadata file for How2Sign")

parser.add_argument('--project-folder', type=str,
                    default="/home/benet/IRI")

parser.add_argument('--json-folder', type=str,
                    default="How2Sign/utterance_level/train/rgb_front/features/openpose_output/json",
                    help="Folder with all the folders with OpenPose json outputs")

parser.add_argument('--text-file', type=str,
                    default="How2Sign/utterance_level/train/text/en/raw_text/train.text.id.en",
                    help="Folder with all the folders with OpenPose json outputs")

parser.add_argument('--out-file', type=str,
                    default="/mnt/gpid07/users/benet.oriol/How2Sign_openpose.json",
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

    utterance_folder_path = args.json_folder

    utterance_folders = glob(utterance_folder_path + "/*")
    assert len(utterance_folders) > 0

    utterance_texts = read_text_file(args.text_file)
    n_text_not_found = 0

    output_data = []
    for utterance_folder in tqdm.tqdm(utterance_folders):

        utterance_name = utterance_folder.split("/")[-1]

        frame_jsons = glob(utterance_folder + "/*")
        frame_jsons.sort()

        utterance_dict = dict()

        utterance_dict["utt_id"] = utterance_name

        try:
            utterance_dict["text"] = utterance_texts[utterance_name]
        except KeyError:
            n_text_not_found += 1
            continue
        utterance_dict["n_frames"] = len(frame_jsons)
        #utterance_dict["frame_jsons"] = frame_jsons
        utterance_dict["frame_jsons"] = []
        for x in frame_jsons:
            utterance_dict["frame_jsons"].append({
                "json_path": x,
                "json_data": json.load(open(x))
            })

        output_data.append(utterance_dict)


    with open(args.out_file, "w") as fp:
        json.dump(output_data, fp)

    print("N correct utterances:\t", len(output_data))
    print("N text not found:\t", n_text_not_found)
    print("output file:\t", args.out_file)

