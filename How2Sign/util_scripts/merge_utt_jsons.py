import argparse
import glob
import json
import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--inp-folder",  type=str,
                    default="/mnt/cephfs/How2Sign/How2Sign/utterance_level/train/rgb_front/features/openpose_output/json")

parser.add_argument("--out-folder", type=str,
                    default="/home/usuaris/imatge/boriol/IRI/How2Sign/merged_jsons")


if __name__ == '__main__':
    args = parser.parse_args()

    # Look for all utterances
    folders = glob.glob(args.inp_folder + "/*")

    assert len(folders) > 0

    for utterance_folder in tqdm.tqdm(folders):

        out_data = []

        # Look for all frames in an utterance
        json_paths = glob.glob(utterance_folder + "/*.json")
        utt_id = utterance_folder.split("/")[-1]

        for utterance_json_path in sorted(json_paths):

            frame_json_id = utterance_json_path.split("/")[-1].replace(".json", "")

            json_data = json.load(open(utterance_json_path))

            out_dict = {
                "id":  frame_json_id,
                "data": json_data
            }

            out_data.append(out_dict)

        out_path = args.out_folder + "/" + utt_id + ".json"
        with open(out_path, "w") as f:
            json.dump(out_data, f)








