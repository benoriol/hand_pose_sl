import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--in-file', type=str,
                    default="metadata/metadata_dev.json",
                    help="path to output metadata file")

if __name__ == '__main__':


    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0

    args = parser.parse_args()
    data = json.load(open(args.in_file))

    n_total = len(data)

    n_train = int(TRAIN_SPLIT * n_total)
    n_valid = int(VAL_SPLIT * n_total)


    train_data = data[:n_train]
    valid_data = data[-n_valid:]
    test_data = data[n_train:-n_valid]

    assert len(train_data) + len(valid_data) + len(test_data) == len(data)

    train_filename = args.in_file.replace(".json", ".train.json")
    with open(train_filename, "w") as fp:
        json.dump(train_data, fp, indent=4)

    valid_filename = args.in_file.replace(".json", ".valid.json")
    with open(valid_filename, "w") as fp:
        json.dump(valid_data, fp, indent=4)

    test_filename = args.in_file.replace(".json", ".test.json")
    with open(test_filename, "w") as fp:
        json.dump(test_data, fp, indent=4)

    print("Original:\t" + args.in_file + " :\t" + str(len(data)))
    print("Train:\t" + train_filename + " :\t" + str(len(train_data)))
    print("Validation:\t" + valid_filename + " :\t" + str(len(valid_data)))
    print("Test:\t" + test_filename + " :\t" + str(len(test_data)))




