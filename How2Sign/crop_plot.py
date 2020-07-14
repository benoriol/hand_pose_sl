import cv2
import glob
import numpy as np
import json
from matplotlib import pyplot as plt
import time
import os



def crop_frame(frame, middle,  shape):
    '''
    Crops frame to given middle point and rectangle shape. 0 Pads outside points
    :param frame: Input frame. Numpy array
    :param middle: Center point. [x, y]
    :param shape: [x_shape, y_shape]
    :return: cropped frame. Numpy array
    '''
    frame = np.array(frame)
    frame = np.pad(frame, ((shape[0], shape[0]), (shape[1], shape[1]), (0, 0)))

    # Adjust to padded image coords
    middle = [middle[0] + shape[0], middle[1] + shape[1]]

    # frame_ = cv2.circle(frame, (int(middle[0]), int(middle[1])),
    #                            radius=4, color=(0, 255, 0), thickness=-1)
    #
    x_0, y_0 = int(middle[0] - shape[0] / 2), int(middle[1] - shape[1] / 2)
    x_1, y_1 = int(middle[0] + shape[0] / 2), int(middle[1] + shape[1] / 2)
    crop = frame[y_0:y_1, x_0:x_1, :]

    # frame_ = cv2.circle(frame_, (x_0, y_0),
    #                     radius=4, color=(0, 255, 0), thickness=-1)
    # frame_ = cv2.circle(frame_, (x_1, y_1),
    #                     radius=4, color=(0, 255, 0), thickness=-1)


    return crop


def plot_points(frame, points):
    frame = np.array(frame)
    for x, y in points:

        x, y = int(x), int(y)

        frame = cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255),
                           thickness=-1)
    plt.figure()
    plt.imshow(frame)
    plt.show()


def get_hand_center(input_json):
    '''
    Returs the computed hand center given the hand keypoints. Implemented as
    average of MP joints p
    :param input_json:
    :return:
    '''
    # Get right hand keypoints
    right_hand_points = input_json["people"][0]["hand_right_keypoints_2d"]

    # format list shape from (N_point x 3,) to (N_points, 3)
    right_hand_points = [right_hand_points[3 * i:3 * i + 3] for i in
                         range(len(right_hand_points) // 3)]

    # Selecting only MP joints
    MP_JOINTS_INDEXES = [5, 9, 13, 17]
    mp_joints = [right_hand_points[i] for i in MP_JOINTS_INDEXES]

    mp_joints_coordinates = [[x[0], x[1]] for x in mp_joints]
    #plot_points(frame_large, mp_joints_coordinates)

    mp_joints_coordinates_numpy = np.array(mp_joints_coordinates)

    mp_joints_center = np.average(mp_joints_coordinates_numpy, axis=0)

    return mp_joints_center


def crop_image_main():
    inputpath = "utterance_level/train/rgb_front/frames/1eFlDHpjPNI_7-8-rgb_front/frame-0000.png"
    json_path = ""
    frame_large = cv2.imread(inputpath)
    if frame_large is None:
        raise FileNotFoundError("Image in path " + inputpath + " not found")
    frame_large = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

    input_json = json.load(open(
        "utterance_level/train/rgb_front/features/json/1eFlDHpjPNI_7-8-rgb_front/1eFlDHpjPNI_7-8-rgb_front_000000000000_keypoints.json"))



    center_coords = get_hand_center(input_json)

    # frame_large_ = cv2.circle(frame_large, (int(center_coords[0]), int(center_coords[1])),
    #                    radius=4, color=(0, 255, 0), thickness=-1)
    # plt.figure()
    # plt.imshow(frame_large_)
    # plt.show()
    crop = crop_frame(frame_large, center_coords, (200, 200))
    plt.figure()
    plt.imshow(crop)
    plt.show()


def crop_video_main():


    input_video_path = "utterance_level/train/rgb_front/raw_videos/fxIoWLKHOuo_4-3-rgb_front.mp4"
    input_json_folder = "utterance_level/train/rgb_front/features/json/fxIoWLKHOuo_4-3-rgb_front"
    output_file = "utterance_level/train/rgb_front/features/hand_video/fxIoWLKHOuo_4-3-rgb_front.mp4"
    utt_id = input_video_path.split("/")[-1].replace(".mp4", "")
    print(utt_id)
    cap = cv2.VideoCapture(input_video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n = 0

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'PIM1'), 24, (200, 200))
    while (cap.isOpened()):
        ret, frame_large = cap.read()

        if frame_large is None:
            break

        #frame_large = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

        json_filename = utt_id + "_" + '{:012d}'.format(n) + "_keypoints.json"
        json_filename = input_json_folder + "/" + json_filename

        keypoints_json = json.load(open(json_filename))
        center_coords = get_hand_center(keypoints_json)
        crop = crop_frame(frame_large, center_coords, (200, 200))

        writer.write(crop)

        n += 1

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    crop_video_main()


