"""2d_vis.py: Create a visualization of 2d keypoints
read in a json file and visualize pose, face and hands
Paths - should be the folder where Open Pose JSON output was stored"""

"""Credit to:
https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/20-02-15_visualization/2d_vis.py
"""

import json
import math
import sys
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cv2 import LINE_AA
import os
from pathlib import Path


class JSONVis:

    def __init__(self, width, height, FPS):
        self.width = width
        self.height = height
        self.FPS = FPS

        self.dist_thumb = []
        self.confidences = {"conf_face": [], "conf_pose": [], "conf_hand_l": [], "conf_hand_r": []}
        self.finger_points_colors = {"thumb": "salmon", "index_finger": "goldenrod", "middle_finger": "springgreen",
                                     "ring_finger": "navy", "little_finger": "darkviolet"}
        self.finger_length_r = {"thumb": [], "index_finger": [], "middle_finger": [],
                                "ring_finger": [], "little_finger": []}
        self.finger_length_l = {"thumb": [], "index_finger": [], "middle_finger": [],
                                "ring_finger": [], "little_finger": []}
        self.finger_length = {"right": self.finger_length_r, "left": self.finger_length_l}

    def get_points(self, key, file=None):
        temp_df = json.load(open(self.path_to_json / self.file))
        if file is not None:
            temp_df = json.load(open(self.path_to_json / file))
        temp_x_pose = temp_df['people'][0][key][0::3]
        temp_y_pose = temp_df['people'][0][key][1::3]
        return [temp_x_pose, temp_y_pose]

    def get_confidence(self, key_file, file=None):
        """
        Get the confidence of one json file of a specific key_file
        :param key_file: the specific key of a json file
        :param file: a json file the confidence of a key is computed
        :return: mean confidence of all values of a key in a certain file, whole array of all confidence values
        """
        temp_df = json.load(open(self.path_to_json / self.file))
        if file is not None:
            temp_df = json.load(open(self.path_to_json / file))
        temp_conf = temp_df['people'][0][key_file][2::3]
        return np.mean(temp_conf), temp_conf

    def cl(self, str):
        """
        Get color on range 0 to 255. Its not rgb (red, green, blue) here, but b,g,r (blue, green, red)
        :param str: name of the needed color
        :return: scalar color code (blue, green, red)
        """
        switched_colors = np.array(mcolors.to_rgb(str)).dot(255)
        switched_colors = np.array([switched_colors[2], switched_colors[1], switched_colors[0]])
        return switched_colors

    def draw_pose(self, frame, key):
        points = self.get_points(key)

        """
        Build pose: right arm is the person's right arm. not the viwer's right arm:
        [0-1]:          neck
        [1-4]:          right arm
        [1-5,6,7]:      left arm
        [1-8]:          back
        [8-11]:         right leg
        [8-12,13,14]:   left leg
        [24, 11, 22, 23]: right foot - not implemented
        [21, 14, 19, 20]: left foot - not implemented
        [17, 15, 0, 16, 18]: head
        position from: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        """

        xs = [int(i) for i in points[0]]
        ys = [int(i) for i in points[1]]

        # neck
        cv2.line(frame, (xs[0], ys[0]), (xs[1], ys[1]), self.cl("gray"), 2, LINE_AA)
        cv2.circle(frame, (xs[0], ys[0]), 4, self.cl('white'), 3)
        cv2.circle(frame, (xs[1], ys[1]), 4, self.cl('white'), 3)
        # print("From %d, %d to %d, %d" % (xs[0], ys[0], xs[1], ys[1]))

        # back
        cv2.line(frame, (xs[1], ys[1]), (xs[8], ys[8]), self.cl('red'), 3)

        # right arm
        joints_x = xs[1:5]
        joints_y = ys[1:5]
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx+1], joints_y[idx+1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]),
                         self.cl("orange"), 3, LINE_AA)

        # left arm
        joints_x = xs[5:8]
        joints_y = ys[5:8]
        cv2.line(frame, (xs[1], ys[1]), (xs[5], ys[5]), self.cl('lime'), 3, LINE_AA)
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]), self.cl("lime"),
                         3, LINE_AA)

        # Currently removed visualization of legs
        # right leg
        """joints_x = xs[8:12]
        joints_y = ys[8:12]
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]),
                         self.cl("green"), 3, LINE_AA)"""

        # left leg
        """joints_x = xs[12:15]
        joints_y = ys[12:15]
        cv2.line(frame, (xs[8], ys[8]), (xs[12], ys[12]), self.cl('cyan'), 3, LINE_AA)
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]), self.cl("cyan"),
                         3, LINE_AA)"""
        # head
        cv2.line(frame, (xs[17], ys[17]), (xs[15], ys[15]), self.cl("pink"), 2, LINE_AA)
        cv2.line(frame, (xs[15], ys[15]), (xs[0], ys[0]), self.cl("magenta"), 2, LINE_AA)
        cv2.line(frame, (xs[0], ys[0]), (xs[16], ys[16]), self.cl("purple"), 2, LINE_AA)
        cv2.line(frame, (xs[16], ys[16]), (xs[18], ys[18]), self.cl("orchid"), 3, LINE_AA)

        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey()
        return frame

    def draw_face(self, frame, key, thickness):
        points = self.get_points(key)

        xs = [int(i) for i in points[0]]
        ys = [int(i) for i in points[1]]
        poly = np.array([xs, ys]).T.tolist()

        # face shape
        cv2.polylines(frame, np.int32([poly[0:16]]), 0, self.cl("white"), thickness)
        # right eye brow
        cv2.polylines(frame, np.int32([poly[17:21]]), 0, self.cl("white"), thickness)
        # left eye brow
        cv2.polylines(frame, np.int32([poly[22:26]]), 0, self.cl("white"), thickness)
        # nose
        cv2.polylines(frame, np.int32([poly[27:30]]), 0, self.cl("white"), thickness)
        cv2.polylines(frame, np.int32([poly[31:35]]), 0, self.cl("white"), thickness)
        # right eye
        cv2.polylines(frame, np.int32([poly[36:41]]), 1, self.cl("white"), thickness)
        # left eye
        cv2.polylines(frame, np.int32([poly[42:47]]), 1, self.cl("white"), thickness)
        # mouth
        cv2.polylines(frame, np.int32([poly[48:59]]), 1, self.cl("white"), thickness)
        cv2.polylines(frame, np.int32([poly[60:67]]), 1, self.cl("white"), thickness)
        return frame

    def dist(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def draw_hand(self, frame, key_file, thickness, idx, file=None):
        points = self.get_points(key_file, file)
        confidence = self.get_confidence(key_file, file)[0]
        conf_array = self.get_confidence(key_file, file)[1]
        # print(conf_array)
        # confidence levels to show hands or single fingers, same levels for left and right hand are used
        confidence_overall = 0.09
        confidence_each_finger = 0.2
        max_finger_length = 230

        # use finger points for computing the length of the fingers
        finger_points = {}
        finger_confidence = {}

        # obtain joint positions
        xs = [int(i) for i in points[0]]
        ys = [int(i) for i in points[1]]
        poly = np.array([xs, ys]).T.tolist()

        # Fill dictionary with finger joint points, used this points
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format

        # thumb
        finger_points["thumb"] = poly[0:5]
        finger_confidence["thumb"] = conf_array[0:5]
        # index
        # Counts for the other finger as well, added joints to dictionary differently so it stay in  pairs
        finger_points["index_finger"] = [poly[0]]
        finger_points["index_finger"].extend([poly[5]])
        finger_points["index_finger"].extend(poly[5:9])
        finger_confidence["index_finger"] = conf_array[5:9]
        # middle
        finger_points["middle_finger"] = [poly[0]]
        finger_points["middle_finger"].extend([poly[9]])
        finger_points["middle_finger"].extend(poly[9:13])
        finger_confidence["middle_finger"] = conf_array[9:13]
        # ring
        finger_points["ring_finger"] = [poly[0]]
        finger_points["ring_finger"].extend([poly[13]])
        finger_points["ring_finger"].extend(poly[13:17])
        finger_confidence["ring_finger"] = conf_array[13:17]

        # little
        finger_points["little_finger"] = [poly[0]]
        finger_points["little_finger"].extend([poly[17]])
        finger_points["little_finger"].extend(poly[17:21])
        finger_confidence["little_finger"] = conf_array[17:21]

        # calculate length of fingers and write them into dictionary for plotting
        for key in finger_points:
            total_dist = 0.0
            for i in range(len(finger_points[key]) - 1):
                # calculate length of fingers
                total_dist += self.dist(finger_points[key][i], finger_points[key][i + 1])

            # add length into dictionary with respect of side
            if key_file == "hand_left_keypoints_2d":
                self.finger_length_l[key].append(total_dist)
            else:
                self.finger_length_r[key].append(total_dist)

            # show finger if confidence is higher than threshold
            if confidence > confidence_overall:
                if np.mean(finger_confidence[key]) > confidence_each_finger and total_dist < max_finger_length:
                    cv2.polylines(frame, np.int32([finger_points[key]]), 0, self.cl(self.finger_points_colors[key]),
                                  thickness)

        else:
            pass
            # print("Confidence overall lower than  %d at frame %d - hand: %s" % (confidence_overall, idx, str(key[:10]).split("_")[1]))

        if confidence > 0.0:
            new_frame = frame
            return new_frame
        else:
            return frame

    def draw_main(self, path_to_json_dir):
        # get subdirectories of the path
        os.walk(path_to_json_dir)
        subdirectories = [x[1] for x in os.walk(path_to_json_dir)]
        data_dir_origin = Path(path_to_json_dir)
        subdirectories = subdirectories[0]

        # create new target directory, the centralized fiels will be saved there
        if not os.path.exists(data_dir_origin.parent / str(data_dir_origin.name + "_visualized")):
            os.makedirs(data_dir_origin.parent / str(data_dir_origin.name + "_visualized"))

        data_dir_target = data_dir_origin.parent / str(data_dir_origin.name + "_visualized")

        for subdir in subdirectories:
            if not os.path.exists(data_dir_target / subdir):
                os.makedirs(data_dir_target / subdir)

        for subdir in subdirectories:
            self.json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir) if pos_json.endswith('.json')]
            self.path_to_json = data_dir_origin / subdir
            print('Found: %d json keypoint frame files in folder %s' % (len(self.json_files), self.path_to_json))
            self.path_to_output = data_dir_target / subdir
            self.draw()
            print("%s done" % subdir)

    def draw(self):
        self.json_files.sort()
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        video = cv2.VideoWriter(str(self.path_to_output / 'json_vis.avi'), fourcc, float(self.FPS), (self.width, self.height))
        idx = 0
        once = 1
        for file in self.json_files:
            # set file for class
            self.file = file

            # save first frame for left and right hand
            if once == 1:
                file_save_r = file
                file_save_l = file
                once = 0

            # used for transparency
            blank_image = np.zeros((height, width, 3), np.uint8)
            blank_image_one = np.ones((height, width, 3), np.uint8)
            frame = self.draw_pose(blank_image_one, 'pose_keypoints_2d')
            alpha = 0.5  # Transparency factor.
            # Following line overlays transparent over the image
            frame = cv2.addWeighted(frame, alpha, blank_image, 1 - alpha, 0)
            frame = self.draw_face(frame, 'face_keypoints_2d', 1)

            # try to display last frame when confidence is too low instead of not showing the frame
            # conf is computed here only for the left hand
            # conf level set to 0.0 (deactivated it) because it looks laggy
            conf_hand_l = self.get_confidence('hand_left_keypoints_2d', file)[0]
            conf_hand_r = self.get_confidence('hand_right_keypoints_2d', file)[0]

            if conf_hand_l > 0.0:
                frame = self.draw_hand(frame, 'hand_left_keypoints_2d', 2, idx, file)
                file_save_l = file
            else:
                frame = self.draw_hand(frame, 'hand_left_keypoints_2d', 2, idx, file_save_l)

            if conf_hand_r > 0.0:
                frame = self.draw_hand(frame, 'hand_right_keypoints_2d', 2, idx, file)
                file_save_r = file
            else:
                frame = self.draw_hand(frame, 'hand_right_keypoints_2d', 2, idx, file_save_r)

            # write index on frames
            cv2.putText(frame, str(idx), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            video.write(frame)


            # helper function to visualize when setting max finger length
            # how frames when fingers are longer than threshold
            # max_finger_length = 250
            # for side in self.finger_length:
            #     for key in self.finger_length[side]:
            #         self.write_file(frame, key, idx, side, max_finger_length)

            # show computing status in console
            # if idx % 100 == 0:
            #     print("Frame: %d of %d" % (idx, len(self.json_files)))
            # self.fill_plotter_data(file)
            # idx += 1

        self.plot_data()
        video.release()

    def write_file(self, frame, key, idx, side, threshold_length):
        save_img_path = self.path_to_output / "finger_pictures"
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        new_path = save_img_path / str("over_" + str(threshold_length)) / side
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if self.finger_length[side][key][idx] > threshold_length:
            # print("%s %s length more than %d in frame: %d" % (side, key, threshold_length, idx))
            cv2.imwrite(new_path / str(side + "_" + str(key) + "_over-" + str(threshold_length) + "_frame-" +
                                       str(idx) + ".jpg", frame))

    def plot_data(self):

        if not os.path.exists(self.path_to_output / "plots"):
            os.makedirs(self.path_to_output / "plots")

        path_to_plots = self.path_to_output / "plots"

        plt.figure(1)
        plt.figure(figsize=(12, 7.2))
        ax1 = plt.subplot(211)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.plot(self.confidences['conf_face'], label='face')
        plt.ylabel('Confidences')
        plt.legend()
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.confidences['conf_pose'], label='pose', color="tab:orange")
        plt.legend()
        plt.ylabel('Confidences')
        plt.xlabel('Frames')
        plt.savefig(path_to_plots / 'confidences-face-pose.png')

        plt.figure(2)
        plt.figure(figsize=(12, 7.2))
        ax1 = plt.subplot(211)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.plot(self.confidences['conf_hand_l'], label='hand left')
        plt.legend()
        plt.ylabel('Confidences')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.confidences['conf_hand_r'], label='hand right', color="tab:orange")
        plt.legend()
        plt.ylabel('Confidences')
        plt.xlabel('Frames')

        plt.savefig(path_to_plots / 'confidences-hands.png')

        plt.figure(3)
        self.plot_finger_length("thumb", path_to_plots)

        plt.figure(4)
        self.plot_finger_length("index_finger", path_to_plots)

        plt.figure(5)
        self.plot_finger_length("middle_finger", path_to_plots)

        plt.figure(6)
        self.plot_finger_length("ring_finger", path_to_plots)

        plt.figure(7)
        self.plot_finger_length("little_finger", path_to_plots)

        plt.show()

    def plot_finger_length(self, label, path_to_plots):
        plt.figure(figsize=(12, 7.2))
        ax1 = plt.subplot(211)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.plot(self.finger_length_l[label], label=label + '_length_l')
        y_mean = [np.mean(self.finger_length_l[label])] * len(self.finger_length_l[label])
        plt.plot(y_mean, label=label + 'avg_length_l')
        plt.legend()
        plt.ylabel('Length')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.finger_length_r[label], label=label + '_length_r', color="tab:green")
        y_mean = [np.mean(self.finger_length_r[label])] * len(self.finger_length_r[label])
        plt.plot(y_mean, label=label + '_mean_length_r', color="tab:red")
        plt.legend()
        plt.xlabel('Frames')
        plt.ylabel('Length')

        plt.savefig(path_to_plots / str("lengths-" + label + ".png"))

    def fill_plotter_data(self, file):
        # get confidences and add them to a dictionary
        self.confidences['conf_face'].append(self.get_confidence('face_keypoints_2d', file)[0])
        self.confidences['conf_pose'].append(self.get_confidence('pose_keypoints_2d', file)[0])
        self.confidences['conf_hand_l'].append(self.get_confidence('hand_left_keypoints_2d', file)[0])
        self.confidences['conf_hand_r'].append(self.get_confidence('hand_right_keypoints_2d', file)[0])


if __name__ == '__main__':
    width = 1280
    height = 720
    FPS = 15

    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        path_to_json_dir = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json"
    vis = JSONVis(width, height, FPS)
    start_time = time.time()
    vis.draw_main(path_to_json_dir)

    print("--- %s seconds ---" % (time.time() - start_time))

    path_to_json_dir = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json\\"

