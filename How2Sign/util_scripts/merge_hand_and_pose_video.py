
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rescale_and_crop(image, size):

    heigth, width, _ = image.shape

    target_widht = int(width / heigth * size)

    downsized = cv2.resize(image, (target_widht, size))

    margin = (target_widht - size) // 2

    if margin != 0:
        cropped = downsized[:, margin:-margin-1]
    else:
        cropped = downsized

    return cropped


def merge_hand_and_pose_videos(hand_video, pose_video, out_video):


    hand_cap = cv2.VideoCapture(hand_video)

    hand_length = int(hand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_fps = hand_cap.get(cv2.CAP_PROP_FPS)

    pose_cap = cv2.VideoCapture(pose_video)

    pose_length = int(pose_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_fps = int(pose_cap.get(cv2.CAP_PROP_FPS))

    assert pose_length == hand_length
    # assert pose_fps == hand_fps

    hand_width = int(hand_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hand_height = int(hand_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas = np.zeros((hand_height, 2 * hand_width, 3), dtype=np.uint8)

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'PIM1'), hand_fps,
                             (2*hand_width, hand_width))


    for kk in range(pose_length):
        ret, hand_frame = hand_cap.read()
        ret, pose_frame = pose_cap.read()

        pose_frame_rescaled_cropped = rescale_and_crop(pose_frame, hand_height)

        canvas[:, :200] = hand_frame
        canvas[:, 200:] = pose_frame_rescaled_cropped

        writer.write(canvas)



if __name__ == '__main__':

    hand_video = "utterance_level/train/rgb_front/features/hand_video/fxIoWLKHOuo_4-3-rgb_front.mp4"
    pose_video = "utterance_level/train/rgb_front/features/hand_openpose_video/fxIoWLKHOuo_4-3-rgb_front.mp4"
    output_video = "utterance_level/train/rgb_front/features/hand_and_openpose_video/fxIoWLKHOuo_4-3-rgb_front.mp4"

    merge_hand_and_pose_videos(hand_video, pose_video, output_video)