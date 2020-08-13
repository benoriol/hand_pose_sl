import glob
import cv2


def create_video(frames_folder, output_video, fps):

    frames = glob.glob(frames_folder + "/*")
    frames.sort()

    frame = cv2.imread(frames[0])

    height, width, channels = frame.shape

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'PIM1'), fps,
                             (width, height))

    for frame in frames:
        frame = cv2.imread(frame)
        writer.write(frame)



if __name__ == '__main__':

    frames_folder = "utterance_level/train/rgb_front/features/hand_pose_frames/fxIoWLKHOuo_4-3-rgb_front"
    output_video = "utterance_level/train/rgb_front/features/hand_pose_video/fxIoWLKHOuo_4-3-rgb_front.mp4"

    create_video(frames_folder, output_video, 24)