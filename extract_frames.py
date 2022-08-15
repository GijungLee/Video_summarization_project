from datetime import timedelta
import cv2
import numpy as np
import os

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 30

def get_saving_frames_durations(cap, saving_fps):
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_paths):
    path = video_paths.split("/")
    print(path)
    files = os.listdir(video_paths)
    if not os.path.isdir(path[-1]):
        os.mkdir(path[-1])
    #read the video file
    count = 0
    for video_file in files:
        print(video_file)
        video_file = os.path.join(video_paths, video_file)
        cap = cv2.VideoCapture(video_file)
        # get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Saving_frames_per_second: ", SAVING_FRAMES_PER_SECOND)
        print("fps: ", fps)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        print(saving_frames_per_second)
        # get the list of duration spots to save
        saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
        # start the loop
        # count = 0
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration,
                # then save the frame
                cv2.imwrite(os.path.join(path[-1], f"frame{count}.jpg"), frame)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1
if __name__ == "__main__":
    import sys
    # video_file = sys.argv[1]
    video_paths = "./video"
    main(video_paths)