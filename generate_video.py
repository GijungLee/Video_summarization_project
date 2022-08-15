import cv2
import pickle

pathOut = 'Summarized_result.mp4'
fps = 15
with open("./important_frames", 'rb') as f:
    important_frames1 = pickle.load(f)

frame_array = []
for i in important_frames1:
    filename = f"./sam/Images/fish-big/frame{i}.jpg"
    # reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    # inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()