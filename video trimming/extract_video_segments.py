import cv2

vid_name = '02_15_2019_01'

vidPath = vid_name+'.mp4'

shotsPath = '%s.mp4'
# a list of starting/ending frame indices pairs

video_names = [
    'simple_walking_1|Amrit|02_15_2019_01|walking',
    'simple_walking_2|Amrit|02_15_2019_01|walking',
    'walk_with_phone_1|Amrit|02_15_2019_01|walking',
    'walk_with_phone_2|Amrit|02_15_2019_01|walking',
    'calling_1|Amrit|02_15_2019_01|walking',
    'calling_2|Amrit|02_15_2019_01|walking',
    'texting_1|Amrit|02_15_2019_01|walking',
    'texting_2|Amrit|02_15_2019_01|walking',
    'vlogging_1|Amrit|02_15_2019_01|walking',
    'vlogging_2|Amrit|02_15_2019_01|walking',
    'over_the_shoulder_1|Amrit|02_15_2019_01|walking',
    'over_the_shoulder_2|Amrit|02_15_2019_01|walking',
]

segRange = [
    (8175, 8300),
    (8475, 8600),
    (9450, 9600),
    (9775, 9925),
    (10800, 10950),
    (11100, 11250),
    (12850, 12975),
    (13100, 13250),
    (14825, 14975),
    (15100, 15250),
    (19300, 19450),
    (19575, 19725),
]


cap = cv2.VideoCapture(vidPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

for idx, (begFidx, endFidx) in enumerate(segRange):
        # print(idx)
        # print((begFidx, endFidx))
        # print('\n')
        writer = cv2.VideoWriter(shotsPath % video_names[idx], fourcc, fps, size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, begFidx)
        ret = True  # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
                ret, frame = cap.read()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if frame_number < endFidx:
                        writer.write(frame)
                else:
                        break
        writer.release()
