import cv2

vid_name = '06_21_2019_02'

vidPath = vid_name+'.mp4'

shotsPath = '%s.mp4'

video_names = [
#     'simple_cycling_1|Gaurav|06_21_2019_02|Cycling',
#     'simple_cycling_2|Gaurav|06_21_2019_02|Cycling',
    'cycle_with_phone_1|Gaurav|06_21_2019_02|Cycling',
#     'cycle_with_phone_2|Gaurav|06_21_2019_02|Cycling',
#     'texting_1|Gaurav|06_21_2019_02|Cycling', 
#     'texting_2|Gaurav|06_21_2019_02|Cycling',
#     'calling_1|Gaurav|06_21_2019_02|Cycling',
#     'calling_2|Gaurav|06_21_2019_02|Cycling',
#     'over_the_shoulder_1|Gaurav|06_21_2019_02|Cycling',
#     'over_the_shoulder_2|Gaurav|06_21_2019_02|Cycling',
#     'vlogging_1|Gaurav|06_21_2019_02|Cycling',
#     'vlogging_2|Gaurav|06_21_2019_02|Cycling',
]       

segRange = [
    (3500),
#     (4475),
#     (14725),
#     (15375),
#     (16250),
#     (16975),
#     (17400),
#     (18475),
#     (18900),
#     (19675),
#     (20125),
#     (20950),
]

cap = cv2.VideoCapture(vidPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

for idx, (begFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(shotsPath % video_names[idx], fourcc, fps, size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, begFidx)
        ret = True  # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
                ret, frame = cap.read()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if frame_number < begFidx+fps*3:
                        writer.write(frame)
                else:
                        break
        writer.release()
