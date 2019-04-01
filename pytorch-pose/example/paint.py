#!/usr/bin/python

from PIL import Image
from PIL import ImageDraw
import sys, subprocess

def main(path, cords = False):
    def get_cords(x,y):
        margin = 5
        x  = int(x)
        y = int(y)
        return (x-margin,y-margin,x+margin,y+margin)

    def draw_circle(x,y):
        lab.ellipse(get_cords(x,y),fill='red', width=10)

    
    def do_job():
        for x in cords:
                draw_circle(x[0],x[1])
    
    img = Image.open(path)
    lab = ImageDraw.Draw(img)

    if not cords:
        cords = []
        print('Enter cords')
        while True:
            line = raw_input()
            if line:
                cords.append(line)
            else:
                break
        cords = eval(''.join(cords))

    do_job()

    img.save('/tmp/output_pose.jpg')
    img.close()

    return subprocess.check_output('xdg-open /tmp/output_pose.jpg', shell=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} <file-path>".format(sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1])

