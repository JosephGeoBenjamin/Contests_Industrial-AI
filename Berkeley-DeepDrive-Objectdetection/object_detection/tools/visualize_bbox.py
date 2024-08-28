import json
import numpy as np
import os
import sys
from PIL import ImageDraw, Image, ImageFont
FONT = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 11, encoding="unic")

class_codes = {
    "bike":"#ff8c00" ,
    "bus": "#00ffff",
    "car": "#000080",
    "motor": "#eee8aa" ,
    "person": "#7f0000",
    "rider": "#ffff00",
    "traffic light": "#ff00ff",
    "traffic sign": "#ff69b4",
    "train": "#2f4f4f",
    "truck": "#1e90ff",
}


def _draw_2dbbox(img, points, title):

    draw = ImageDraw.Draw(img)
    draw.rectangle(xy = [points["x1"], points["y1"], points["x2"], points["y2"]],
                        fill=None, outline=class_codes.get(title,0), width=2)

    txtsz = FONT.getsize(title)
    draw.rectangle(xy = [ points["x1"], points["y1"],
                          points["x1"]+txtsz[0], points["y1"]+txtsz[1]],
                    fill=class_codes.get(title, 0) )
    draw.text((points["x1"], points["y1"]), title, font=FONT, fill= (255,255,255))

    return img

def show_2dbbox_on_image(json_file, image_folder, save_prefix=""):

    json_data = json.load(open(json_file))

    for i, dic in enumerate(json_data):
        out_image = Image.open(os.path.join(image_folder, dic["name"]) )
        for j, jdic in enumerate(dic["labels"]):
            try:
                category = jdic["category"]
                points = jdic["box2d"]
                out_image =  _draw_2dbbox(out_image, points, category)
            except:
                pass
        out_image.save(os.path.join(save_prefix, dic["name"]), 'JPEG')
        print(i)

        # out_image.show()
        # sys.exit()


if __name__ == "__main__":
    BASE_PATH = "/home/jgeob/quater_ws/autonomousDriving/Vehicle-Intelligence_Perception/"

    show_2dbbox_on_image(
    json_file= BASE_PATH+"data/object_detect/object_detect_2dbbox_val.json",
    image_folder=BASE_PATH+"data/BDD_100k/val",
    save_prefix=BASE_PATH+"data/object_detect/sample_val"
    )