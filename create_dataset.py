import json
from pathlib import Path
from shutil import copy


ANNOTATION_FILE = "all_annotations.json"
CLASSES = {
    0: "ribs",
    1: "chop",
    2: "salad1",
    3: "salad2",
    4: "bortsch",
    5: "pumpkin soup"
}
CONFIG_FILE_NAME = "dataset.yaml"
NEGATIVE_SAMPLES = ["split_video32", "split_video4"]
SOURCE = "./data"
TRAIN_VIDEO = ["split_video21"]
VAL_VIDEO = ["split_video31", "split_video32", "split_video4"]

# names of folders for images and labels, also for training and validation datasets
IMAGES = "images"
LABELS = "labels"
TRAIN_FOLDER = "train"
VAL_FOLDER = "val"

NEGATIVE_RATIO = 10
SELECTION_RATIO = 3
VAL_PART = 5

H, W = 960, 540

def main() -> None:

    with open(f"{SOURCE}/{ANNOTATION_FILE}") as jsonfile:
        all_annotations = json.load(jsonfile)

    image_folders = [fldr for fldr in Path(SOURCE).iterdir() if "split" in fldr.name]

    # Creating yaml-файл, описывающий датасет
    string = f"path: {SOURCE}\ntrain: {IMAGES}/{TRAIN_FOLDER}\nval: {IMAGES}/{VAL_FOLDER}\ntest:\n\nnames:\n  "
    string += "\n  ".join([f"{k}: {v}" for k,v in CLASSES.items()])

    Path(f"{SOURCE}/{IMAGES}/{TRAIN_FOLDER}").mkdir( parents=True, exist_ok=True)
    Path(f"{SOURCE}/{IMAGES}/{VAL_FOLDER}").mkdir( parents=True, exist_ok=True)
    Path(f"{SOURCE}/{LABELS}/{TRAIN_FOLDER}").mkdir( parents=True, exist_ok=True)
    Path(f"{SOURCE}/{LABELS}/{VAL_FOLDER}").mkdir( parents=True, exist_ok=True)

    with open(CONFIG_FILE_NAME, "w") as yamlfile:
        yamlfile.write(string)

    for fldr in image_folders:
        files_list = sorted(list(Path(fldr).iterdir()))
        if fldr.stem in NEGATIVE_SAMPLES:
            negative_samples = files_list[::NEGATIVE_RATIO][::VAL_PART]
            for file in negative_samples:
                copy(file, f"{SOURCE}/{IMAGES}/{VAL_FOLDER}/{fldr.name}_{file.name}")
                with open(f"{SOURCE}/{LABELS}/{VAL_FOLDER}/{fldr.name}_{file.stem}.txt", "w") as txtfile:
                    txtfile.write("")
        else:
            selected_images = files_list[::SELECTION_RATIO]
            if fldr.stem in TRAIN_VIDEO:
                selected_images_train = selected_images
                for file in selected_images_train:
                    copy(file, f"{SOURCE}/{IMAGES}/{TRAIN_FOLDER}/{fldr.name}_{file.name}")
                    label_file = f"{SOURCE}/{LABELS}/{TRAIN_FOLDER}/{fldr.name}_{file.stem}.txt"

                    annotation_list = []
                    if f"{file.parent.name}-{file.name}" in all_annotations:
                        annotations = all_annotations[f"{file.parent.name}-{file.name}"]
                        for cls in annotations:
                            y1, y2, x1, x2 = annotations[cls]
                            annotation_list.append(f"{cls} {(x2+x1)/(2*W)} {(y2+y1)/(2*H)} {(x2-x1)/W} {(y2-y1)/H}")

                    with open(label_file, "w") as txtfile:
                        txtfile.write("\n".join(annotation_list))
            if fldr.stem in VAL_VIDEO:
                selected_images_val = selected_images[::VAL_PART]

                for file in selected_images_val:
                    copy(file, f"{SOURCE}/{IMAGES}/{VAL_FOLDER}/{fldr.name}_{file.name}")
                    label_file = f"{SOURCE}/{LABELS}/{VAL_FOLDER}/{fldr.name}_{file.stem}.txt"

                    annotation_list = []
                    if f"{file.parent.name}-{file.name}" in all_annotations:
                        annotations = all_annotations[f"{file.parent.name}-{file.name}"]
                        for cls in annotations:
                            y1, y2, x1, x2 = annotations[cls]
                            annotation_list.append(f"{cls} {(x2+x1)/(2*W)} {(y2+y1)/(2*H)} {(x2-x1)/W} {(y2-y1)/H}")

                    with open(label_file, "w") as txtfile:
                        txtfile.write("\n".join(annotation_list))

    move("data/images/val/split_video32_000.jpeg", "data/images/train/split_video32_000.jpeg")
    move("data/images/val/split_video32_000.txt", "data/images/train/split_video32_000.txt")


if __name__ == "__main__":
    main()

