from collections import defaultdict
import json
import os

import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import torch


ANNOTATION_FILE = "all_annotations.json"
# список пар (название видео, координаты объекта для отслеживания)
# названия видео генерируются автоматически в процессе подготовки датасета
OBJECTS = [
    ("split_video21", np.array([320, 450, 510, 620], dtype=np.float32), 0, 0),
    ("split_video21", np.array([30, 410, 220, 630], dtype=np.float32), 0, 1),
    ("split_video21", np.array([50, 290, 220, 410], dtype=np.float32), 0, 2),
    ("split_video21", np.array([365, 335, 475, 430], dtype=np.float32), 0, 3),
    ("split_video21", np.array([445, 150, 540, 240], dtype=np.float32), 58, 4),
    ("split_video21", np.array([110, 180, 190, 260], dtype=np.float32), 259, 5),
    ("split_video31", np.array([325, 450, 510, 620], dtype=np.float32), 0, 0),
    ("split_video31", np.array([20, 410, 225, 625], dtype=np.float32), 0, 1),
    ("split_video31", np.array([55, 275, 225, 405], dtype=np.float32), 0, 2),
    ("split_video31", np.array([360, 330, 475, 430], dtype=np.float32), 0, 3),
    ("split_video31", np.array([360, 215, 450, 300], dtype=np.float32), 0, 4),
    ("split_video31", np.array([100, 180, 180, 260], dtype=np.float32), 0, 5),
    ("split_video32", np.array([380, 310, 540, 440], dtype=np.float32), 0, 0),
    ("split_video32", np.array([50, 330, 230, 520], dtype=np.float32), 0, 1),
    ("split_video32", np.array([80, 210, 240, 330], dtype=np.float32), 0, 2),
    ("split_video32", np.array([290, 275, 395, 370], dtype=np.float32), 0, 3),
    ("split_video32", np.array([290, 180, 380, 260], dtype=np.float32), 0, 4),
    ("split_video32", np.array([120, 150, 190, 220], dtype=np.float32), 0, 5),
    ("split_video4", np.array([370, 330, 540, 460], dtype=np.float32), 0, 0),
    ("split_video4", np.array([55, 325, 230, 520], dtype=np.float32), 0, 1),
]
SOURCE = "../data"

def main() -> None:
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


    sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    all_annotations = defaultdict(list)

    for object_data in OBJECTS:

        # video_annotations = {}
        video_dir_name, box_coords, ann_frame_idx, ann_obj_id = object_data

        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        video_dir = f"{SOURCE}/{video_dir_name}"

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)

        box = box_coords
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
        # run propagation throughout the video
        for out_frame_idx, out_obj_id, out_mask_logits in predictor.propagate_in_video(inference_state):
            segment= [(out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)]
            if segment[0].sum() != 0:
                img = (segment[0]*255).transpose(1,2,0).repeat(3,2).astype(np.uint8)

                h, w = np.nonzero(img)[:2]
                y1, y2, x1, x2 = min(h), max(h), min(w), max(w)
                # video_annotations[frame_names[out_frame_idx]] = (int(y1), int(y2), int(x1), int(x2))
                if f"{video_dir_name}-{frame_names[out_frame_idx]}" in all_annotations:
                    all_annotations[f"{video_dir_name}-{frame_names[out_frame_idx]}"][ann_obj_id] = (int(y1), int(y2), int(x1), int(x2))
                else:
                    all_annotations[f"{video_dir_name}-{frame_names[out_frame_idx]}"] = {ann_obj_id: (int(y1), int(y2), int(x1), int(x2))}
        
        predictor.reset_state(inference_state)
        # all_annotations[f"{video_dir_name}-{ann_obj_id}"] = video_annotations

    with open(f"{SOURCE}/{ANNOTATION_FILE}", "w") as jsonfile:
        json.dump(
            all_annotations, 
            jsonfile,
            ensure_ascii=False,
            indent=4)


if __name__ == "__main__":
    main()
