import json
import os
import imageio
import numpy as np

# --------------------------------------------------------------------------------------
### Hardcoded values for NeRF Synthetic dataset
# --------------------------------------------------------------------------------------
DATASET_ROOT_DIR = "../../data/nerf_synthetic/lego/" # change this to your dataset dir
NEAR, FAR = 2.0, 6.0
SPLITS = ["train", "val", "test"]

# --------------------------------- Main script -------------------------------------- #
meta_jsons = {}
for split in SPLITS:
    with open(os.path.join(DATASET_ROOT_DIR, f"transforms_{split}.json"), 'r') as fp:
        meta_jsons[split] = json.load(fp)

modified_meta_jsons = {}
for split in meta_jsons.keys():
    modified_meta_jsons[split] = {}

    for idx, frame in enumerate(meta_jsons[split]["frames"]):
        filename = frame["file_path"].split("/")[-1]+".png"
        
        if idx==0:
            # get height, width
            img = imageio.imread(os.path.join(DATASET_ROOT_DIR, split, filename))
            H, W = img.shape[:2]
            focal = .5 * W / np.tan(.5 * float(meta_jsons[split]['camera_angle_x']))
        
        ### FILL:
        camera_param = {
            "intrinsic": {
                "bounds": [NEAR, FAR],
                "height": H,
                "width": W,
                "focal": focal,
            },
            "extrinsic": {
                "rotation": np.array(frame["transform_matrix"])[:3,:3].tolist(),
                "translation": np.array(frame["transform_matrix"])[:3,-1:].tolist()
            }
        }

        modified_meta_jsons[split][filename] = camera_param

for split in modified_meta_jsons.keys():
    with open(os.path.join(DATASET_ROOT_DIR, f"{split}_camera_params.json"), 'w', encoding='utf-8') as f:
        json.dump(modified_meta_jsons[split], f, ensure_ascii=False, indent=4)
