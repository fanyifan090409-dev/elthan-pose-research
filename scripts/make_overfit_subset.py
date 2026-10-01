import os
import random
import shutil

random.seed(42)

SRC_IMG = "/home/ubuntu/isef/datasets/pose-round3/images/train"
SRC_LBL = "/home/ubuntu/isef/datasets/pose-round3/labels/train"
DST = "/home/ubuntu/isef/datasets/overfit-test"
N = 100

os.makedirs(f"{DST}/images", exist_ok=True)
os.makedirs(f"{DST}/labels", exist_ok=True)

lbls = [f for f in os.listdir(SRC_LBL) if f.endswith(".txt")]
sample = random.sample(lbls, N)
missing = 0
for f in sample:
    shutil.copy(os.path.join(SRC_LBL, f), os.path.join(DST, "labels", f))
    stem = f[:-4]
    found = False
    for ext in (".jpg", ".png", ".jpeg", ".webp", ".JPG"):
        src = os.path.join(SRC_IMG, stem + ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DST, "images", stem + ext))
            found = True
            break
    if not found:
        missing += 1
        print("MISSING", stem)

n_img = len(os.listdir(f"{DST}/images"))
n_lbl = len(os.listdir(f"{DST}/labels"))
print(f"copied {n_lbl} labels, {n_img} images, missing={missing}")
