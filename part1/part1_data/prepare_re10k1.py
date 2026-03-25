import json
import shutil
from pathlib import Path
from PIL import Image

SRC_ROOT = Path.home() / "CV_Project" / "dataset" / "Re10k-1"
DST_ROOT = Path.home() / "CV_Project" / "part1" / "part1_data" / "re10k_1"

SRC_IMAGES = SRC_ROOT / "images"
SRC_CAMERAS = SRC_ROOT / "cameras.json"
SRC_INTRINSICS = SRC_ROOT / "intrinsics.json"

DST_IMAGES = DST_ROOT / "images"
DST_META = DST_ROOT / "meta"
DST_GT_COLMAP = DST_ROOT / "gt_colmap_text"
DST_PLAN_A = DST_ROOT / "planA_colmap"
DST_PLAN_B = DST_ROOT / "planB_vggt"
DST_EVAL = DST_ROOT / "eval"


def log(msg):
    print(f"[prepare_re10k1] {msg}", flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    log(f"ensured dir: {path}")


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main():
    log("start")

    log(f"SRC_ROOT = {SRC_ROOT}")
    log(f"DST_ROOT = {DST_ROOT}")

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SRC_ROOT}")
    if not SRC_IMAGES.exists():
        raise FileNotFoundError(f"Images folder not found: {SRC_IMAGES}")
    if not SRC_CAMERAS.exists():
        raise FileNotFoundError(f"cameras.json not found: {SRC_CAMERAS}")
    if not SRC_INTRINSICS.exists():
        raise FileNotFoundError(f"intrinsics.json not found: {SRC_INTRINSICS}")

    log("source files verified")

    ensure_dir(DST_ROOT)
    ensure_dir(DST_IMAGES)
    ensure_dir(DST_META)
    ensure_dir(DST_GT_COLMAP)
    ensure_dir(DST_PLAN_A)
    ensure_dir(DST_PLAN_B)
    ensure_dir(DST_EVAL)

    log("loading json")
    with open(SRC_CAMERAS, "r") as f:
        cameras_data = json.load(f)

    with open(SRC_INTRINSICS, "r") as f:
        intrinsics_data = json.load(f)

    log(f"loaded cameras.json entries: {len(cameras_data)}")

    fx_n = intrinsics_data["fx"]
    fy_n = intrinsics_data["fy"]
    cx_n = intrinsics_data["cx"]
    cy_n = intrinsics_data["cy"]

    first_img_name = cameras_data[0]["image_name"]
    first_img_path = SRC_IMAGES / first_img_name
    if not first_img_path.exists():
        raise FileNotFoundError(f"First image not found: {first_img_path}")

    width, height = Image.open(first_img_path).size
    log(f"image size = {width} x {height}")

    fx = fx_n * width
    fy = fy_n * height
    cx = cx_n * width
    cy = cy_n * height
    log(f"pixel intrinsics = fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    shutil.copy2(SRC_CAMERAS, DST_META / "cameras.json")
    shutil.copy2(SRC_INTRINSICS, DST_META / "intrinsics.json")
    log("copied json metadata")

    image_names = []
    for i, entry in enumerate(cameras_data, start=1):
        img_name = entry["image_name"]
        src_img = SRC_IMAGES / img_name
        dst_img = DST_IMAGES / img_name

        if not src_img.exists():
            raise FileNotFoundError(f"Image listed in cameras.json not found: {src_img}")

        safe_symlink(src_img, dst_img)
        image_names.append(img_name)

        if i <= 5 or i % 50 == 0 or i == len(cameras_data):
            log(f"linked images: {i}/{len(cameras_data)}")

    with open(DST_META / "image_list.txt", "w") as f:
        for name in image_names:
            f.write(name + "\n")
    log("wrote image_list.txt")

    cameras_txt = DST_GT_COLMAP / "cameras.txt"
    images_txt = DST_GT_COLMAP / "images.txt"
    points3D_txt = DST_GT_COLMAP / "points3D.txt"

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE {} {} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
            width, height, fx, fy, cx, cy
        ))
    log("wrote cameras.txt")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, entry in enumerate(cameras_data, start=1):
            img_name = entry["image_name"]
            qx, qy, qz, qw = entry["cam_quat"]
            tx, ty, tz = entry["cam_trans"]

            f.write(
                f"{idx} "
                f"{qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                f"{tx:.12f} {ty:.12f} {tz:.12f} "
                f"1 {img_name}\n"
            )
            f.write("\n")

            if idx <= 5 or idx % 50 == 0 or idx == len(cameras_data):
                log(f"wrote image poses: {idx}/{len(cameras_data)}")

    with open(points3D_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    log("wrote empty points3D.txt")

    with open(DST_ROOT / "prepare_summary.txt", "w") as f:
        f.write(f"SRC_ROOT: {SRC_ROOT}\n")
        f.write(f"DST_ROOT: {DST_ROOT}\n")
        f.write(f"Num images: {len(cameras_data)}\n")
        f.write(f"Image size: {width} x {height}\n")
        f.write(f"Pixel intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}\n")
    log("wrote prepare_summary.txt")

    log("done")


if __name__ == "__main__":
    main()