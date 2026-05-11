import os
import numpy as np
from PIL import Image


def load_image_float(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return img


def save_image_uint8(img, path):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def resize_mask_nearest(mask, target_hw):
    target_h, target_w = target_hw
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
    return np.array(mask_img).astype(np.float32) / 255.0


def tensor_to_numpy_2d(x):
    """
    Accept torch tensor / numpy array with shape:
    [H, W], [1, H, W], [H, W, 1]
    """
    if x is None:
        return None

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[-1] == 1:
            x = x[..., 0]
        else:
            x = x.mean(axis=0)

    return x.astype(np.float32)


def compute_confidence_mask_and_diff(
    img_a_path,
    img_b_path,
    mask_threshold=0.1,
    opacity=None,
    depth=None,
    opacity_threshold=0.3,
    use_depth_valid=True,
    save_dir=None,
    prefix="pseudo",
):
    """
    Final mask = RGB bidirectional consistency mask
               * Gaussian opacity validity mask
               * optional depth validity mask
    """
    img_a = load_image_float(img_a_path)
    img_b = load_image_float(img_b_path)

    diff = np.abs(img_a - img_b).mean(axis=2)
    rgb_mask = (diff < mask_threshold).astype(np.float32)

    final_mask = rgb_mask.copy()

    opacity_mask = None
    if opacity is not None:
        opacity_np = tensor_to_numpy_2d(opacity)
        opacity_mask = (opacity_np > opacity_threshold).astype(np.float32)

        if opacity_mask.shape != final_mask.shape:
            opacity_mask = resize_mask_nearest(opacity_mask, final_mask.shape)

        final_mask = final_mask * opacity_mask

    depth_mask = None
    if use_depth_valid and depth is not None:
        depth_np = tensor_to_numpy_2d(depth)
        depth_mask = ((depth_np > 0) & np.isfinite(depth_np)).astype(np.float32)

        if depth_mask.shape != final_mask.shape:
            depth_mask = resize_mask_nearest(depth_mask, final_mask.shape)

        final_mask = final_mask * depth_mask

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # RGB consistency mask
        save_image_uint8(
            np.stack([rgb_mask] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_rgb_mask.png"),
        )

        # Gaussian opacity mask
        if opacity_mask is not None:
            save_image_uint8(
                np.stack([opacity_mask] * 3, axis=2),
                os.path.join(save_dir, f"{prefix}_opacity_mask.png"),
            )

        # Depth valid mask
        if depth_mask is not None:
            save_image_uint8(
                np.stack([depth_mask] * 3, axis=2),
                os.path.join(save_dir, f"{prefix}_depth_mask.png"),
            )

        # Final confidence mask
        save_image_uint8(
            np.stack([final_mask] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_final_mask.png"),
        )

        # Diff visualization
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        save_image_uint8(
            np.stack([diff_norm] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_diff.png"),
        )

    return final_mask, diff

def fuse_pseudo_views(
    refined_prev_path,
    refined_curr_path,
    coarse_path,
    confidence_mask,
    save_path=None,
):
    """
    Mask-guided fusion.

    High-confidence regions:
        average bidirectional Difix outputs.
    Low-confidence regions:
        fall back to coarse 3DGS render.
    """
    import numpy as np
    from PIL import Image
    import os

    prev = np.array(Image.open(refined_prev_path).convert("RGB")).astype(np.float32) / 255.0
    curr = np.array(Image.open(refined_curr_path).convert("RGB")).astype(np.float32) / 255.0
    coarse = np.array(Image.open(coarse_path).convert("RGB")).astype(np.float32) / 255.0

    h, w = prev.shape[:2]

    if curr.shape[:2] != (h, w):
        curr = np.array(
            Image.open(refined_curr_path).convert("RGB").resize((w, h))
        ).astype(np.float32) / 255.0

    if coarse.shape[:2] != (h, w):
        coarse = np.array(
            Image.open(coarse_path).convert("RGB").resize((w, h))
        ).astype(np.float32) / 255.0

    mask = confidence_mask.astype(np.float32)
    if mask.shape != (h, w):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask = np.array(mask_img).astype(np.float32) / 255.0

    mask_3 = np.stack([mask] * 3, axis=2)

    refined_avg = 0.5 * (prev + curr)
    fused = mask_3 * refined_avg + (1.0 - mask_3) * coarse
    fused = np.clip(fused, 0.0, 1.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray((fused * 255).astype(np.uint8)).save(save_path)

    return fused

import cv2
import torch
from dust3r.inference import inference
from mast3r.fast_nn import fast_reciprocal_NNs
from utils.init_pose import torch_images_to_dust3r_format


def pil_path_to_tensor_image(path, device="cuda"):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).to(device)


@torch.no_grad()
def mast3r_mutual_matches_from_paths(
    img_a_path,
    img_b_path,
    matcher,
    size=512,
    subsample=8,
    device="cuda",
):
    """
    Return mutual MASt3R matches from image A to image B.

    Returns:
        pts_a_orig: Nx2 coordinates in original image A resolution
        pts_b_orig: Nx2 coordinates in original image B resolution
        score_map_a: sparse match mask in original A resolution
    """
    img_a = pil_path_to_tensor_image(img_a_path, device=device)
    img_b = pil_path_to_tensor_image(img_b_path, device=device)

    H0, W0 = img_a.shape[1], img_a.shape[2]

    images = torch_images_to_dust3r_format([img_a, img_b], size=size)
    output = inference([tuple(images)], matcher, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()

    matches_a, matches_b = fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=subsample,
        device=device,
        dist="dot",
        block_size=2**13,
    )

    Hm = view1["img"].shape[2]
    Wm = view1["img"].shape[3]

    # map MASt3R resized/cropped coordinates back to original image space.
    scale_x = W0 / float(Wm)
    scale_y = H0 / float(Hm)

    pts_a_orig = matches_a.astype(np.float32).copy()
    pts_b_orig = matches_b.astype(np.float32).copy()
    pts_a_orig[:, 0] *= scale_x
    pts_a_orig[:, 1] *= scale_y
    pts_b_orig[:, 0] *= scale_x
    pts_b_orig[:, 1] *= scale_y

    return pts_a_orig, pts_b_orig


def points_to_soft_mask(points_xy, h, w, radius=2):
    mask = np.zeros((h, w), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return mask

    pts = np.round(points_xy).astype(np.int32)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    pts = pts[valid]

    for x, y in pts:
        cv2.circle(mask, (int(x), int(y)), radius, 1.0, -1)

    return mask

def _safe_numpy_depth(depth, target_hw=None):
    d = tensor_to_numpy_2d(depth)
    if d is None:
        return None
    if target_hw is not None and d.shape != target_hw:
        d = resize_mask_nearest(d, target_hw)
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    return d


def _get_intrinsics_from_camera(cam):
    """
    Return fx, fy, cx, cy from S3PO-GS Camera.
    Uses FoV if explicit fx/fy are unavailable.
    """
    H = int(cam.image_height)
    W = int(cam.image_width)

    if hasattr(cam, "fx") and hasattr(cam, "fy"):
        fx, fy = float(cam.fx), float(cam.fy)
    else:
        import math
        fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
        fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))

    if hasattr(cam, "cx") and hasattr(cam, "cy"):
        cx, cy = float(cam.cx), float(cam.cy)
    else:
        cx, cy = W * 0.5, H * 0.5

    return fx, fy, cx, cy, H, W


def _world_to_camera_matrix_np(cam):
    """
    S3PO-GS Camera uses R/T as world-to-camera parameters.
    x_cam = R^T x_world + T is commonly used in the original 3DGS convention.
    This helper follows that convention.
    """
    R = cam.R.detach().cpu().numpy().astype(np.float32)
    T = cam.T.detach().cpu().numpy().astype(np.float32)

    Twc = np.eye(4, dtype=np.float32)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = T
    return Twc


def _project_depth_a_to_b(cam_a, depth_a, cam_b, depth_b=None, depth_tol=0.15, eps=1e-6):
    """
    BRPO-style reprojection overlap from view a to b.

    For each pixel p_a with depth d_a:
        X_a = Pi^{-1}(p_a, d_a)
        X_w = T_a^{-1} X_a
        X_b = T_b X_w
        p_b = Pi(X_b)

    Returns:
        overlap_mask: valid reprojection and optional depth-consistency mask
        depth_score: Eq.5-like depth consistency score
    """
    fx_a, fy_a, cx_a, cy_a, Ha, Wa = _get_intrinsics_from_camera(cam_a)
    fx_b, fy_b, cx_b, cy_b, Hb, Wb = _get_intrinsics_from_camera(cam_b)

    if depth_a.shape != (Ha, Wa):
        depth_a = cv2.resize(depth_a, (Wa, Ha), interpolation=cv2.INTER_NEAREST)

    if depth_b is not None and depth_b.shape != (Hb, Wb):
        depth_b = cv2.resize(depth_b, (Wb, Hb), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.meshgrid(
        np.arange(Ha, dtype=np.float32),
        np.arange(Wa, dtype=np.float32),
        indexing="ij",
    )

    za = depth_a.astype(np.float32)
    valid_a = (za > eps) & np.isfinite(za)

    xa = (xs - cx_a) / fx_a * za
    ya = (ys - cy_a) / fy_a * za

    pts_a = np.stack([xa, ya, za, np.ones_like(za)], axis=-1).reshape(-1, 4).T

    Ta = _world_to_camera_matrix_np(cam_a)
    Tb = _world_to_camera_matrix_np(cam_b)
    Tba = Tb @ np.linalg.inv(Ta)

    pts_b = Tba @ pts_a
    xb, yb, zb = pts_b[0], pts_b[1], pts_b[2]

    ub = fx_b * (xb / (zb + eps)) + cx_b
    vb = fy_b * (yb / (zb + eps)) + cy_b

    ub_i = np.round(ub).astype(np.int32)
    vb_i = np.round(vb).astype(np.int32)

    inside = (
        (zb > eps)
        & (ub_i >= 0) & (ub_i < Wb)
        & (vb_i >= 0) & (vb_i < Hb)
        & valid_a.reshape(-1)
    )

    overlap = np.zeros((Ha * Wa,), dtype=np.float32)
    depth_score = np.zeros((Ha * Wa,), dtype=np.float32)

    if depth_b is None:
        overlap[inside] = 1.0
        depth_score[inside] = 1.0
    else:
        db_sample = np.zeros_like(zb, dtype=np.float32)
        db_sample[inside] = depth_b[vb_i[inside], ub_i[inside]]

        valid_b = db_sample > eps

        rel_depth_err = np.abs(zb - db_sample) / ((zb + db_sample) * 0.5 + eps)
        score = np.exp(-rel_depth_err)

        consistent = inside & valid_b & (rel_depth_err < depth_tol)
        overlap[consistent] = 1.0
        depth_score[consistent] = score[consistent]

    return overlap.reshape(Ha, Wa), depth_score.reshape(Ha, Wa)


def _pose_translation_score(cam_a, cam_b):
    Ta = cam_a.T.detach().cpu().numpy().astype(np.float32)
    Tb = cam_b.T.detach().cpu().numpy().astype(np.float32)
    return float(np.exp(-np.linalg.norm(Ta - Tb)))


def compute_reprojection_overlap_score(
    pseudo_cam,
    prev_cam,
    curr_cam,
    pseudo_depth,
    prev_depth=None,
    curr_depth=None,
    depth_tol=0.15,
):
    """
    Implements BRPO Eq.3-Eq.6 approximately:
        C_t,i(p) = s_d(p) * s_t

    Returns:
        c_prev, c_curr: per-pixel overlap confidence maps in pseudo-view space.
    """
    dp = _safe_numpy_depth(pseudo_depth)
    if dp is None:
        return None, None

    target_hw = dp.shape
    d_prev = _safe_numpy_depth(prev_depth, target_hw=None)
    d_curr = _safe_numpy_depth(curr_depth, target_hw=None)

    _, score_prev = _project_depth_a_to_b(
        cam_a=pseudo_cam,
        depth_a=dp,
        cam_b=prev_cam,
        depth_b=d_prev,
        depth_tol=depth_tol,
    )
    _, score_curr = _project_depth_a_to_b(
        cam_a=pseudo_cam,
        depth_a=dp,
        cam_b=curr_cam,
        depth_b=d_curr,
        depth_tol=depth_tol,
    )

    score_prev *= _pose_translation_score(pseudo_cam, prev_cam)
    score_curr *= _pose_translation_score(pseudo_cam, curr_cam)

    score_prev = np.clip(score_prev, 0.0, 1.0).astype(np.float32)
    score_curr = np.clip(score_curr, 0.0, 1.0).astype(np.float32)

    return score_prev, score_curr


def _dense_match_score_from_points(points_xy, h, w, radius=2):
    """
    Converts sparse mutual NN matches into a smoother correspondence evidence map.
    This keeps BRPO Eq.9 semantics but reduces excessive sparsity.
    """
    sparse = points_to_soft_mask(points_xy, h, w, radius=radius)

    if sparse.max() <= 0:
        return sparse

    # Smooth and normalize. This is safer than using a huge hard dilation radius.
    k = max(3, radius * 4 + 1)
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(sparse, (k, k), sigmaX=max(1.0, radius))
    soft = soft / (soft.max() + 1e-6)

    return soft.astype(np.float32)

def save_confidence_overlay(image_path, confidence, output_path, alpha=0.45):
    img = load_image_float(image_path)
    h, w = img.shape[:2]

    mask = confidence.astype(np.float32)
    if mask.shape != (h, w):
        mask = resize_mask_nearest(mask, (h, w))

    overlay = img.copy()

    # confidence = 1.0: green, confidence = 0.5: yellow, confidence = 0: red
    color = np.zeros_like(img)
    color[mask >= 0.99] = [0.0, 1.0, 0.0]
    color[(mask > 0.0) & (mask < 0.99)] = [1.0, 1.0, 0.0]
    color[mask <= 0.0] = [1.0, 0.0, 0.0]

    out = (1.0 - alpha) * overlay + alpha * color
    save_image_uint8(out, output_path)
    return output_path

def compute_brpo_confidence_mask(
    pseudo_path,
    prev_ref_path,
    curr_ref_path,
    matcher,
    opacity=None,
    depth=None,
    opacity_threshold=0.3,
    use_depth_valid=True,
    point_radius=2,
    size=512,
    subsample=8,
    save_dir=None,
    prefix="pseudo",
    device="cuda",

    # New BRPO-style geometric inputs.
    pseudo_cam=None,
    prev_cam=None,
    curr_cam=None,
    prev_depth=None,
    curr_depth=None,

    # New controls.
    use_reprojection=True,
    depth_tol=0.15,
    corr_threshold=0.15,
    overlap_threshold=0.05,
):
    """
    More complete BRPO-style confidence mask.

    Compared with the previous minimum version:
      1. Uses soft dense correspondence evidence instead of only sparse point dilation.
      2. Adds reprojection/depth consistency score following BRPO Eq.3-Eq.6.
      3. Keeps Eq.9 multi-level confidence:
           1.0: both neighboring real frames support the pseudo pixel
           0.5: only one neighboring real frame supports it
           0.0: unsupported / likely hallucinated
    """
    pseudo_img = load_image_float(pseudo_path)
    h, w = pseudo_img.shape[:2]

    pts_p_prev, _ = mast3r_mutual_matches_from_paths(
        pseudo_path,
        prev_ref_path,
        matcher=matcher,
        size=size,
        subsample=subsample,
        device=device,
    )

    pts_p_curr, _ = mast3r_mutual_matches_from_paths(
        pseudo_path,
        curr_ref_path,
        matcher=matcher,
        size=size,
        subsample=subsample,
        device=device,
    )

    # 1) Feature mapping evidence, smoother than hard sparse points.
    corr_prev = _dense_match_score_from_points(
        pts_p_prev, h, w, radius=point_radius
    )
    corr_curr = _dense_match_score_from_points(
        pts_p_curr, h, w, radius=point_radius
    )

    # 2) Optional BRPO Eq.3-Eq.6 reprojection/depth overlap score.
    reproj_prev = np.ones((h, w), dtype=np.float32)
    reproj_curr = np.ones((h, w), dtype=np.float32)

    if (
        use_reprojection
        and pseudo_cam is not None
        and prev_cam is not None
        and curr_cam is not None
        and depth is not None
    ):
        reproj_prev, reproj_curr = compute_reprojection_overlap_score(
            pseudo_cam=pseudo_cam,
            prev_cam=prev_cam,
            curr_cam=curr_cam,
            pseudo_depth=depth,
            prev_depth=prev_depth,
            curr_depth=curr_depth,
            depth_tol=depth_tol,
        )

        if reproj_prev is None or reproj_curr is None:
            reproj_prev = np.ones((h, w), dtype=np.float32)
            reproj_curr = np.ones((h, w), dtype=np.float32)
        else:
            if reproj_prev.shape != (h, w):
                reproj_prev = cv2.resize(reproj_prev, (w, h), interpolation=cv2.INTER_LINEAR)
            if reproj_curr.shape != (h, w):
                reproj_curr = cv2.resize(reproj_curr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Combined support: feature evidence * geometric overlap evidence.
    support_prev = corr_prev * reproj_prev
    support_curr = corr_curr * reproj_curr

    prev_supported = (support_prev > corr_threshold) & (reproj_prev > overlap_threshold)
    curr_supported = (support_curr > corr_threshold) & (reproj_curr > overlap_threshold)

    both = prev_supported & curr_supported
    one = np.logical_xor(prev_supported, curr_supported)

    confidence = np.zeros((h, w), dtype=np.float32)
    confidence[both] = 1.0
    confidence[one] = 0.5

    # 3) Gaussian visibility filtering.
    opacity_mask = None
    if opacity is not None:
        opacity_np = tensor_to_numpy_2d(opacity)
        opacity_mask = (opacity_np > opacity_threshold).astype(np.float32)
        if opacity_mask.shape != confidence.shape:
            opacity_mask = resize_mask_nearest(opacity_mask, confidence.shape)
        confidence *= opacity_mask

    depth_mask = None
    if use_depth_valid and depth is not None:
        depth_np = tensor_to_numpy_2d(depth)
        depth_mask = ((depth_np > 0) & np.isfinite(depth_np)).astype(np.float32)
        if depth_mask.shape != confidence.shape:
            depth_mask = resize_mask_nearest(depth_mask, confidence.shape)
        confidence *= depth_mask

    stats = {
        "prev_ratio": float(prev_supported.mean()),
        "curr_ratio": float(curr_supported.mean()),
        "full_ratio": float((confidence == 1.0).mean()),
        "half_ratio": float((confidence == 0.5).mean()),
        "valid_ratio": float((confidence > 0).mean()),
        "corr_prev_mean": float(corr_prev.mean()),
        "corr_curr_mean": float(corr_curr.mean()),
        "reproj_prev_mean": float(reproj_prev.mean()),
        "reproj_curr_mean": float(reproj_curr.mean()),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        save_image_uint8(
            np.stack([corr_prev] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_corr_prev.png"),
        )
        save_image_uint8(
            np.stack([corr_curr] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_corr_curr.png"),
        )
        save_image_uint8(
            np.stack([reproj_prev] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_reproj_prev.png"),
        )
        save_image_uint8(
            np.stack([reproj_curr] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_reproj_curr.png"),
        )
        save_image_uint8(
            np.stack([support_prev] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_support_prev.png"),
        )
        save_image_uint8(
            np.stack([support_curr] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_support_curr.png"),
        )
        save_image_uint8(
            np.stack([confidence] * 3, axis=2),
            os.path.join(save_dir, f"{prefix}_final_mask.png"),
        )

        vis = np.zeros((h, w, 3), dtype=np.float32)
        vis[confidence == 1.0] = [1.0, 1.0, 1.0]
        vis[confidence == 0.5] = [0.5, 0.5, 0.5]
        save_image_uint8(vis, os.path.join(save_dir, f"{prefix}_brpo_mask_vis.png"))
        save_confidence_overlay(
            image_path=pseudo_path,
            confidence=confidence,
            output_path=os.path.join(save_dir, f"{prefix}_mask_overlay.png"),
        )

    return confidence, stats