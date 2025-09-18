# -*- coding: utf-8 -*-
# BoT-SORT with DINOv3 appearance features (via timm; HF ID または .pth に対応)
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms as T
import timm

# --- safetensors (任意) ---
try:
    from safetensors.torch import load_file as safe_load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks, remove_duplicate_stracks, sub_stracks
)
from boxmot.utils.matching import (
    embedding_distance, fuse_score, iou_distance, linear_assignment
)


# ---------- DINO feature backend ----------
class DINOReIDBackend:
    """
    get_features(boxes_xyxy, img_bgr) -> np.ndarray (N, D), L2-normalized

    - weights:
        * HF ID 例: "facebook/dinov3-vitl16-pretrain-lvd1689m"
        * ローカル .pth（state_dict 互換; strict=False でロード）
        * 未指定/その他は timm の "vit_large_patch16_224" を使用（タグ無し）
    - device: int (cuda index) | "cuda" | "cpu"
    - half: True で float16 実行（autocast は使わない）
    """
    DEFAULT_TIMM_NAME = "vit_large_patch16_224"  # ← タグ無し（timm 0.9.x 互換）
    DEFAULT_HF_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self, weights: Path | str, device, half: bool = False, batch_size: int = 64):
        # --- device 正規化 ---
        if isinstance(device, int):
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        elif str(device) == "cuda":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu" if device is None else str(device)

        self.batch_size = int(batch_size)
        self.dtype = torch.float16 if (half and self.device != "cpu") else torch.float32

        # --- どのルートで重みを用意するかを決定 ---
        self.model = None
        model_name = self.DEFAULT_TIMM_NAME

        local_sd_path: Path | None = None
        if isinstance(weights, (str, Path)) and str(weights):
            w = str(weights)
            p = Path(w)
            if p.suffix == ".pth" and p.exists():
                # ローカル .pth を timm モデルへ読み込み
                self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
                local_sd_path = p
            elif "dinov3-vitl16" in w or "facebook/dinov3" in w:
                # HF ID の場合：timm の素の ViT-L/16 を作ってから HF の state_dict を上書き
                self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
                try:
                    # 代表的ファイル名を順に試す（safetensors 優先）
                    hf_file = None
                    for fn in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt"):
                        try:
                            hf_file = hf_hub_download(w, filename=fn)
                            break
                        except Exception:
                            hf_file = None

                    if hf_file:
                        # safetensors / bin / pt を判別してロード
                        if hf_file.endswith(".safetensors"):
                            if not _HAS_SAFETENSORS:
                                print("[DINOReID] safetensors 未導入のため HF 上書きをスキップします。")
                                sd = None
                            else:
                                sd = safe_load_file(hf_file)
                        else:
                            sd = torch.load(hf_file, map_location="cpu")

                        if isinstance(sd, dict) and "state_dict" in sd:
                            sd = sd["state_dict"]
                        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
                            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                        if isinstance(sd, dict):
                            missing, unexpected = self.model.load_state_dict(sd, strict=False)
                            print(f"[DINOReID] HF state_dict loaded (strict=False): "
                                  f"missing={len(missing)} unexpected={len(unexpected)}")
                        else:
                            print("[DINOReID] HF から取得した重みの形式が不正のため上書きしませんでした。")
                    else:
                        print("[DINOReID] HF weights file not found; timm base を使用します。")
                except Exception as e:
                    print(f"[DINOReID] HF weights overlay skipped: {e}")
            else:
                # 不明な文字列 → timm の素の ViT-L/16(Imagenet系)を既定使用
                self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        else:
            # 未指定 → timm 既定（素の ViT-L/16）
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        # ローカル .pth 指定だった場合のロード
        if local_sd_path is not None:
            try:
                sd = torch.load(str(local_sd_path), map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                missing, unexpected = self.model.load_state_dict(sd, strict=False)
                print(f"[DINOReID] local .pth loaded (strict=False): "
                      f"missing={len(missing)} unexpected={len(unexpected)}")
            except Exception as e:
                print(f"[DINOReID] local state_dict load skipped: {e}")

        self.model.to(self.device, dtype=self.dtype).eval()
        # timm の ViT-L/16 は通常 embed_dim=1024
        self.feat_dim = int(getattr(self.model, "embed_dim", 1024))

        # ---- 入力前処理（224, bicubic, ImageNet 正規化）----
        img_size = 224
        try:
            # 可能なら HF の preprocessor_config / config から image_size を拾う
            for fn in ("preprocessor_config.json", "config.json"):
                try:
                    cfg_path = hf_hub_download(self.DEFAULT_HF_ID, filename=fn)
                    with open(cfg_path, "r") as f:
                        j = json.load(f)
                    if isinstance(j, dict) and "image_size" in j:
                        img_size = int(j["image_size"])
                        break
                except Exception:
                    pass
        except Exception:
            pass

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def get_features(self, boxes_xyxy: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        """
        boxes_xyxy: (N,4) [l,t,r,b]
        img_bgr: HxWx3 uint8 (OpenCV BGR)
        returns: (N, D) float32, L2-normalized
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        H, W = img_bgr.shape[:2]
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H - 1)
        boxes = boxes.astype(np.int32)

        # 切り出し & RGB PIL 化
        patches: List[Image.Image] = []
        for (l, t, r, b) in boxes:
            if r <= l or b <= t:  # 変な矩形の救済
                l, t = max(0, l), max(0, t)
                r, b = min(W - 1, max(r, l + 1)), min(H - 1, max(b, t + 1))
            crop = img_bgr[t:b, l:r, :]
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            patches.append(Image.fromarray(rgb))

        # バッチ推論
        vecs = []
        i = 0
        while i < len(patches):
            chunk = patches[i:i + self.batch_size]
            tensor_list = [self.transform(p.convert("RGB")) for p in chunk]
            pixel_values = torch.stack(tensor_list, dim=0).to(self.device, dtype=self.dtype)

            # timm: forward_features から CLS 相当を取得
            feats_chunk = self.model.forward_features(pixel_values)
            if isinstance(feats_chunk, dict):
                x = (feats_chunk.get("x_norm_clstoken")
                     or feats_chunk.get("x")
                     or feats_chunk.get("last_hidden_state"))
                if x is None:
                    raise RuntimeError("Unsupported timm forward_features output dict keys")
                if x.ndim == 3:
                    cls = x[:, 0, :].float()
                else:
                    cls = x.float()
            else:
                # Tensor 想定 [B, tokens, D] なら 先頭を CLS とみなす
                cls = feats_chunk[:, 0, :].float()

            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vecs.append(cls.cpu().numpy())
            i += self.batch_size

        feats = np.concatenate(vecs, axis=0).astype(np.float32)
        return feats


# ---------- BoT-SORT (DINO features) ----------
class BotSort_dino(BaseTracker):
    """
    BoTSORT Tracker with DINOv3 appearance features.

    Args:
        reid_weights (Path|str): DINOv3 weights (HF ID or .pth or timm 既定)
        device (torch.device|int|str): 0/'cuda:0'/cpu
        half (bool): use float16 (no autocast)
        per_class, track_high_thresh, ...: 元実装同様
    """

    def __init__(
        self,
        reid_weights: Path | str,
        device,
        half: bool,
        per_class: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.35,  # DINO: CLIPよりやや緩め推奨
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID (DINO)
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            self.model = DINOReIDBackend(reid_weights, device, half)
            self.feat_dim = self.model.feat_dim
        else:
            self.model = None
            self.feat_dim = None

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None

        # Extract appearance features
        if self.with_reid and embs is None and len(dets_first) > 0:
            features_high = self.model.get_features(dets_first[:, 0:4], img)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
        else:
            detections = []

        # Separate tracks
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association
        STrack.multi_predict(strack_pool)
        warp = self.cmc.apply(img, dets)  # camera motion compensation
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track_first, u_det_first = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # Second association
        if len(dets_second) > 0:
            detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]
        else:
            detections_second = []

        r_tracked = [strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked]
        dists2 = iou_distance(r_tracked, detections_second)
        matches2, u_track_second, u_det_second = linear_assignment(dists2, thresh=0.5)

        for itracked, idet in matches2:
            track = r_tracked[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track_second:
            track = r_tracked[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Handle unconfirmed
        detections_unc = [detections[i] for i in u_det_first]
        ious_unc = iou_distance(unconfirmed, detections_unc)
        ious_unc_mask = ious_unc > self.proximity_thresh
        ious_unc = fuse_score(ious_unc, detections_unc)

        if self.with_reid:
            emb_unc = embedding_distance(unconfirmed, detections_unc) / 2.0
            emb_unc[emb_unc > self.appearance_thresh] = 1.0
            emb_unc[ious_unc_mask] = 1.0
            d_unc = np.minimum(ious_unc, emb_unc)
        else:
            d_unc = ious_unc

        matches_unc, u_unconfirmed, u_det_unc = linear_assignment(d_unc, thresh=0.7)

        for itracked, idet in matches_unc:
            unconfirmed[itracked].update(detections_unc[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Initialize new tracks
        for inew in u_det_unc:
            track = detections_unc[inew]
            if track.conf < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)

        # Update states
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Prepare output
        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(self.active_tracks, self.lost_stracks)

        outputs = [
            [*t.xyxy, t.id, t.cls, t.conf, t.det_ind]
            for t in self.active_tracks
            if t.is_activated
        ]
        return np.asarray(outputs)
