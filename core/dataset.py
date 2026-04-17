import os
import glob
import re
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

HEIGHT_NORM_CONSTANT = 30.0

def _normalize_core_id(filename):
    """
    Extracts the pure core ID by stripping all known prefixes,
    embedding suffixes, and year suffixes.
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # 1. Strip label prefix
    if base.startswith("label_"):
        base = base[len("label_"):]

    # 2. Strip embedding prefixes
    for prefix in ("gee_emb_", "tessera_emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # 3. Strip trailing embedding suffixes (if any)
    if base.endswith("_embedding"):
        base = base[:-len("_embedding")]
    if base.endswith("_embeddings"):
        base = base[:-len("_embeddings")]

    # 4. Strip trailing year suffixes (e.g., '_2021', '_2023')
    # base = re.sub(r'_\d{4}$', '', base)

    return base


def find_file_pairs(emb_dir, tar_dir):
    """
    Fast and robust O(N) file matching using a hash map and regex normalization.
    Searches recursively and guarantees a match regardless of prefixes/suffixes.
    """
    pairs = []

    # 1. Grab ALL files from the disk exactly ONCE
    emb_files = glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True)
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)

    # 2. Build a fast lookup dictionary for the labels: {normalized_id: full_path}
    label_map = {}
    for l_path in label_files:
        norm_id = _normalize_core_id(l_path)
        label_map[norm_id] = l_path

    # 3. Match embeddings to the lookup dictionary instantly
    for e_path in emb_files:
        norm_id = _normalize_core_id(e_path)

        if norm_id in label_map:
            pairs.append((e_path, label_map[norm_id]))

    return pairs

# ---------------------------------------------------------
# DATASET 1: Pixel-Based (Alpha Earth, Tessera)
# 1:1 Spatial Resolution (e.g., 256x256 -> 256x256)
# ---------------------------------------------------------
class PixelEmbeddingDataset(Dataset):
    def __init__(self, file_pairs, patch_size=128, is_train=True):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(tar_path) as src:
            target = src.read().astype(np.float32)

        image, target = np.nan_to_num(image), np.nan_to_num(target)
        target[3, :, :] = np.clip(target[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        # 1:1 Padding
        c, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape[1], image.shape[2]

        # 1:1 Random Cropping
        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        target = target[:, top:top + self.patch_size, left:left + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target)

# ---------------------------------------------------------
# DATASET 2: Latent Token-Based (TerraMind, Thor)
# Upscaled Spatial Resolution (e.g., 16x16 -> 256x256)
# ---------------------------------------------------------
class LatentTokenDataset(Dataset):
    def __init__(self, file_pairs, patch_size=256, scale_factor=16, is_train=True):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(tar_path) as src:
            target = src.read().astype(np.float32)

        image, target = np.nan_to_num(image), np.nan_to_num(target)
        target[3, :, :] = np.clip(target[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        emb_patch_size = self.patch_size // self.scale_factor

        # Pad Embedding to its specific small size
        c, h_emb, w_emb = image.shape
        if h_emb < emb_patch_size or w_emb < emb_patch_size:
            pad_h = max(0, emb_patch_size - h_emb)
            pad_w = max(0, emb_patch_size - w_emb)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h_emb, w_emb = image.shape[1], image.shape[2]

        # Pad Target to full size
        _, h_tar, w_tar = target.shape
        if h_tar < self.patch_size or w_tar < self.patch_size:
            pad_h = max(0, self.patch_size - h_tar)
            pad_w = max(0, self.patch_size - w_tar)
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        # Multi-scale Cropping
        if self.is_train:
            top_emb = np.random.randint(0, h_emb - emb_patch_size + 1)
            left_emb = np.random.randint(0, w_emb - emb_patch_size + 1)
        else:
            top_emb = (h_emb - emb_patch_size) // 2
            left_emb = (w_emb - emb_patch_size) // 2

        top_tar = top_emb * self.scale_factor
        left_tar = left_emb * self.scale_factor

        image = image[:, top_emb:top_emb + emb_patch_size, left_emb:left_emb + emb_patch_size]
        target = target[:, top_tar:top_tar + self.patch_size, left_tar:left_tar + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target)
