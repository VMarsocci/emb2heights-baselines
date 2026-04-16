import os
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import PixelEmbeddingDataset, LatentTokenDataset, find_file_pairs, _normalize_core_id, \
    HEIGHT_NORM_CONSTANT

# --- DEFAULTS ---
EXPERIMENT_NAME = "terramind_decoder_run01"
BASE_DIR = "./runs"
TEST_EMBEDDINGS_DIR = ""
TEST_TARGETS_DIR = ""
MODEL_TYPE = "decoder_residual"
PATCH_SIZE = 256
MAX_SAMPLES = 0

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run inference, saving predictions as .npy files."
    )
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--base-dir", type=str, default=BASE_DIR,
                        help="Root directory containing experiment subfolders.")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE,
                        choices=["auto", "lightunet", "decoder", "decoder_residual"],
                        help="Model architecture used during training.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the .pth checkpoint. Defaults to <base-dir>/<experiment-name>/model_best.pth.")
    parser.add_argument("--test-embeddings-dir", type=str, required=True,
                        help="Directory containing embedding .tif files.")
    parser.add_argument("--test-targets-dir", type=str, required=True,
                        help="Directory containing label .tif files (used only for file pairing).")
    parser.add_argument("--predictions-dir", type=str, default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Limit inference to N samples (0 = all).")
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")

    os.makedirs(predictions_dir, exist_ok=True)

    # --- Load data pairs ---
    print(f"Loading file pairs from embeddings: {args.test_embeddings_dir}")
    pairs = find_file_pairs(args.test_embeddings_dir, args.test_targets_dir)
    if not pairs:
        raise RuntimeError("No matching file pairs found. Check --test-embeddings-dir and --test-targets-dir.")
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    is_lightunet = args.model_type.lower() == "lightunet"
    if is_lightunet:
        test_ds = PixelEmbeddingDataset(pairs, patch_size=args.patch_size, is_train=False)
    else:
        test_ds = LatentTokenDataset(pairs, patch_size=args.patch_size, scale_factor=16, is_train=False)

    # --- Load model ---
    sample_img, _ = test_ds[0]
    model, selected_model = build_model(args.model_type, n_channels=sample_img.shape[0], n_classes=4)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {selected_model} from {model_path} (input channels={sample_img.shape[0]})")

    # --- Run inference ---
    print(f"Running inference on {len(pairs)} samples...")
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Predicting"):
            img_tensor, _ = test_ds[i]
            img_batch = img_tensor.unsqueeze(0).to(DEVICE)

            output_batch = model(img_batch)
            pred_np = output_batch.squeeze().cpu().numpy().astype(np.float32)

            # Denormalize height channel: model output [0,1] -> physical meters
            pred_np[3] = pred_np[3] * HEIGHT_NORM_CONSTANT

            emb_path, _ = test_ds.file_pairs[i]
            core_id = _normalize_core_id(emb_path)

            save_path = os.path.join(predictions_dir, f"{core_id}.npy")
            np.save(save_path, pred_np)

    print(f"Predictions saved to: {predictions_dir}")
    print(f"Output shape per file: {pred_np.shape}  [building%, veg%, water%, height_m]")


if __name__ == "__main__":
    main()
