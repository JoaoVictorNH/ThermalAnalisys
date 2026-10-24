# ==============================================================================
# change_detection.py — Change Detection with Deep Learning (Siamese U-Net)
# ==============================================================================
# Contains the DualTaskLateFusionSiameseUnet architecture, the inference dataset,
# GeoTIFF I/O functions, and the complete inference pipeline.
# Reference: https://doi.org/10.3390/rs15215135
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import rasterio
from rasterio.warp import transform_bounds
from collections import OrderedDict
from pathlib import Path
from tqdm.notebook import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    """Input layer — single double convolution block."""

    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Down layer — MaxPool2d followed by a convolution block."""

    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    """Up layer — TransposeConv2d and concatenation with encoder feature."""

    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Output layer — 1x1 convolution to map to number of classes."""

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """Siamese U-Net encoder."""

    def __init__(self, topology):
        super(Encoder, self).__init__()
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:
        inputs = [x1]
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)
        inputs.reverse()
        return inputs


class Decoder(nn.Module):
    """Siamese U-Net decoder."""

    def __init__(self, topology):
        super(Decoder, self).__init__()
        n_layers = len(topology)
        up_topo = [topology[0]]
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]
            up_topo.append(out_dim)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:
        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)
        return x1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN NETWORK: DualTaskLateFusionSiameseUnet
# ═══════════════════════════════════════════════════════════════════════════════

class DualTaskLateFusionSiameseUnet(nn.Module):
    """
    Siamese U-Net with Late Fusion for Urban Change Detection.

    The network processes image pairs (T1, T2) from Sentinel-1 (SAR) and Sentinel-2
    (optical) to detect changes and perform semantic segmentation.

    Reference: Hafner et al., 2023 (https://doi.org/10.3390/rs15215135)
    """

    def __init__(self):
        super(DualTaskLateFusionSiameseUnet, self).__init__()

        n_classes = 1
        self.topology = [64, 128, 256, 512]
        self.s1_bands = [0, 1]
        self.s2_bands = [2, 1, 0, 3]

        # --- SAR Branches (Sentinel-1) ---
        self.inc_sar = InConv(len(self.s1_bands), self.topology[0], DoubleConv)
        self.encoder_sar = Encoder(self.topology)
        self.decoder_sar_change = Decoder(self.topology)
        self.decoder_sar_sem = Decoder(self.topology)
        self.outc_sar_change = OutConv(self.topology[0], n_classes)
        self.outc_sar_sem = OutConv(self.topology[0], n_classes)

        # --- Optical Branches (Sentinel-2) ---
        self.inc_optical = InConv(len(self.s2_bands), self.topology[0], DoubleConv)
        self.encoder_optical = Encoder(self.topology)
        self.decoder_optical_change = Decoder(self.topology)
        self.decoder_optical_sem = Decoder(self.topology)
        self.outc_optical_change = OutConv(self.topology[0], n_classes)
        self.outc_optical_sem = OutConv(self.topology[0], n_classes)

        # --- Fusion Branches (Late Fusion) ---
        self.outc_fusion_change = OutConv(2 * self.topology[0], n_classes)
        self.outc_fusion_sem = OutConv(2 * self.topology[0], n_classes)

    @staticmethod
    def difference_features(features_t1: torch.Tensor, features_t2: torch.Tensor):
        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        return features_diff

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        # --- SAR Processing ---
        s1_t1, s1_t2 = x_t1[:, :len(self.s1_bands), ], x_t2[:, :len(self.s1_bands), ]
        x1_sar_t1 = self.inc_sar(s1_t1)
        features_sar_t1 = self.encoder_sar(x1_sar_t1)
        x1_sar_t2 = self.inc_sar(s1_t2)
        features_sar_t2 = self.encoder_sar(x1_sar_t2)
        features_sar_diff = self.difference_features(features_sar_t1, features_sar_t2)

        x2_sar_change = self.decoder_sar_change(features_sar_diff)
        out_sar_change = self.outc_sar_change(x2_sar_change)

        x2_sar_sem_t1 = self.decoder_sar_sem(features_sar_t1)
        out_sar_sem_t1 = self.outc_sar_sem(x2_sar_sem_t1)

        x2_sar_sem_t2 = self.decoder_sar_sem(features_sar_t2)
        out_sar_sem_t2 = self.outc_sar_sem(x2_sar_sem_t2)

        # --- Optical Processing ---
        s2_t1, s2_t2 = x_t1[:, len(self.s1_bands):, ], x_t2[:, len(self.s1_bands):, ]
        x1_optical_t1 = self.inc_optical(s2_t1)
        features_optical_t1 = self.encoder_optical(x1_optical_t1)
        x1_optical_t2 = self.inc_optical(s2_t2)
        features_optical_t2 = self.encoder_optical(x1_optical_t2)
        features_optical_diff = self.difference_features(features_optical_t1, features_optical_t2)

        x2_optical_change = self.decoder_optical_change(features_optical_diff)
        out_optical_change = self.outc_optical_change(x2_optical_change)

        x2_optical_sem_t1 = self.decoder_optical_sem(features_optical_t1)
        out_optical_sem_t1 = self.outc_optical_sem(x2_optical_sem_t1)

        x2_optical_sem_t2 = self.decoder_optical_sem(features_optical_t2)
        out_optical_sem_t2 = self.outc_optical_sem(x2_optical_sem_t2)

        # --- Late Fusion ---
        x2_fusion_change = torch.concat((x2_sar_change, x2_optical_change), dim=1)
        out_fusion_change = self.outc_fusion_change(x2_fusion_change)

        x2_fusion_sem_t1 = torch.concat((x2_sar_sem_t1, x2_optical_sem_t1), dim=1)
        out_fusion_sem_t1 = self.outc_fusion_sem(x2_fusion_sem_t1)

        x2_fusion_sem_t2 = torch.concat((x2_sar_sem_t2, x2_optical_sem_t2), dim=1)
        out_fusion_sem_t2 = self.outc_fusion_sem(x2_fusion_sem_t2)

        return out_fusion_change, out_sar_sem_t1, out_sar_sem_t2, out_optical_sem_t1, out_optical_sem_t2,\
            out_fusion_sem_t1, out_fusion_sem_t2


# ═══════════════════════════════════════════════════════════════════════════════
# GeoTIFF I/O FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def read_tif(file):
    """
    Reads a GeoTIFF file and returns the numpy array, affine transform, and CRS.

    Args:
        file (str or Path): Path to the GeoTIFF file.

    Returns:
        tuple: (numpy array (height × width × bands), transform, crs)
    """
    with rasterio.open(file) as dataset:
        arr = dataset.read()
        transform = dataset.transform
        crs = dataset.crs
    return arr.transpose((1, 2, 0)), transform, crs


def write_tif(file, arr, transform, crs):
    """
    Writes a numpy array to a GeoTIFF file.

    Args:
        file (str or Path): Output GeoTIFF file path.
        arr (numpy.ndarray): Numpy array (height × width) or (height × width × bands).
        transform (affine.Affine): Affine transform.
        crs (rasterio.crs.CRS): Coordinate reference system.
    """
    if len(arr.shape) == 3:
        height, width, bands = arr.shape
    else:
        height, width = arr.shape
        bands = 1
        arr = arr[:, :, None]
    with rasterio.open(file, 'w', driver='GTiff', height=height, width=width,
                       count=bands, dtype=arr.dtype, crs=crs,
                       transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)


def load_raster_info(file_path):
    """Loads raster information."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
        n_bands = src.count
    return bounds, crs, transform, shape, n_bands


def bounds_to_wgs84(bounds, crs):
    """Converts bounds to WGS84."""
    bounds_wgs84 = transform_bounds(crs, 'EPSG:4326',
                                     bounds.left, bounds.bottom,
                                     bounds.right, bounds.top)
    return bounds_wgs84


def create_rgb_geotiff_advanced(
    input_path,
    output_path,
    bands=[3, 2, 1],
    scale_factor=10000,
    enhance_contrast=True,
    percentile_min=2,
    percentile_max=98,
    gamma=1.0,
    brightness=0.0
):
    """
    Creates an RGB GeoTIFF with advanced visualization control.
    """
    with rasterio.open(input_path) as src:
        rgb_data = src.read(bands)
        rgb_normalized = rgb_data.astype(np.float32) / scale_factor

        if enhance_contrast:
            p_min = np.nanpercentile(rgb_normalized, percentile_min)
            p_max = np.nanpercentile(rgb_normalized, percentile_max)
            print(f"  📊 Percentiles: {p_min:.4f} ({percentile_min}%) - {p_max:.4f} ({percentile_max}%)")
            rgb_normalized = (rgb_normalized - p_min) / (p_max - p_min)

        if gamma != 1.0:
            rgb_normalized = np.power(rgb_normalized, 1.0 / gamma)
            print(f"  ✿ Gamma applied: {gamma}")

        if brightness != 0.0:
            rgb_normalized = rgb_normalized + brightness
            print(f"  💡 Brightness adjusted: {brightness:+.2f}")

        rgb_normalized = np.clip(rgb_normalized, 0, 1)
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
        print(f"  📊 Result: min={np.min(rgb_uint8)}, max={np.max(rgb_uint8)}, mean={np.mean(rgb_uint8):.1f}")

        meta = src.meta.copy()
        meta.update({
            'count': 3,
            'dtype': 'uint8',
            'nodata': None
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(rgb_uint8)

        print(f"  ✔ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class SceneInferenceDataset(torch.utils.data.Dataset):
    """
    Custom dataset to load and process satellite data
    (Sentinel-1 and Sentinel-2) for inference, splitting the scene into
    tiles with edge padding.
    """

    def __init__(self, s1_t1: np.ndarray, s1_t2: np.ndarray, s2_t1: np.ndarray,
                 s2_t2: np.ndarray, tile_size: int = 128):
        super().__init__()

        self.tile_size = tile_size

        m, n, _ = s1_t1.shape
        self.original_m = m
        self.original_n = n

        pad_m = (tile_size - (m % tile_size)) % tile_size
        pad_n = (tile_size - (n % tile_size)) % tile_size

        self.s1_t1 = np.pad(s1_t1, ((0, pad_m), (0, pad_n), (0, 0)), mode='reflect')
        self.s1_t2 = np.pad(s1_t2, ((0, pad_m), (0, pad_n), (0, 0)), mode='reflect')
        self.s2_t1 = np.pad(s2_t1, ((0, pad_m), (0, pad_n), (0, 0)), mode='reflect')
        self.s2_t2 = np.pad(s2_t2, ((0, pad_m), (0, pad_n), (0, 0)), mode='reflect')

        self.m = m + pad_m
        self.n = n + pad_n

        print(f"  - Padding applied: height +{pad_m}, width +{pad_n}")
        print(f"  - Dimensions after padding: {self.m} x {self.n}")

        self.s1_bands = [0, 1]
        self.s2_bands = [2, 1, 0, 3]

        self.tiles = []
        for i in range(0, self.m, self.tile_size):
            for j in range(0, self.n, self.tile_size):
                tile = {'i': i, 'j': j}
                self.tiles.append(tile)
        self.length = len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        i, j = tile['i'], tile['j']

        tile_s1_t1 = self.s1_t1[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
        tile_s1_t2 = self.s1_t2[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
        tile_s2_t1 = self.s2_t1[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]
        tile_s2_t2 = self.s2_t2[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]

        # Convert each sensor to tensor separately (as in the original code)
        tile_s1_t1, tile_s1_t2 = TF.to_tensor(tile_s1_t1), TF.to_tensor(tile_s1_t2)
        tile_s2_t1, tile_s2_t2 = TF.to_tensor(tile_s2_t1), TF.to_tensor(tile_s2_t2)

        # Concatenate along channel axis (dim=0 in CHW)
        x_t1 = torch.concat((tile_s1_t1, tile_s2_t1), dim=0)
        x_t2 = torch.concat((tile_s1_t2, tile_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def get_arr(self, c=1):
        """Creates zero array with dimensions after padding."""
        if c == 1:
            return np.zeros((self.m, self.n), dtype=np.uint8)
        else:
            return np.zeros((self.m, self.n, c), dtype=np.uint8)

    def get_original_arr(self, c=1):
        """Creates zero array with original dimensions (without padding)."""
        if c == 1:
            return np.zeros((self.original_m, self.original_n), dtype=np.uint8)
        else:
            return np.zeros((self.original_m, self.original_n, c), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model_path: str, device: torch.device):
    """
    Loads the trained model checkpoint.

    Args:
        model_path (str): Full path to the model .pt file.
        device (torch.device): Device (GPU/CPU).

    Returns:
        nn.Module: Neural network with loaded weights.
    """
    net = nn.DataParallel(DualTaskLateFusionSiameseUnet())
    net.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['network'])
    return net


def run_inference(folder_imgs, roi_name, folder_model, model_name='mmcr_train100', tile_size=128):
    """
    Executes the complete Change Detection inference pipeline.

    Args:
        folder_imgs (str): Folder containing GeoTIFFs (sentinel1/2_roi_t1/t2.tif).
        roi_name (str): ROI name (used in file names).
        folder_model (str): Folder containing the trained model.
        model_name (str): Model name (without .pt extension).
        tile_size (int): Tile size for inference.

    Returns:
        str: Path to the generated prediction file (pred_{roi_name}.tif).
    """
    print("\n" + "=" * 80)
    print("🧠 CHANGE DETECTION - INFERENCE WITH SIAMESE U-NET")
    print("=" * 80)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")

    # Load model
    print("\n📥 Loading model...")
    model_path = f'{folder_model}{model_name}.pt'
    net = load_checkpoint(model_path, device)
    print(f"✔ Model loaded: {model_path}")

    # Load data
    print("\n📂 Loading satellite data...")

    def load_and_clean(filepath):
        """Reads GeoTIFF and cleans GEE nodata values (float32 min ≈ -3.4e+38)."""
        arr, tf, cr = read_tif(filepath)
        arr = arr.astype(np.float32)
        # GEE uses float32 min as nodata for pixels outside the scene
        # np.nan_to_num only handles NaN/inf, not extreme finite values
        nodata_mask = np.abs(arr) > 1e10
        arr[nodata_mask] = 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        n_nodata = nodata_mask.sum()
        n_total = arr.size
        if n_nodata > 0:
            pct = 100 * n_nodata / n_total
            print(f"    ⚠️ {n_nodata} nodata pixels ({pct:.1f}%) replaced with 0")
        return arr, tf, cr

    s1_t1_file = f'{folder_imgs}sentinel1_{roi_name}_t1.tif'
    s1_t1, transform, crs = load_and_clean(s1_t1_file)

    s1_t2_file = f'{folder_imgs}sentinel1_{roi_name}_t2.tif'
    s1_t2, *_ = load_and_clean(s1_t2_file)

    s2_t1_file = f'{folder_imgs}sentinel2_{roi_name}_t1.tif'
    s2_t1, *_ = load_and_clean(s2_t1_file)

    s2_t2_file = f'{folder_imgs}sentinel2_{roi_name}_t2.tif'
    s2_t2, *_ = load_and_clean(s2_t2_file)

    print(f"✔ Files loaded successfully!")
    print(f"  - S1 T1: {s1_t1_file} | shape: {s1_t1.shape}")
    print(f"  - S1 T2: {s1_t2_file} | shape: {s1_t2.shape}")
    print(f"  - S2 T1: {s2_t1_file} | shape: {s2_t1.shape}")
    print(f"  - S2 T2: {s2_t2_file} | shape: {s2_t2.shape}")

    # Input value diagnostics
    print(f"\n🔍 Input value diagnostics:")
    print(f"  S1 T1 — min: {s1_t1.min():.4f}, max: {s1_t1.max():.4f}, mean: {s1_t1.mean():.4f}")
    print(f"  S1 T2 — min: {s1_t2.min():.4f}, max: {s1_t2.max():.4f}, mean: {s1_t2.mean():.4f}")
    print(f"  S2 T1 — min: {s2_t1.min():.4f}, max: {s2_t1.max():.4f}, mean: {s2_t1.mean():.4f}")
    print(f"  S2 T2 — min: {s2_t2.min():.4f}, max: {s2_t2.max():.4f}, mean: {s2_t2.mean():.4f}")

    # Create dataset
    print(f"\n📊 Inference configuration:")
    print(f"  - Tile size: {tile_size}x{tile_size}")
    print(f"  - S1 T1 dimensions: {s1_t1.shape}")
    print(f"  - S2 T1 dimensions: {s2_t1.shape}")

    dataset = SceneInferenceDataset(s1_t1, s1_t2, s2_t1, s2_t2, tile_size)
    print(f"  - Total number of tiles: {len(dataset)}")

    # Array with 3 bands: [change, seg_t1, seg_t2]
    pred_padded = dataset.get_arr(3)
    print(f"  - Prediction array dimensions (with padding): {pred_padded.shape}")

    if pred_padded.shape[0] == 0 or pred_padded.shape[1] == 0:
        raise ValueError(f"❌ Error: Prediction array with invalid dimensions: {pred_padded.shape}")

    print("\n✔ Dataset configured successfully!")
    print(f"✔ All {dataset.original_m}x{dataset.original_n} pixels will be processed!")

    # Inference
    print("\n🔄 Starting inference...")
    net.eval()

    for index in tqdm(range(len(dataset)), desc="Processing tiles"):
        tile = dataset.__getitem__(index)
        x_t1 = tile['x_t1'].to(device)
        x_t2 = tile['x_t2'].to(device)

        with torch.no_grad():
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))

        assert(isinstance(logits, tuple))

        # logits[0] = fusion change
        # logits[5] = fusion semantic T1
        # logits[6] = fusion semantic T2
        logits_ch = logits[0]
        logits_sem_t1 = logits[5]
        logits_sem_t2 = logits[6]

        # Apply sigmoid to get probabilities
        y_pred_ch = torch.sigmoid(logits_ch).squeeze().detach().cpu().numpy()
        y_pred_sem_t1 = torch.sigmoid(logits_sem_t1).squeeze().detach().cpu().numpy()
        y_pred_sem_t2 = torch.sigmoid(logits_sem_t2).squeeze().detach().cpu().numpy()

        # Convert to values between 0 and 100
        y_pred_ch = np.clip(y_pred_ch * 100, 0, 100).astype(np.uint8)
        y_pred_sem_t1 = np.clip(y_pred_sem_t1 * 100, 0, 100).astype(np.uint8)
        y_pred_sem_t2 = np.clip(y_pred_sem_t2 * 100, 0, 100).astype(np.uint8)

        i, j = tile['i'], tile['j']
        pred_padded[i:i + tile_size, j:j + tile_size, 0] = y_pred_ch
        pred_padded[i:i + tile_size, j:j + tile_size, 1] = y_pred_sem_t1
        pred_padded[i:i + tile_size, j:j + tile_size, 2] = y_pred_sem_t2

        # Free memory every 10 tiles
        if (index + 1) % 10 == 0:
            torch.cuda.empty_cache()

    print("✔ Inference completed!")

    # Remove padding
    print("\n✂️ Removing padding to original dimensions...")
    pred = pred_padded[:dataset.original_m, :dataset.original_n, :]
    print(f"  - Final dimensions: {pred.shape}")
    print(f"  - Coverage: 100% of original pixels processed! ✅")

    # Verify before saving
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError(f"❌ Error: Prediction array with invalid dimensions: {pred.shape}")

    # Statistics
    print(f"\n📊 Prediction statistics:")
    print(f"  - Change map     — Min: {pred[:,:,0].min()}, Max: {pred[:,:,0].max()}, Mean: {pred[:,:,0].mean():.2f}")
    print(f"  - Segmentation T1 — Min: {pred[:,:,1].min()}, Max: {pred[:,:,1].max()}, Mean: {pred[:,:,1].mean():.2f}")
    print(f"  - Segmentation T2 — Min: {pred[:,:,2].min()}, Max: {pred[:,:,2].max()}, Mean: {pred[:,:,2].mean():.2f}")

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    # Individual files for each map
    # Change map
    output_file = f'{folder_imgs}pred_{roi_name}.tif'
    print(f"\n💾 Saving results...")
    write_tif(output_file, pred[:, :, 0], transform, crs)
    print(f"  ✔ Change map: {output_file}")

    # Segmentation T1
    seg_t1_file = f'{folder_imgs}seg_t1_{roi_name}.tif'
    write_tif(seg_t1_file, pred[:, :, 1], transform, crs)
    print(f"  ✔ Segmentation T1: {seg_t1_file}")

    # Segmentation T2
    seg_t2_file = f'{folder_imgs}seg_t2_{roi_name}.tif'
    write_tif(seg_t2_file, pred[:, :, 2], transform, crs)
    print(f"  ✔ Segmentation T2: {seg_t2_file}")

    print(f"\n✅ Change map saved successfully!")

# ==========================================
    # VISUALIZE RESULTS
    # ==========================================
    try:
        import matplotlib.pyplot as plt

        # Figure 1: S2 T1, S2 T2, Segmentation T1, Segmentation T2
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=3.0)

        for _, ax in np.ndenumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])

        # Sentinel-2 RGB (original dimensions)
        s2_t1_rgb = s2_t1[:dataset.original_m, :dataset.original_n, [2, 1, 0]]
        s2_t2_rgb = s2_t2[:dataset.original_m, :dataset.original_n, [2, 1, 0]]

        axs[0, 0].imshow(np.clip(s2_t1_rgb / 0.4, 0, 1))
        axs[0, 0].set_title('Sentinel-2 T1', fontsize=14)

        axs[0, 1].imshow(np.clip(s2_t2_rgb / 0.4, 0, 1))
        axs[0, 1].set_title('Sentinel-2 T2', fontsize=14)

        axs[1, 0].imshow(pred[:, :, 1], cmap='gray')
        axs[1, 0].set_title('Semantic Segmentation T1', fontsize=14)

        axs[1, 1].imshow(pred[:, :, 2], cmap='gray')
        axs[1, 1].set_title('Semantic Segmentation T2', fontsize=14)

        plt.suptitle(f'Change Detection — {roi_name}', fontsize=16, fontweight='bold')
        plt.show()

        print("\n\n")

        # Figure 2: Change Map
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

        ax2.imshow(pred[:, :, 0], cmap='hot')
        ax2.set_title('Change Map', fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.suptitle(f'Change Detection — {roi_name}', fontsize=16, fontweight='bold')
        plt.show()

    except Exception as e:
        print(f"  ⚠️ Could not generate visualization: {e}")

    return output_file