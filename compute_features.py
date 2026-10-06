#!/usr/bin/env python3
"""
Independent recomputation of nine three-dimensional CT features.

Reproduces the feature definitions used in the manuscript's reproducibility
substudy using only SimpleITK (no commercial software output). Features are
computed on a binary ROI mask after (i) nearest-neighbour resampling onto the
CT grid and (ii) one-voxel erosion with a ball-shaped structuring element
(radius 1 voxel), matching Figure S7.

Features (N = number of voxels in the eroded mask M):
  Volume       = N * voxel_volume
  Surface area = physical surface area of M (SimpleITK LabelShapeStatistics)
  Sphericity   = (36 * pi * Volume^2)^(1/3) / Surface_area
  CT mean      = (1/N) * sum HU(x)
  CT median    = median{ HU(x) }
  CT max       = max{ HU(x) }
  CT min       = min{ HU(x) }
  Entropy      = -sum_k p_k * log2(p_k),  p_k from fixed 25-HU bins whose
                 edges start at the minimum HU value in M
  PSC          = #{ x in M : HU(x) > -300 } / N

Usage:
  python compute_features.py --dicom <series_dir> --mask <roi.nii.gz> \
      --out features.json [--no-erosion]
"""

import argparse
import json

import numpy as np
import SimpleITK as sitk

SOLID_THRESHOLD_HU = -300.0
ENTROPY_BIN_WIDTH_HU = 25.0
EROSION_RADIUS = 1


def dicom_to_hu(image):
    """Convert stored pixel values to Hounsfield units (raw * slope + intercept)."""
    slope = float(image.GetMetaData("0028|1053") or 1.0)
    intercept = float(image.GetMetaData("0028|1052") or 0.0)
    return sitk.Cast(image, sitk.sitkFloat32) * slope + intercept


def read_dicom_series(directory):
    """Read the (single) CT series from a directory of DICOM files."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise RuntimeError(f"no DICOM series found in {directory}")
    files = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def resample_mask_to_image(mask, image):
    """Resample the mask onto the image grid (nearest neighbour) when needed."""
    same_grid = (mask.GetSize() == image.GetSize()) and \
        all(abs(a - b) <= 1e-4 for a, b in zip(mask.GetSpacing(), image.GetSpacing()))
    if same_grid:
        return sitk.Cast(mask, sitk.sitkUInt8)
    return sitk.Resample(sitk.Cast(mask, sitk.sitkUInt8), image,
                         sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, sitk.sitkUInt8)


def erode_mask(mask, radius=EROSION_RADIUS):
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetKernelRadius(radius)
    eroder.SetForegroundValue(1)
    return eroder.Execute(mask)


def compute_features(image, mask):
    """Compute the nine features on the (already eroded) mask M."""
    array = sitk.GetArrayFromImage(image).astype(np.float64)
    mask_array = sitk.GetArrayFromImage(mask) > 0
    n = int(mask_array.sum())
    if n < 10:
        raise ValueError("mask contains fewer than 10 voxels")

    values = array[mask_array]

    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    volume = float(shape.GetPhysicalSize(1))     # mm^3
    surface_area = float(shape.GetPerimeter(1))  # mm^2

    sphericity = (36.0 * np.pi * volume ** 2) ** (1.0 / 3.0) / surface_area \
        if surface_area > 0 else float("nan")

    bins = np.floor((values - values.min()) / ENTROPY_BIN_WIDTH_HU).astype(np.int64)
    counts = np.bincount(bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log2(probs)).sum())

    return {
        "volume_mm3": volume,
        "surface_area_mm2": surface_area,
        "sphericity": float(sphericity),
        "ct_mean_hu": float(values.mean()),
        "ct_median_hu": float(np.median(values)),
        "ct_max_hu": float(values.max()),
        "ct_min_hu": float(values.min()),
        "entropy_bits": entropy,
        "psc": float((values > SOLID_THRESHOLD_HU).mean()),
        "n_voxels": n,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom", required=True, help="directory containing a DICOM CT series")
    p.add_argument("--mask", required=True, help="binary ROI mask (.nii.gz or other SimpleITK-readable)")
    p.add_argument("--out", default="features.json", help="output JSON path")
    p.add_argument("--no-erosion", action="store_true",
                   help="skip the one-voxel erosion (diagnostics only)")
    args = p.parse_args()

    image = dicom_to_hu(read_dicom_series(args.dicom))
    mask = resample_mask_to_image(sitk.ReadImage(args.mask), image)
    if not args.no_erosion:
        mask = erode_mask(mask)

    result = compute_features(image, mask)
    result["spacing_mm"] = [round(float(s), 4) for s in image.GetSpacing()]
    result["size_voxels"] = [int(s) for s in image.GetSize()]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
