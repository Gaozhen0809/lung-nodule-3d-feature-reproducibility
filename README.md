# Independent recomputation of nine three-dimensional CT features

This repository contains the independent (SimpleITK) implementation used to
verify the reproducibility of nine three-dimensional CT features described in
the associated manuscript. The implementation reads a thin-slice CT DICOM
series and a binary ROI mask, and recomputes the features without access to the
commercial software output.

## Workflow (Figure S7)

Thin-slice CT -> read DICOM series -> convert to Hounsfield units ->
resample ROI mask onto the CT grid (nearest neighbour) -> one-voxel erosion
(ball, radius 1 voxel) -> compute features on the eroded mask M.

## Feature definitions

Let M be the eroded ROI mask and N the number of voxels in M; HU(x) denotes the
Hounsfield unit value at voxel x.

| Feature | Formula |
|---|---|
| Volume | N x (voxel volume) |
| Surface area | physical surface area of M (SimpleITK `LabelShapeStatisticsImageFilter.GetPerimeter`) |
| Sphericity | (36 x pi x Volume^2)^(1/3) / Surface area |
| CT mean | (1/N) sum HU(x) |
| CT median | median{ HU(x) } |
| CT max | max{ HU(x) } |
| CT min | min{ HU(x) } |
| Entropy | -sum_k p_k log2(p_k), with p_k from fixed 25-HU bins whose edges start at the minimum HU in M |
| PSC | #{ x : HU(x) > -300 } / N |

## Implementation notes

- DICOM values are converted to HU as `raw x rescale slope + rescale intercept`.
- "Surface area" uses SimpleITK's `GetPerimeter` on the 3-D label (the
  definition used in the study; it may differ slightly from a strict
  face-counting definition).
- Two additional platform features, Compactness and Quality, could not be
  matched to an independent definition and are therefore not implemented here.
- The study computed features on the one-voxel-eroded mask; pass `--no-erosion`
  only for diagnostics.

## Requirements

Python 3.8+ with numpy, pandas, SimpleITK (`pip install -r requirements.txt`).

## Usage

Single case:

    python compute_features.py --dicom /path/to/series --mask /path/to/roi.nii.gz --out features.json

Batch (loop over a manifest):

    while read -r series mask; do
      python compute_features.py --dicom "$series" --mask "$mask" --out "$series.json"
    done < manifest.txt

## Citation

## Citation

If you use this software, please cite it as:

Gao Z. Independent recomputation of nine three-dimensional CT features for lung nodules. Zenodo; 2026. doi:10.5281/zenodo.23192347

## License

MIT - see LICENSE.
