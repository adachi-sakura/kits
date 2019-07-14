from pathlib import Path
import argparse

import scipy.misc
import numpy as np

from utils import load_case

# Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_PLANE = "axial"


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation,1)] = k_color
    seg_color[np.equal(segmentation,2)] = t_color
    return seg_color

def visualize(cid, destination_img,destination_label, hu_min=DEFAULT_HU_MIN, hu_max=DEFAULT_HU_MAX,
    k_color=DEFAULT_KIDNEY_COLOR, t_color=DEFAULT_TUMOR_COLOR,
    alpha=DEFAULT_OVERLAY_ALPHA, plane=DEFAULT_PLANE):

    plane = plane.lower()

    plane_opts = ["axial", "coronal", "sagittal"]
    if plane not in plane_opts:
        raise ValueError((
            "Plane \"{}\" not understood. " 
            "Must be one of the following\n\n\t{}\n"
        ).format(plane, plane_opts))

    # Prepare output location
    out_path = Path(destination_img)
    if not out_path.exists():
        out_path.mkdir()

    out_path2=Path(destination_label)
    if not out_path2.exists():
        out_path2.mkdir()

    # Load segmentation and volume
    vol, seg = load_case(cid)
    spacing = vol.affine
    vol = vol.get_data()
    seg = seg.get_data()
    seg = seg.astype(np.int32)

    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    seg_ims = class_to_color(seg, k_color, t_color)

    for i in range(vol_ims.shape[0]):
        fpath = out_path / ("{:05d}.png".format(i))
        fpath2=out_path2/("{:05d}.png".format(i))
        scipy.misc.imsave(str(fpath), vol_ims[i])
        scipy.misc.imsave(str(fpath2),seg_ims[i])