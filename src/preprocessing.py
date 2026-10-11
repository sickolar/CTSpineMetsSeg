import os
import json
import numpy as np
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from scipy import ndimage

def load_nifti(ct_path, mask_path):
    """
        Loads CT and segmentation NIfTI files and extracts
        initial metadata (spacing, size, origin and direction).

        Args:
            ct_path (str): Path to the input CT NIfTI file.
            mask_path (str): Path to the input mask NIfTI file.

        Returns:
            tuple: A tuple containing:
                - sitk_ct (sitk.Image): The loaded CT volume.
                - sitk_mask (sitk.Image): The loaded segmentation mask.
                - meta (dict): A dictionary containing the original spatial metadata
                  (spacing, size, origin, direction).
    """
    sitk_ct = sitk.ReadImage(ct_path)
    sitk_mask = sitk.ReadImage(mask_path)

    meta = {
        "original_spacing":   list(sitk_ct.GetSpacing()),
        "original_size":      list(sitk_ct.GetSize()),
        "original_origin":    list(sitk_ct.GetOrigin()),
        "original_direction": list(sitk_ct.GetDirection()),
    }

    return sitk_ct, sitk_mask, meta


def otsu_body_crop(sitk_ct, sitk_mask, meta, margin=10):
    """
        Isolates the patient's body and crops out excess background air.

        Applies a [-1024, 1000] HU clip and Otsu thresholding to create a body mask.
        Retains only the largest connected component to filter out artifacts or scanning bed parts.
        Calculates a 3D bounding box with a specified margin to crop both the CT and mask,
        and updates the physical origin to maintain correct spatial mapping.

        Args:
            sitk_ct (sitk.Image): The SimpleITK CT volume.
            sitk_mask (sitk.Image): The SimpleITK mask volume.
            meta (dict): The metadata dictionary to be updated with crop coordinates.
            margin (int, optional): Number of background voxels to keep around the body bounding box. Defaults to 10.

        Returns:
            tuple: (sitk_ct_cropped, sitk_mask_cropped, meta) containing the cropped volumes and updated metadata.
    """
    spacing = sitk_ct.GetSpacing()
    origin = sitk_ct.GetOrigin()

    ct_array = sitk.GetArrayFromImage(sitk_ct).astype(np.float32)
    mask_array = sitk.GetArrayFromImage(sitk_mask)

    clipped = np.clip(ct_array, -1024, 1000)
    thresh = threshold_otsu(clipped)
    body_mask = clipped > thresh

    labeled, num_features = ndimage.label(body_mask)
    if num_features > 1:
        sizes = ndimage.sum(body_mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        body_mask = labeled == largest #boolean comparator

    coords = np.argwhere(body_mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    D, H, W = ct_array.shape
    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, D - 1)
    y_max = min(y_max + margin, H - 1)
    x_max = min(x_max + margin, W - 1)

    ct_cropped   = ct_array  [z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    mask_cropped = mask_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

    new_origin = (
        origin[0] + x_min * spacing[0],
        origin[1] + y_min * spacing[1],
        origin[2] + z_min * spacing[2],
    )

    sitk_ct_cropped = sitk.GetImageFromArray(ct_cropped)
    sitk_ct_cropped.SetSpacing(spacing)
    sitk_ct_cropped.SetOrigin(new_origin)
    sitk_ct_cropped.SetDirection(sitk_ct.GetDirection())

    sitk_mask_cropped = sitk.GetImageFromArray(mask_cropped)
    sitk_mask_cropped.SetSpacing(spacing)
    sitk_mask_cropped.SetOrigin(new_origin)
    sitk_mask_cropped.SetDirection(sitk_ct.GetDirection())

    meta["crop_z"] = [int(z_min), int(z_max)]
    meta["crop_y"] = [int(y_min), int(y_max)]
    meta["crop_x"] = [int(x_min), int(x_max)]
    meta["crop_origin"] = list(new_origin)

    return sitk_ct_cropped, sitk_mask_cropped, meta


def resample_isotropic(sitk_ct, sitk_mask, meta, target_spacing=(1.0, 1.0, 1.0)):
    """
        Resamples volumes to a uniform isotropic physical spacing.

        Converts the voxel spacing to the target dimensions (e.g., 1x1x1 mm) while maintaining
        the physical volume size. Uses Linear interpolation for the continuous CT data (filling
        outside bounds with -1024 HU) and Nearest Neighbor interpolation for the categorical mask
        data (filling outside bounds with 0).

        Args:
            sitk_ct (sitk.Image): The input CT volume.
            sitk_mask (sitk.Image): The input mask volume.
            meta (dict): The metadata dictionary to be updated with the target spacing.
            target_spacing (tuple, optional): The desired physical spacing in mm. Defaults to (1.0, 1.0, 1.0).

        Returns:
            tuple: (sitk_ct_resampled, sitk_mask_resampled, meta) containing the isotropic volumes and updated metadata.
    """
    def _resample(sitk_img, interpolator, default_value):
        original_spacing = sitk_img.GetSpacing()
        original_size = sitk_img.GetSize()

        new_size = []
        for i in range(3):
            new_size.append(int(round(original_size[i] * original_spacing[i] / target_spacing[i]))) #keeps original mm

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(target_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputOrigin(sitk_img.GetOrigin())
        resample_filter.SetOutputDirection(sitk_img.GetDirection())
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetDefaultPixelValue(default_value)

        return resample_filter.Execute(sitk_img)

    sitk_ct_resampled = _resample(sitk_ct, sitk.sitkLinear, default_value=-1024)
    sitk_mask_resampled = _resample(sitk_mask, sitk.sitkNearestNeighbor, default_value=0)

    meta["target_spacing"] = list(target_spacing)

    return sitk_ct_resampled, sitk_mask_resampled, meta


def clip_and_normalize(ct_array, meta, hu_min=-200, hu_max=1000):
    """
        Applies Hounsfield Unit (HU) windowing and normalizes the voxel intensities.

        Clips the CT array values to a specific HU range to enhance the contrast of the tissues
        of interest (e.g., bones/spine), and applies min-max normalization to scale the resulting
        values between 0.0 and 1.0.

        Args:
            ct_array (np.ndarray): The raw CT intensity values as a NumPy array.
            meta (dict): The metadata dictionary to be updated with the windowing limits.
            hu_min (int, optional): The lower bound for HU clipping. Defaults to -200.
            hu_max (int, optional): The upper bound for HU clipping. Defaults to 1000.

        Returns:
            tuple: (ct_normalized, meta) where the array is cast to float32, along with the updated metadata.
    """
    ct_clipped = np.clip(ct_array, hu_min, hu_max)
    ct_normalized = (ct_clipped - hu_min) / (hu_max - hu_min)
    ct_normalized = np.clip(ct_normalized, 0.0, 1.0)

    meta["hu_min"] = hu_min
    meta["hu_max"] = hu_max

    return ct_normalized.astype(np.float32), meta

def save_preprocessed(ct_norm, mask_norm, meta, out_path, subject_id):
    """
        Saves the preprocessed dataset to output path.

        Creates subdirectories and saves the normalized CT and mask arrays as
        NumPy binary files (.npy). The mask is cast to uint8 to save memory, and
        the accumulated tracking metadata is saved as a JSON file.

        Args:
            ct_norm (np.ndarray): The preprocessed and normalized CT array (float32).
            mask_norm (sitk.Image): The preprocessed mask as a SimpleITK image.
            meta (dict): The accumulated tracking metadata.
            out_path (str): The root directory where output folders will be created.
            subject_id (str or int): The unique identifier for the processed patient/scan.

        Returns:
            None: Files are written directly to the target subdirectories.
    """

    ct_dir = os.path.join(out_path, 'ct_images')
    mask_dir = os.path.join(out_path, 'mask_lab')
    meta_dir = os.path.join(out_path, 'meta')

    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    #Saving numpy for speed
    mask_array = sitk.GetArrayFromImage(mask_norm)

    ct_array = ct_norm.astype(np.float32)
    mask_array = mask_array.astype(np.uint8) #labels dont need float 0-14

    ct_file = os.path.join(ct_dir, f'ct_{subject_id}.npy')
    mask_file = os.path.join(mask_dir, f'mask_{subject_id}.npy')
    meta_file = os.path.join(meta_dir, f'meta_{subject_id}.json')

    np.save(ct_file, ct_array)
    np.save(mask_file, mask_array)

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)

