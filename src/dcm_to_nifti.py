import SimpleITK as sitk
import os
import pydicom
import numpy as np

def dcm_to_niftii(ct_dicom_dir, seg_dicom_dir, out_ct_path, out_mask_path):
    """
        Converts a DICOM CT series and a DICOM Segmentation (DICOM-SEG) into NIfTI format.

        This function loads a directory of CT slices and a corresponding multi-frame DICOM-SEG file.
        It accurately maps the segmentation frames to the CT slices based on the Z-axis coordinate
        (ImagePositionPatient[2]), applies the corresponding segment IDs, matches the spatial metadata,
        and saves both volumes as NIfTI files.

        Args:
            ct_dicom_dir (str): Path to the directory containing the CT DICOM slices.
            seg_dicom_dir (str): Path to the directory containing the DICOM-SEG file.
            out_ct_path (str): Output file path for the converted CT NIfTI (e.g., .nii.gz).
            out_mask_path (str): Output file path for the converted mask NIfTI (e.g., .nii.gz).

        Returns:
            None: The function saves the generated files directly to the specified paths.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dicom_dir)
    reader.SetFileNames(dicom_names)
    sitk_ct = reader.Execute()

    ct_files = []
    for f in os.listdir(ct_dicom_dir):
        if f.endswith('.dcm'):
            ct_files.append(pydicom.dcmread(os.path.join(ct_dicom_dir, f)))
    ct_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    ct_z_coords = []
    for dcm in ct_files:
        ct_z_coords.append(float(dcm.ImagePositionPatient[2]))
    ct_z_coords = np.array(ct_z_coords)

    depth = len(ct_files)
    rows = ct_files[0].Rows
    cols = ct_files[0].Columns
    mask_array = np.zeros((depth, rows, cols), dtype=np.uint8)

    for f in os.listdir(seg_dicom_dir):
        if f.endswith('.dcm'):
            seg_file = f
            break
    seg_dcm = pydicom.dcmread(os.path.join(seg_dicom_dir, seg_file))

    for i, frame in enumerate(seg_dcm.PerFrameFunctionalGroupsSequence):
        z_pos = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
        z_index = np.argmin(np.abs(ct_z_coords - z_pos))

        try:
            segment_id = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except Exception:
            segment_id = 1

        mask_array[z_index][seg_dcm.pixel_array[i] > 0] = segment_id

    mask_array = np.flip(mask_array, axis=1).copy()

    sitk_mask = sitk.GetImageFromArray(mask_array)
    sitk_mask.CopyInformation(sitk_ct)

    os.makedirs(os.path.dirname(out_ct_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

    sitk.WriteImage(sitk_ct, out_ct_path)
    sitk.WriteImage(sitk_mask, out_mask_path)

    print(f'CT image saved to: {out_ct_path}')
    print(f'Mask saved to: {out_mask_path}')

if __name__ == "__main__":
    dcm_to_niftii(
        ct_dicom_dir=r"C:\Projekty\segmentace_pater\subjects\manifest-1724965242274\Spine-Mets-CT-SEG\10250\04-04-2009-NA-SpineSPINEBONESBRT Adult-04098\5.000000-SKINTOSKINSIM0.5MM10250a iMAR-27242",
        seg_dicom_dir=r"C:\Projekty\segmentace_pater\subjects\manifest-1724965242274\Spine-Mets-CT-SEG\10250\04-04-2009-NA-SpineSPINEBONESBRT Adult-04098\300.000000-Spine Segmentation-86171",
        out_ct_path=r"C:\Projekty\segmentace_pater\dataset\images_dcm_to_nifti\spine_10250_0000.nii.gz",
        out_mask_path=r"C:\Projekty\segmentace_pater\dataset\labels_dcm_to_nifti\spine_10250.nii.gz"
    )
