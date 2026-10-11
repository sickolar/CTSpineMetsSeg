import os
import napari
import numpy as np
import pydicom
import SimpleITK as sitk


def view_napari(dicom_path, mask_path):
    """
        Visualizes a DICOM CT series and a DICOM-SEG mask in 3D using Napari.

        Reads a directory of DICOM CT slices and a corresponding multi-frame DICOM-SEG file.
        Reconstructs the 3D mask by mapping segmentation frames to CT slices based on
        physical Z-coordinates (ImagePositionPatient[2]), and displays both volumes in Napari
        with real physical spacing.

        Args:
            dicom_path (str): Path to the directory containing the CT DICOM slices.
            mask_path (str): Path to the directory containing the DICOM-SEG file.

        Returns:
            None: Opens an interactive Napari viewer window.
    """
    dcm_files = []
    file_names = []
    pixel_arrays = []
    mask_file_names = []

    for file in os.listdir(dicom_path):
        if file.endswith('.dcm'):
            file_names.append(file)

    if not file_names:
        print('Could not find any dicom files')
        return

    file_names.sort()

    for file in file_names:
        file_path = os.path.join(dicom_path, file)
        dcm_data = pydicom.dcmread(file_path)
        dcm_files.append(dcm_data)

    z_spacing = abs(float(dcm_files[1].ImagePositionPatient[2]) - float(dcm_files[0].ImagePositionPatient[2]))
    y_spacing, x_spacing = dcm_files[0].PixelSpacing
    real_spacing = (z_spacing, float(y_spacing), float(x_spacing))

    for dcm in dcm_files:
        pixel_arrays.append(dcm.pixel_array)

    data_for_napari = np.array(pixel_arrays)

    for file in os.listdir(mask_path):
        if file.endswith('.dcm'):
            mask_file_names.append(file)

    mask_file_path = os.path.join(mask_path, mask_file_names[0])
    mask_dcm = pydicom.dcmread(mask_file_path)

    print(f"Původní CT: {data_for_napari.shape}")
    print(f"Původní Maska: {mask_dcm.pixel_array.shape} -> Z toho Napari blázní")

    # --- OPRAVA: PŘESKLÁDÁNÍ 1420 VRSTEV NA 797 PLÁTKŮ ---
    # 1. Vytvoříme si prázdnou masku, která bude mít přesně 797 plátků jako pacient
    mask_for_napari = np.zeros(data_for_napari.shape, dtype=np.uint8)

    # 2. Vytáhneme si fyzické Z-s
    #
    #
    # ouřadnice všech CT plátků pacienta
    ct_z_coords = [float(dcm.ImagePositionPatient[2]) for dcm in dcm_files]

    # 3. Projdeme všech 1420 vrstev v masce a "přilepíme" je na správný CT plátek
    for i, frame in enumerate(mask_dcm.PerFrameFunctionalGroupsSequence):
        # Kde fyzicky v těle leží tato vrstva masky?
        z_pos = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])

        # Najdeme, který CT plátek (0 až 796) je této pozici nejblíž
        z_index = np.argmin(np.abs(np.array(ct_z_coords) - z_pos))

        # Zjistíme třídu (číslo obratle nebo nádoru)
        try:
            segment_id = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except:
            segment_id = 1

        # Vepíšeme pixely z vrstvy masky do naší nové prázdné masky
        mask_for_napari[z_index][mask_dcm.pixel_array[i] > 0] = segment_id

    mask_for_napari = np.flip(mask_for_napari, axis=1)

    viewer = napari.Viewer()

    # Přidáme scale k oběma, jak jsi správně odhalil
    viewer.add_image(data_for_napari, colormap='gray', name='spine', scale=real_spacing)
    viewer.add_labels(mask_for_napari, name='anotated', opacity=0.5, scale=real_spacing)

    napari.run()



def view_nifti(ct_path, mask_path):
    """
        Visualizes a NIfTI CT volume and its corresponding NIfTI mask in 3D using Napari.

        Loads volumetric data and voxel spacing from NIfTI files using SimpleITK,
        and displays them as an image layer and a label layer in Napari.

        Args:
            ct_path (str): Path to the CT NIfTI file (e.g., .nii.gz).
            mask_path (str): Path to the segmentation mask NIfTI file.

        Returns:
            None: Opens an interactive Napari viewer window.
    """
    sitk_ct = sitk.ReadImage(ct_path)
    sitk_mask = sitk.ReadImage(mask_path)

    ct_array = sitk.GetArrayFromImage(sitk_ct)
    mask_array = sitk.GetArrayFromImage(sitk_mask)

    spacing = sitk_ct.GetSpacing()
    real_spacing = (spacing[2], spacing[1], spacing[0])

    viewer = napari.Viewer()
    viewer.add_image(ct_array, colormap='gray', name='spine', scale=real_spacing)
    viewer.add_labels(mask_array, name='anotated', opacity=0.5, scale=real_spacing)
    napari.run()


if __name__ == "__main__":
    #cesta_ct = r'C:\Projekty\segmentace_pater\subjects\manifest-1724965242274\Spine-Mets-CT-SEG\10250\04-04-2009-NA-SpineSPINEBONESBRT Adult-04098\5.000000-SKINTOSKINSIM0.5MM10250a iMAR-27242'

    #cesta_masky = r'C:\Projekty\segmentace_pater\subjects\manifest-1724965242274\Spine-Mets-CT-SEG\10250\04-04-2009-NA-SpineSPINEBONESBRT Adult-04098\300.000000-Spine Segmentation-86171'
    #view_napari(cesta_ct, cesta_masky)

    view_nifti(
        ct_path=r"C:\Projekty\segmentace_pater\dataset\images_dcm_to_nifti\spine_10250_0000.nii.gz",
        mask_path=r"C:\Projekty\segmentace_pater\dataset\labels_dcm_to_nifti\spine_10250.nii.gz"
    )
