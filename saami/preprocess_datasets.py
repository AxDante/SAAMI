import os
import pydicom
import numpy as np
import nibabel as nib
import dicom2nifti
import SimpleITK as sitk
from dicompylercore import dicomparser
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

def create_output_dirs(input_path, output_path):

    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            output_dir = os.path.join(output_path, os.path.relpath(os.path.join(root, dir), input_path))
            os.makedirs(output_dir, exist_ok=True)


def is_rtstruct(dicom_file):
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        return ds.Modality == 'RTSTRUCT'
    except Exception as e:
        return False


def combine_segmentation_files(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nii.gz") and f != "combined_seg.nii.gz"])
    combined_data = None
    label_mapping = [("0", "background")]

    for i, file in enumerate(files):
        nii_path = os.path.join(folder_path, file)
        img = nib.load(nii_path)
        data = img.get_fdata()

        if combined_data is None:
            combined_data = np.zeros_like(data, dtype=np.uint8)

        label = i + 1
        combined_data[data >= 1] = label
        label_mapping.append((str(label), os.path.splitext(file)[0]))

    # Save combined segmentation
    combined_img = nib.Nifti1Image(combined_data, img.affine)
    combined_nii_path = os.path.join(folder_path, "combined_seg.nii.gz")
    nib.save(combined_img, combined_nii_path)

    # Save label mapping to text file
    mapping_file_path = os.path.join(folder_path, "label_mapping.txt")
    with open(mapping_file_path, "w") as mapping_file:
        for label, file in label_mapping:
            mapping_file.write(f"{label}: {file}\n")

    print("Files combined successfully.")


def get_adj_folder(path):
    parent_folder = os.path.dirname(path)
    subfolders = os.listdir(parent_folder)
    subfolders.remove(os.path.basename(path))
    return os.path.join(parent_folder, subfolders[0])

def convert_dataset(input_path, output_path, dataset_type='LCTSC'):

    if dataset_type == 'LCTSC' or dataset_type == 'BraTS':
        patient_names = os.listdir(input_path)
        print(patient_names)




    for patient in patient_names:

        patient_input_folders = os.path.join(input_path, patient)
        patient_img_folders = os.path.join(output_path, patient, 'Images')
        patient_labels_folders = os.path.join(output_path, patient, 'Labels')

        # Create output folders
        if not os.path.exists(patient_img_folders):
            os.makedirs(patient_img_folders)
        if not os.path.exists(patient_labels_folders):
            os.makedirs(patient_labels_folders)

        for root, dirs, files in os.walk(patient_input_folders):

            # Convert RTSTRUCT to nifti
            if dataset_type == 'LCTSC':
                dicom_files = [file for file in files if file.lower().endswith(".dcm")]

                if dicom_files:
                    rtstruct_file = None
                    for file in dicom_files:
                        if is_rtstruct(os.path.join(root, file)):
                            rtstruct_file = os.path.join(root, file)
                            break
                
                    if rtstruct_file:
                        print('Processing data from : ', root)
                        adj_folder = get_adj_folder(root)
                        dcmrtstruct2nii(rtstruct_file,
                                        adj_folder, 
                                        patient_labels_folders)
                        image_file = os.path.join(patient_labels_folders, 'image.nii.gz')
                        os.rename(image_file, os.path.join(patient_img_folders, 'image.nii.gz'))
                        combine_segmentation_files(patient_labels_folders)




def main():

    input_main_folder = "data/LCTSC_example"
    output_main_folder = "data/LCTSC_example_processed"

    # input_main_folder = "data/BraTS_example"
    # output_main_folder = "data/BraTS_example_processed"


    os.makedirs(output_main_folder, exist_ok=True)

    convert_dataset(input_main_folder, output_main_folder)

if __name__ == "__main__":
    main()