import os
import pydicom
import cv2
import numpy as np
import nibabel as nib
import argparse
from dcmrtstruct2nii import dcmrtstruct2nii
import shutil


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
            mapping_file.write("{}: {}\n".format(label, file))

    print("Files combined successfully.")

def get_adj_folder(path):

    parent_folder = os.path.dirname(path)
    subfolders = os.listdir(parent_folder)
    subfolders.remove(os.path.basename(path))
    return os.path.join(parent_folder, subfolders[0])

def convert_dataset(input_path, output_path, dataset_name='LCTSC'):

    # Volume Dataset
    if dataset_name == 'LCTSC' or dataset_name == 'BraTS':

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

                if dataset_name == 'LCTSC':
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

    if dataset_name == 'amos22':

        images_folder = os.path.join(input_path, 'imagesTr')
        labels_folder = os.path.join(input_path, 'labelsTr')
        
        image_files = os.listdir(images_folder)
        label_files = os.listdir(labels_folder)

        for idx, (image_file, label_file) in enumerate(zip(sorted(image_files), sorted(label_files))):
            case_dir = os.path.join(output_path, "{:04d}".format(idx+1))
            images_dir = os.path.join(case_dir, 'Images')
            labels_dir = os.path.join(case_dir, 'Labels')

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            shutil.copy(os.path.join(images_folder, image_file), os.path.join(images_dir, 'image.nii.gz'))
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(labels_dir, 'combined_seg.nii.gz'))


    # Image Dataset
    if dataset_name == 'LungSeg':
        
        # Nothing much to change here. This dataset is already in the desired format.
        image_folder = os.path.join(input_path, 'CXR_png')
        label_folder = os.path.join(input_path, 'masks')
        readings_folder = os.path.join(input_path, 'ClinicalReadings')


        output_img_folder = os.path.join(output_path, 'Images')
        output_label_folder = os.path.join(output_path, 'Labels')

        # Create output folders
        if not os.path.exists(output_img_folder):
            os.makedirs(output_img_folder)
        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

        images = os.listdir(image_folder)
        labels = os.listdir(label_folder)
        readings = os.listdir(readings_folder)
        images.sort()
        labels.sort()
        readings.sort()
        
        for image, label, reading in zip(images, labels, readings):

            shutil.copy(os.path.join(image_folder, image), os.path.join(output_img_folder, image))
            shutil.copy(os.path.join(label_folder, label), os.path.join(output_label_folder, label))
            shutil.copy(os.path.join(readings_folder, reading), os.path.join(output_label_folder, reading))


def parse_args():

    parser = argparse.ArgumentParser(description="Convert dataset to desired format")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_main_folder",
        default="data/BraTS_example",
        help="Path to the input main folder (default: 'data/BraTS_example')",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_main_folder",
        default="data/BraTS_example_processed",
        help="Path to the output main folder (default: 'data/BraTS_example_processed')",
    )

    parser.add_argument(
        "-d",
        "--dataset_name",
        dest="dataset_name",
        default="BraTS",
        choices=["BraTS", "LCTSC", "LungSeg", "amos22"],
        help="Dataset name (default: 'BraTS')",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_main_folder = args.input_main_folder
    output_main_folder = args.output_main_folder
    dataset_name = args.dataset_name

    os.makedirs(output_main_folder, exist_ok=True)

    convert_dataset(input_main_folder, output_main_folder, dataset_name=dataset_name)

if __name__ == "__main__":
    main()

