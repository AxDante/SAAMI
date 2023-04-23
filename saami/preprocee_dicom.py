import os
import pydicom
import numpy as np
import nibabel as nib
import dicom2nifti

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

def convert_rtstruct_to_nifti(parent_folder, output_file):
    dicom2nifti.convert_directory(parent_folder, output_file, compression=True, reorient=True)

def convert_dicom_to_nifti(dicom_folder, output_nifti_file):
    dicom2nifti.convert_directory(dicom_folder, output_nifti_file)

def convert_dicom_to_nifti(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        dicom_files = [file for file in files if file.lower().endswith(".dcm")]
        if dicom_files:
            rtstruct_file = None
            for file in dicom_files:
                if is_rtstruct(os.path.join(root, file)):
                    rtstruct_file = os.path.join(root, file)
                    break

            output_dir = os.path.join(output_path, os.path.relpath(root, input_path))
            os.makedirs(output_dir, exist_ok=True)

            if rtstruct_file:
                output_file = os.path.join(output_dir, 'rtstruct.nii.gz')
                parent_folder = os.path.dirname(rtstruct_file)
                print('output file is : ', output_file)
                convert_rtstruct_to_nifti(parent_folder, output_file)
            else:
                pass
                #convert_dicom_to_nifti(dicom_folder, output_nifti_file)


def main():
    input_main_folder = "data/LCTSC_example"
    output_main_folder = "data/LCTSC_example_nifti"
    os.makedirs(output_main_folder, exist_ok=True)

    create_output_dirs(input_main_folder, output_main_folder)
    convert_dicom_to_nifti(input_main_folder, output_main_folder)

if __name__ == "__main__":
    main()