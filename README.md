# Segment-Anything-Automatically-on-Medical-Image

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/napari-sam.svg?color=green)](https://github.com/MIC-DKFZ/napari-sam/raw/main/LICENSE)

Automatically segment anything on 3D medical images using Meta AI's new **Segment Anything Model (SAM)**. This is a simple project that helps the user visualize the automaticlly generated segmentation using the ``SamAutomaticMaskGenerator`` function on medical images. The project will be expanded so that the labels between each slice stays consistent and will eventually generate an output of 3D mask for the input images. The automaticlly generated 3D masks could be useful for further neural network training and fine-tuning in a semi-supervised fashion.


![](images/spine_example.png)


## Installation

To install the dependencies required for "Segment-Anything-Automatically-on-Medical-Image":

- Open a terminal or command prompt.

- Navigate to the directory where the "Segment-Anything-Automatically-on-Medical-Image" project is located.

- Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run testing on a MRI spine dataset using the following command:
```bash
python sammmi_example.py
```

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"saami" is free and open source software
