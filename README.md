# Dataleon_test_technique

### Introduction

This project focuses on testing cases related to image object detection using transformers. Specifically, it utilizes a pre-trained model available at Hugging Face to create a Python class capable of predicting tables in invoice and bank document images. The project includes the creation of a DocumentTableDetector class and writing pytest scripts to cover various scenarios, including successful table extraction and error handling for both types of documents.


### Getting Started
##### Prerequisites
Ensure you have Python 3 installed on your system.

##### Step 1 : Installation
To set up the environment and install all the required packages, run the following command in your bash shell:

```bash
make install
```bash

##### Step 2: Running Tests
To run the tests and validate the functionality of the DocumentTableDetector class, use:

```bash
make test
```
##### Running All

To install the requirements and run the tests consecutively, use:
```bash
make all
```

### Pytest Scripts
The project includes pytest scripts that cover the following scenarios:

    - Successful table extraction from invoice and bank document images.
    - Error handling for unsupported file formats.
    - Detection in rotated images to ensure robustness against image orientation.
    - Detection of multiple tables in a single image.


### Requirements
All required packages are listed in the requirements.txt file. You can install them using the make install command.


### Acknowledgments
- The pre-trained DETR model from Hugging Face.
- The developers and contributors of the Python libraries and tools used in this project.

By following the instructions provided in this README.md, you should be able to set up the project, install necessary dependencies, and run tests efficiently using the bash shell and the provided Makefile.


### Project Structure
```bash
project_root/
├── notebooks/
│   └── Table_detection.ipynb          # Notebook that tests model output on a single sample
├── src/
│   └── document_table_detection.py    # DocumentTableDetector class with methods for initializing the model, processing images, predicting tables, and drawing boxes with confidence scores
├── unit_tests/
│   ├── test_document_table_detector.py # Unit tests for different scenarios
│   └── images/                         # Directory containing test images
│       ├── invoice.jpg
│       ├── no_table.jpg
│       ├── bank_document.png
│       ├── rotated_table.png
│       ├── unsupported_format.gif
│       └── multiple_tables.png
├── requirements.txt                   # Required packages
├── Makefile                           # Makefile for installing requirements and running tests
└── README.md                          # Project README file
```
