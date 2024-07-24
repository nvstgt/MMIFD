# MMIFD Script: Multi-Modal Image Forgery Detection Script

## Description

MMIFD is a comprehensive tool designed for detecting image and video forgeries using various machine learning and computer vision techniques. The script extracts features from images and videos, performs sparse representation, and compares the signatures against a database of authentic samples to determine the authenticity of the media. Additionally, it performs multiple analyses to enhance the detection accuracy, including audio consistency checks, temporal inconsistencies checks, and dynamic thresholding based on statistical analysis. The system logs resource usage such as memory, CPU, and thread count, providing detailed logging and reporting of results.

MMIFD utilizes parallel processing to efficiently analyze multiple files, generating robust fingerprints for videos to ensure reliable detection of forgeries. The tool also includes custom Keras layers and optimizers for deep learning models, and it has functions for validating and analyzing the database, PCA model, dictionary, and extracted features. The system is capable of building, validating, and saving databases of authentic samples for comparison, making it a powerful solution for image and video forgery detection.

### Updated README for the Final Version of the Script

---

## Overview

This script provides an image and video forgery detection system written in Python. The system leverages various deep learning models and analytical techniques to detect forgeries in both images and videos. The main components include feature extraction using pre-trained models, Principal Component Analysis (PCA) for dimensionality reduction, sparse representation, robust fingerprinting, optical flow analysis, texture analysis, and liveness detection. Additionally, the system includes audio consistency checks, temporal inconsistencies checks, dynamic thresholding, and comprehensive logging and reporting features. It employs parallel processing to efficiently handle large datasets and utilizes custom Keras layers and optimizers for enhanced deep learning capabilities. The script also supports building, validating, and saving databases of authentic samples for effective forgery detection.

## Features

- **Feature Extraction**: Uses pre-trained models (VGG16, ResNet50, and InceptionV3) to extract features from images and videos. These features capture the essential characteristics of the media files, making it possible to identify unique patterns and anomalies.
- **Sparse Representation**: Utilizes SparseCoder and dictionary learning for creating sparse representations of the extracted features. This process encodes the features in a compact form, allowing for efficient comparison and storage.
- **Robust Fingerprinting**: Generates robust fingerprints for images and videos by combining multiple features (texture, optical flow, lighting consistency, etc.) into a unique signature. These fingerprints are used to identify and verify the authenticity of media files, ensuring reliable detection of forgeries.
- **Dimensionality Reduction**: Employs PCA (Principal Component Analysis) for reducing the complexity of feature data. This step helps in maintaining the most important aspects of the data while reducing its dimensionality, which improves the efficiency of comparisons.
- **Optical Flow Analysis**: Detects inconsistencies in video frames by analyzing the motion between consecutive frames. Optical flow analysis helps in identifying unnatural movements that may indicate a forgery.
- **Texture Analysis**: Analyzes the visual texture of images and videos to detect anomalies. This method assesses the smoothness or granularity of the media files, helping to identify manipulated areas.
- **Shadow and Lighting Consistency**: Assesses lighting direction and consistency in images. This feature checks for discrepancies in the direction and consistency of shadows and lighting, which can reveal tampering.
- **JPEG Artifacts and Noise Analysis**: Identifies anomalies in JPEG compression and image noise. This analysis detects unusual patterns in the compression artifacts and noise levels, which can indicate manipulation.
- **Metadata Analysis**: Examines the metadata for inconsistencies. Metadata analysis checks the EXIF data of images for irregularities that may suggest tampering or forgery.
- **PRNU Analysis**: Detects inconsistencies in sensor noise patterns of images. Photo-Response Non-Uniformity (PRNU) analysis compares the sensor noise pattern of an image to known patterns to identify forgeries.
- **Liveness Detection**: Assesses videos for signs of genuine human presence. This feature analyzes facial movements, eye aspect ratios, and blink rates to distinguish real videos from deepfakes or other forgeries.
- **Frequency Domain Analysis**: Analyzes the frequency components of images to identify tampering. This method examines the high-frequency details in the frequency domain, which can reveal compression artifacts and manipulations.
- **Audio Consistency Check**: Extracts audio features using MFCC, chroma, and spectral contrast, and checks for consistency with visual features.
- **Deepfake Detection**: Performs deepfake detection using a custom-trained deepfake detection model, VGG16, InceptionV3, and ResNet50.
- **Temporal Inconsistencies Check**: Checks for temporal inconsistencies in videos by analyzing frame-to-frame differences.
- **Comprehensive Logging and Reporting**: Provides detailed logs and results. The system records all steps of the processing and analysis, ensuring transparency and traceability of the forgery detection process.
- **Automated Data Augmentation**: Enhances the training data by applying transformations (rotation, zoom, noise addition) to create multiple versions of each authentic image. This step increases the diversity and robustness of the training dataset.
- **Dynamic Thresholding**: Uses dynamic thresholds based on statistical analysis of feature distributions to improve the accuracy of forgery detection. This approach adapts the decision criteria based on the characteristics of the dataset.
- **Parallel Processing**: Utilizes concurrent processing to speed up the analysis of multiple files. This feature leverages multi-threading and multiprocessing techniques to handle large datasets efficiently.

## Requirements

- Python 3.7 or higher
- See `requirements.txt` for a full list of dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the script, ensure you have a `config.ini` file in the same directory. This file should contain paths to directories and other configuration parameters required by the script. Here is a sample configuration:

```ini
[Paths]
database_path = /path/to/MMIFD-DB/mmifd_db.npy
pca_model_path = /path/to/MMIFD-DB/pca_model.pkl
light_direction_model_path = /path/to/MMIFD-DB/light_direction_model.pkl
dictionary_path = /path/to/MMIFD-DB/dictionary.npy
authentic_images_directory = /path/to/MMIFD-DB/Still
authentic_videos_directory = /path/to/MMIFD-DB/Video
corpus_directory = /path/to/Unknown Corpus
ground_truth_labels_path = /path/to/MMIFD-DB/ground_truth_labels.csv

[ModelParameters]
fixed_length = 473 

[FeatureExtraction]
n_components = 473

[Timeouts]
timeout_duration = 5400

[Logging]
log_file_path = MMIFD_log.txt
result_file_path = MMIFD_results.txt
logging_level = DEBUG

[AccuracyMetrics]
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

[Thresholds]
lighting_consistency_threshold = 0.66
frequency_domain_threshold = 0.0044
prnu_threshold = 0.000075
metadata_threshold = 0.40
jpeg_artifacts_threshold = 0.15
noise_threshold = 0.92
cfa_threshold = 0.5
texture_threshold = 90.0
temporal_consistency_threshold = 1.125
optical_flow_threshold = 0.125
dynamic_threshold_factor = 0.05
static_threshold_value = 0.05
deepfake_detection_threshold = 0.35
eye_ar_threshold = 0.1

```

### Note
Ensure all paths specified in the `config.ini` file are correct and the necessary directories and files exist before running the script.

## Running the Script

To run the script, execute:
```
python MMIFD-v5.py
```

The script performs the following steps:

1. **Initialization and Setup**:
   - Imports necessary libraries and modules.
   - Reads configurations from the `config.ini` file.
   - Sets up logging.

2. **Loading Pre-trained Models**:
   - Loads pre-trained models (VGG16, ResNet50, InceptionV3) for feature extraction.

3. **Database and PCA Initialization**:
   - Validates or builds the database of authentic media files.
   - If no valid database exists, it creates one from the provided authentic media files.
   - Trains or loads the PCA model for dimensionality reduction.

4. **Feature Extraction and Sparse Representation**:
   - Extracts features from images and videos using the pre-trained models.
   - Applies PCA for dimensionality reduction.
   - Converts features to sparse representations for efficient comparison.

5. **Forgery Detection**:
   - Processes media files in the specified corpus directory.
   - For images:
     - Extracts features.
     - Creates sparse representations.
     - Compares them to the database.
     - Performs texture analysis, shadow and lighting consistency checks, JPEG artifacts analysis, noise analysis, metadata analysis, PRNU analysis, CFA analysis, and frequency domain analysis.
   - For videos:
     - Performs liveness detection.
     - Conducts optical flow analysis.
     - Analyzes texture.
     - Checks AV consistency.
     - Checks temporal consistency.
     - Conducts deepfake detection.

6. **Dynamic Threshold Calculation**:
   - Calculates dynamic thresholds based on the statistical analysis of feature distributions in the database.

7. **Logging and Results**:
   - Logs detailed information about each processing step.
   - Writes results to the specified results file.

TL;DR: The script will process each file in the specified corpus directory, perform the necessary analyses, and log the results.

---

## Estimation of Accuracy

The system combines multiple methods to achieve high accuracy in detecting forgeries:

- **Texture Analysis**: Achieves an accuracy of 80-90% by examining the smoothness or granularity of media files to identify manipulated areas.
- **PRNU Analysis**: Provides 85-95% accuracy by comparing sensor noise patterns of images to known patterns to detect inconsistencies.
- **Shadow and Lighting Consistency**: Offers 70-85% accuracy by assessing the direction and consistency of shadows and lighting in images to reveal tampering.
- **Frequency Domain Analysis**: Attains 75-90% accuracy by examining high-frequency details in the frequency domain to identify compression artifacts and manipulations.
- **Metadata Analysis**: Delivers 60-80% accuracy by checking the EXIF data of images for irregularities that may suggest tampering or forgery.
- **Optical Flow Analysis**: Reaches 80-90% accuracy by detecting inconsistencies in video frames through the analysis of motion between consecutive frames.
- **Liveness Detection**: Ensures 85-95% accuracy by analyzing facial movements, eye aspect ratios, and blink rates to distinguish real videos from forgeries.
- **Robust Fingerprinting**: Provides 85-95% accuracy by generating unique signatures from multiple features (texture, optical flow, lighting consistency, etc.) to verify the authenticity of media files.
- **JPEG Artifacts Analysis**: Achieves 70-85% accuracy by identifying unusual patterns in the compression artifacts, which can indicate manipulation.
- **Noise Analysis**: Provides 70-85% accuracy by examining image noise levels for inconsistencies that may suggest tampering.
- **Dynamic Thresholding**: Uses statistical analysis of feature distributions to improve the accuracy of forgery detection dynamically, estimated at 75-85%.
- **AV Consistency**: Offers 70-85% accuracy by checking the consistency between audio and video features, which helps in detecting mismatches indicative of forgeries.
- **Temporal Consistency**: Attains 75-90% accuracy by analyzing temporal inconsistencies in video frames to detect unnatural transitions or manipulations.
- **Deepfake Detection**: Reaches 85-95% accuracy by employing deep learning models to identify deepfake videos based on subtle inconsistencies in facial movements and other features.
- **Color Filter Array (CFA) Analysis**: Ensures 75-85% accuracy by examining the CFA pattern consistency in images to detect manipulations.
- **Sparse Representation**: Provides 80-90% accuracy by using sparse representation techniques to encode features in a compact form, allowing for efficient comparison and anomaly detection.

Combined, the system achieves a theoretical overall accuracy of approximately 90-95%. PRactical accuracy is highly dependent on your authentic image dataset.

---

## Index of Functions

### Loading Models
```python
vgg16_model = VGG16(weights='imagenet', include_top=False)
resnet_model = ResNet50(weights='imagenet', include_top=False)
inception_model = InceptionV3(weights='imagenet', include_top=False)
```

### Building the Database
```python
database, pca = build_database(authentic_images_directory, authentic_videos_directory, dictionary, model1, model2, model3, fixed_length)
```

### Feature Extraction
```python
def extract_features(image, model1, model2, model3, fixed_length, pca):
    # Extract features from the image using pre-trained models and PCA
```

### Sparse Representation
```python
def sparse_representation(descriptors, dictionary, fixed_length):
    # Create sparse representation of the extracted features
```

### Robust Fingerprinting
```python
def generate_fingerprints(video_path, model1, model2, model3, pca, dictionary, fixed_length):
    # Generate robust fingerprints for video frames
```

### Comparison
```python
def compare_signatures(test_signature, database_signatures, dynamic_threshold=False):
    # Compare test signature with database signatures using Euclidean distance and cosine similarity
```

### Liveness Detection
```python
def detect_liveness(video_path, EYE_AR_THRESH):
    # Detect liveness in a video by analyzing facial landmarks and eye aspect ratio
```

### Optical Flow Analysis
```python
def process_video_optical_flow(video_path):
    # Calculate the average optical flow magnitude in a video
```

### Texture Analysis
```python
def process_video_texture(video_path):
    # Analyze the texture of video frames
```

### JPEG Artifacts Analysis
```python
def jpeg_artifacts_analysis(file_path):
    # Analyze JPEG artifacts to detect compression inconsistencies
```

### Noise Analysis
```python
def noise_analysis(image):
    # Analyze the noise levels in an image to detect anomalies
```

### Shadow and Lighting Consistency
```python
def shadow_lighting_consistency(image):
    # Assess the consistency of shadows and lighting in an image
```

### Metadata Analysis
```python
def metadata_analysis(image_path):
    # Examine the metadata of an image for inconsistencies
```

### PRNU Analysis
```python
def prnu_analysis(image, prnu_patterns):
    # Detect inconsistencies in sensor noise patterns of images
```

### Frequency Domain Analysis
```python
def frequency_domain_analysis(image):
    # Analyze the frequency components of an image to identify tampering
```

### Color Filter Array (CFA) Analysis
```python
def cfa_analysis(image):
    # Perform CFA analysis to detect inconsistencies in the color filter array pattern
```

### AV Consistency
```python
def check_av_consistency(audio_features, optical_flow_score, texture_score):
    # Check the consistency between audio and video features
```

### Temporal Consistency
```python
def check_temporal_inconsistencies(video_path, temporal_threshold):
    # Check for temporal inconsistencies in video frames
```

### Deepfake Detection
```python
def deepfake_detection_check(file_path, deepfake_model_path, threshold):
    # Perform deepfake detection using a pre-trained model
```

### Logging and Results
```python
def write_log(message, level='info'):
    # Write log messages to file and console
```

```python
def write_results(*args):
    # Write analysis results to the results file
```

### Dynamic Thresholding
```python
def calculate_thresholds(database, config):
    # Calculate dynamic thresholds based on statistical analysis of feature distributions
```

### Loading and Saving Models
```python
def load_pca_model(pca_model_path):
    # Load the PCA model from a specified path
```

```python
def save_pca_model(pca_model, pca_model_path):
    # Save the PCA model to a specified path
```

```python
def load_dictionary(dictionary_path):
    # Load a dictionary from a specified path
```

```python
def save_database(database, database_path):
    # Save the database to a specified path
```

```python
def load_database(database_path):
    # Load the database from a specified path
```

---

## Contributing

Contributions are welcome! Please create a pull request or submit an issue for any feature requests or bug reports. I don't honestly know how much I'll be supporting this going forward, but feel free to email me at cjohnson@uwsp.edu if you have any questions or suggestions!

## License

This project is licensed under the MIT License - see https://opensource.org/license/MIT for details.
