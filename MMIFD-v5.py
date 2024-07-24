# Standard library imports
import concurrent.futures
import configparser
import csv
import ctypes
import io
import logging
import multiprocessing
import os
import random
import struct
import threading
import time
import warnings
from datetime import datetime

# Related third-party imports
import cv2
import dlib
import filetype
import h5py
import joblib
import json
import keras
import librosa
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import psutil
import pywt
import subprocess
import tensorflow as tf
from colorama import Fore, Style, init
from imutils import face_utils
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.saving import register_keras_serializable
from moviepy.editor import AudioFileClip
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
from scipy.spatial import distance as dist
from scipy.spatial.distance import cosine, euclidean
from skimage.feature import hog
from skimage.restoration import estimate_sigma
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, PCA, SparseCoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC, SVR
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects
from tqdm import tqdm

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize colorama
init(autoreset=True)

# Global variable for light direction model
light_direction_model = None

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging
log_file_path = config['Logging']['log_file_path']
result_file_path = config['Logging']['result_file_path']
logging_level = config['Logging']['logging_level']
logging.basicConfig(level=getattr(logging, logging_level.upper()),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path),
                              logging.StreamHandler()])
log_lock = threading.Lock()
results_lock = threading.Lock()

# Define timeout duration
timeout_duration = config.getint('Timeouts', 'timeout_duration')

class TimeoutException(Exception):
    pass

# Custom CastToFloat32 layer definition
@register_keras_serializable(package='Custom', name='CastToFloat32')
class CustomCastToFloat32(Layer):
    def __init__(self, **kwargs):
        super(CustomCastToFloat32, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        config = super(CustomCastToFloat32, self).get_config()
        return config

# Custom Adam optimizer definition
@register_keras_serializable(package='Custom', name='Adam')
class CustomAdam(Adam):
    def __init__(self, *args, **kwargs):
        super(CustomAdam, self).__init__(*args, **kwargs)

# Custom Functional definition
@register_keras_serializable(package='Custom', name='Functional')
class CustomFunctional(Model):
    pass
 
# Accuracy metrics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Loads ground truth labels from a CSV file for comparison with predicted labels. (For testing purposes when the provenance is known.)
def load_ground_truth_labels(csv_path):
    ground_truth = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip header row
            for rows in reader:
                filename = rows[0].strip('"')
                label = rows[1].strip()
                ground_truth[filename] = label
    else:
        write_log(f"Ground truth labels file {csv_path} not found. Proceeding without ground truth labels.", 'info')
    return ground_truth

# Computes and returns accuracy, precision, recall, and F1 score based on the provided results.
def report_accuracy(results):
    tp = results.count('TP')
    fp = results.count('FP')
    tn = results.count('TN')
    fn = results.count('FN')

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

# Calculates and logs threshold values for various scores using the data from the database.
def report_thresholds(database):
    texture_threshold = np.mean(database['texture_scores']) - np.std(database['texture_scores'])
    optical_flow_threshold = np.mean(database['optical_flow_magnitudes']) - np.std(database['optical_flow_magnitudes'])
    lighting_threshold = np.mean(database['lighting_consistency_scores']) + np.std(database['lighting_consistency_scores'])
    frequency_threshold = np.mean(database['frequency_domain_scores']) + np.std(database['frequency_domain_scores'])
    metadata_threshold = np.mean(database['metadata_scores'])
    prnu_threshold = np.mean(database['prnu_scores']) + np.std(database['prnu_scores'])
    jpeg_artifacts_threshold = np.mean(database['jpeg_artifacts_scores']) + np.std(database['jpeg_artifacts_scores'])
    noise_threshold = np.mean(database['noise_scores']) + np.std(database['noise_scores'])

    write_results(f"Texture Threshold: {texture_threshold}")
    write_results(f"Optical Flow Threshold: {optical_flow_threshold}")
    write_results(f"Lighting Consistency Threshold: {lighting_threshold}")
    write_results(f"Frequency Domain Threshold: {frequency_threshold}")
    write_results(f"Metadata Threshold: {metadata_threshold}")
    write_results(f"PRNU Threshold: {prnu_threshold}")
    write_results(f"JPEG Artifacts Threshold: {jpeg_artifacts_threshold}")
    write_results(f"Noise Threshold: {noise_threshold}")

    return (texture_threshold, optical_flow_threshold, lighting_threshold, 
            frequency_threshold, metadata_threshold, prnu_threshold, 
            jpeg_artifacts_threshold, noise_threshold)

# Logs the memory usage, CPU usage, and thread count of the current process. (Added after exrtreme debugging session and kept as a monument to my failures.)
def log_resource_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    cpu_usage = process.cpu_percent(interval=1)
    thread_count = process.num_threads()
    write_log(f"Memory usage: {memory_usage:.2f} MB")
    write_log(f"CPU usage: {cpu_usage:.1f}%")
    write_log(f"Thread count: {thread_count}")

# Executes a function with a specified timeout duration, returning the result or None if it times out. (Turns out data analytics can take an unreasonable amount of time.)
def process_with_timeout(func, args=(), kwargs={}, timeout_duration=20000):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_duration)
            return result
        except concurrent.futures.TimeoutError:
            write_log(f"Function {func.__name__} timed out after {timeout_duration} seconds", 'info')
            return None

# Logs a message with a specified severity level (info, debug, error).
def write_log(message, level='info'):
    levels = {'info': logging.INFO, 'debug': logging.DEBUG, 'error': logging.ERROR}
    color_levels = {'info': Style.RESET_ALL, 'debug': Style.RESET_ALL, 'error': Fore.RED + Style.BRIGHT}
    logging.log(levels.get(level, logging.INFO), color_levels.get(level, '') + message + Style.RESET_ALL)

# Writes results to a results file with a timestamp.
def write_results(*args):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    results_message = f"{timestamp} - {message}"
    try:
        with results_lock, open(result_file_path, "a") as results_file:
            results_file.write(results_message + "\n")
            results_file.flush()
    except Exception as e:
        write_log(f"Failed to write to results: {e}", 'error')

# Handles and logs errors, optionally exiting based on user input.      
def handle_error(message, exception):
    write_log(message, 'error')
    if exception:
        write_log(str(exception), 'error')
        import traceback
        traceback.print_exc()
    response = input("An error occurred. Do you want to continue? (yes/no): ").strip().lower()
    if response not in ('yes', 'y'):
        write_log("Exiting due to user request.")
        exit(1)

# Calculates the number of PCA components needed to explain the specified variance threshold.
def calculate_cumulative_explained_variance(feature_matrix, variance_threshold=0.95):
    pca = PCA()
    pca.fit(feature_matrix)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components needed to explain the desired variance
    n_components = np.argmax(cumulative_explained_variance >= variance_threshold) + 1
    write_log(f"Number of PCA components to explain {variance_threshold*100}% variance: {n_components}")
    
    # Optional: Plot the cumulative explained variance
    plt.plot(cumulative_explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.show()
    
    return n_components

# Loads an image from a given path using OpenCV.
def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

# Estimates the direction of light in a grayscale image using Sobel gradients.
def estimate_light_direction(image):
    try:
        # The image should already be in grayscale
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
        light_direction = np.mean(angle)
        return light_direction
    except Exception as e:
        handle_error(f"Error in estimating light direction: {e}", e)
        return None

# Preprocesses an image for PCA by resizing, converting to float, expanding dimensions, and applying VGG16 preprocessing.
def preprocess_image_for_pca(image):
    write_log("Preprocessing image for PCA", 'debug')
    image = cv2.resize(image, (224, 224))  # Resize image to 224x224 as required by VGG16
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    write_log(f"Preprocessed image shape: {image.shape}", 'debug')
    return image

# Converts an image to grayscale and resizes it for light direction analysis.
def preprocess_image_for_light_model(image, target_size=(128, 128)):
    # Check the number of channels in the image
    if len(image.shape) == 2:
        # Image is already in grayscale
        gray_image = image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # Image is in BGR format
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            # Image is in BGRA format
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    resized_image = cv2.resize(gray_image, target_size)  # Resize image to target size
    return resized_image

# Trains and saves a PCA model with the specified number of components using the provided feature matrix.
def train_pca_model(feature_matrix, n_components):
    write_log("Training PCA model.", 'debug')
    write_log(f"Feature matrix shape before PCA training: {feature_matrix.shape}", 'debug')

    if feature_matrix.ndim != 2:
        handle_error(f"Error: Expected 2D array, got {feature_matrix.ndim}D array instead", 'error')
        return None

    pca = PCA(n_components=n_components)
    try:
        pca.fit(feature_matrix)
        write_log(f"PCA model fit successfully with n_components={n_components}.", 'debug')
    except Exception as e:
        handle_error(f"Error fitting PCA model: {e}", e)
        return None

    try:
        joblib.dump(pca, pca_model_path)
        write_log(f"PCA model saved to {pca_model_path}.", 'debug')
    except Exception as e:
        handle_error(f"Error saving PCA model: {e}", e)
        return None

    return pca

# Trains and saves a light direction model using HOG features and a database of light directions.
def train_light_direction_model_with_db(light_directions_db, target_size=(128, 128)):
    X = []
    y = []

    # HOG parameters
    pixels_per_cell = (16, 16)
    cells_per_block = (2, 2)
    orientations = 9

    # Calculate expected feature size based on target size and HOG parameters
    expected_feature_size = (
        (target_size[0] // pixels_per_cell[0] - 1) *
        (target_size[1] // pixels_per_cell[1] - 1) *
        cells_per_block[0] *
        cells_per_block[1] *
        orientations
    )

    for file_path, light_direction in light_directions_db.items():
        image = load_image(file_path)
        try:
            image = preprocess_image_for_light_model(image, target_size)
            if image is not None:
                features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block, block_norm='L2-Hys')

                if features.size == expected_feature_size:  # Ensure features are of expected size
                    X.append(features)
                    y.append(light_direction)
                else:
                    handle_error(f"Inconsistent feature shape for {file_path}: {features.size}, expected {expected_feature_size}", 'error')
            else:
                handle_error(f"Preprocessing failed for {file_path}", 'error')
        except Exception as e:
            handle_error(f"Error processing {file_path}: {e}", e)

    write_log(f"Training light direction model with {len(X)} samples.", 'debug')

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid training data found for light direction model.")

    X = np.array(X)
    y = np.array(y)

    model = SVR()
    model.fit(X, y)

    joblib.dump(model, light_direction_model_path)
    write_log(f"Light direction model saved to {light_direction_model_path}.", 'debug')
    return model

# Builds a database of light directions from authentic images in the specified directory.
def build_light_direction_database(authentic_images_directory, target_size=(128, 128)):
    light_directions_db = {}
    for file_name in os.listdir(authentic_images_directory):
        file_path = os.path.join(authentic_images_directory, file_name)
        if os.path.isfile(file_path):
            image = load_image(file_path)
            if image is not None:
                try:
                    image = preprocess_image_for_light_model(image, target_size)
                    light_direction = estimate_light_direction(image)
                    light_directions_db[file_path] = light_direction
                except Exception as e:
                    handle_error(f"Failed to estimate light direction for {file_path}: {e}", e)
    return light_directions_db

# Loads and returns a previously saved light direction model.
def load_light_direction_model(light_direction_model_path):
    try:
        model = joblib.load(light_direction_model_path)
        write_log("Light direction model loaded successfully.", 'info')
        return model
    except Exception as e:
        handle_error(f"Failed to load light direction model: {e}", e)
        return None

# Extracts features from an image using three models, combines them, and applies PCA if provided.
def extract_features(image, model1, model2, model3, fixed_length, pca):
    try:
        write_log(f"Extracting features for image with shape: {image.shape}")
        preprocessed_image = preprocess_image(image)
        write_log(f"Preprocessed image shape: {preprocessed_image.shape}")
        
        features1 = model1.predict(preprocessed_image).flatten()
        write_log(f"Shape of features1 from model1: {features1.shape}")
        features2 = model2.predict(preprocessed_image).flatten()
        write_log(f"Shape of features2 from model2: {features2.shape}")
        features3 = model3.predict(preprocessed_image).flatten()
        write_log(f"Shape of features3 from model3: {features3.shape}")

        combined_features = np.concatenate([features1, features2, features3])
        write_log(f"Shape of combined_features: {combined_features.shape}")

        if pca is not None:
            combined_features = pca.transform([combined_features])[0]
            write_log(f"Shape of combined_features after PCA: {combined_features.shape}")

        # Ensure the combined_features length matches the fixed_length
        if fixed_length is not None:
            if combined_features.shape[0] < fixed_length:
                combined_features = np.pad(combined_features, (0, fixed_length - combined_features.shape[0]), 'constant')
            else:
                combined_features = combined_features[:fixed_length]
        return combined_features
    except Exception as e:
        handle_error(f"Error extracting features: {e}", e)
        return None

# Preprocesses an image for feature extraction by resizing, converting to RGB if needed, and applying VGG16 preprocessing.
def preprocess_image(image):
    write_log(f"Preprocessing image {image}", 'debug')
    image = cv2.resize(image, (224, 224))  # Resize image to 224x224 as required by VGG16
    if len(image.shape) == 2:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    write_log(f"Preprocessed image shape: {image.shape}", 'debug')
    return image

# Extracts features from all images in a directory using the specified models and PCA.
def extract_features_from_directory(directory, model1, model2, model3, fixed_length, pca=None):
    image_descriptors = []
    image_paths = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if is_image_file(os.path.join(directory, file_name))]

    with tqdm(total=len(image_paths), desc="Preprocessing and Extracting Features") as pbar:
        for file_path in image_paths:
            try:
                write_log(f"Processing file: {file_path}")
                image = load_image(file_path)

                if image is not None:
                    descriptors = extract_features(image, model1, model2, model3, fixed_length, pca)
                    
                    if descriptors is not None:
                        image_descriptors.append(descriptors)
                pbar.update(1)
            except Exception as e:
                handle_error(f"Error extracting features from {file_path}: {str(e)}", e)
                continue

    return np.array(image_descriptors)

# Computes the sparse representation of descriptors using a given dictionary, ensuring a fixed length.
def sparse_representation(descriptors, dictionary, fixed_length):
    if descriptors is None:
        handle_error("Error: Descriptors are None, skipping sparse representation.", 'error')
        return None

    if descriptors.shape[0] != fixed_length:
        handle_error(f"Error: Descriptors length mismatch. Expected {fixed_length}, but got {descriptors.shape[0]}.", 'error')
        return None

    write_log(f"Starting sparse representation. Descriptors shape: {descriptors.shape}", 'debug')
    try:
        coder = SparseCoder(dictionary=dictionary, transform_n_nonzero_coefs=10, transform_algorithm='omp')
        sparse_rep = coder.transform(descriptors.reshape(1, -1))  # Ensure descriptors is 2D
        sparse_rep_flat = sparse_rep.flatten()
        if sparse_rep_flat.shape[0] < fixed_length:
            sparse_rep_flat = np.pad(sparse_rep_flat, (0, fixed_length - sparse_rep_flat.shape[0]), 'constant')
        else:
            sparse_rep_flat = sparse_rep_flat[:fixed_length]
        write_log(f"Sparse representation shape: {sparse_rep_flat.shape}", 'debug')
        return sparse_rep_flat
    except Exception as e:
        handle_error(f"Error in sparse representation: {e}", e)
        return None

# Validates the provided dictionary for non-emptiness and correct type.
def validate_dictionary(dictionary):
    if dictionary is None or not isinstance(dictionary, np.ndarray) or dictionary.size == 0:
        return False
    return True

# Validates the dictionary against the PCA model, ensuring matching dimensions.  
def validate_dictionary_with_pca(dictionary, pca):
    if dictionary is None or not isinstance(dictionary, np.ndarray) or dictionary.size == 0:
        handle_error("Dictionary validation failed: Dictionary is None, not a ndarray, or empty.", 'error')
        return False

    if dictionary.shape[1] != pca.n_components_:
        handle_error(f"Dictionary validation failed: Expected {pca.n_components_} components, but got {dictionary.shape[1]}.", 'error')
        return False
    
    write_log("Dictionary validation succeeded: Dictionary is consistent with PCA components.")
    return True

# Loads and validates the PCA model, checking for necessary attributes.
def validate_pca_model(pca_model_path):
    try:
        pca = joblib.load(pca_model_path)
        if pca is not None and hasattr(pca, 'n_components_') and pca.n_components_ > 0:
            return True
    except Exception as e:
        handle_error(f"Failed to validate PCA model: {e}", e)
    return False

# Loads and validates the light direction model.
def validate_light_model(light_direction_model_path):
    try:
        model = joblib.load(light_direction_model_path)
        if model is not None:
            return True
    except Exception as e:
        handle_error(f"Failed to validate light direction model: {e}", e)
    return False

def calculate_dynamic_threshold_score(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate brightness (mean pixel value)
    brightness = np.mean(gray_image)

    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(gray_image)

    # Calculate edge sharpness (mean gradient magnitude)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    edge_sharpness = np.mean(gradient_magnitude)

    # Combine the calculated metrics into a dynamic threshold score
    # Here, we can use a simple weighted sum. Adjust the weights as needed.
    dynamic_threshold_score = (0.4 * brightness) + (0.4 * contrast) + (0.2 * edge_sharpness)

    return dynamic_threshold_score

# Validates the database structure and contents for expected keys.
def validate_database(database):
    expected_keys = [
        'sparse_representations', 
        'video_fingerprints', 
        'optical_flow_magnitudes', 
        'lighting_consistency_scores', 
        'frequency_domain_scores', 
        'prnu_patterns', 
        'metadata_scores', 
        'jpeg_artifacts_scores', 
        'noise_scores',
        'texture_scores'
    ]
    
    if database is None or not isinstance(database, dict):
        return False
    
    for key in expected_keys:
        if key not in database or not database[key]:
            handle_error(f"Database validation failed: Missing or empty key - {key}", 'error')
            return False
    
    return True

# Checks for consistency in the shape of all provided fingerprints.
def validate_fingerprints(fingerprints):
    base_shape = fingerprints[0].shape
    for i, fp in enumerate(fingerprints):
        if fp.shape != base_shape:
            handle_error(f"Mismatch at index {i}: Expected {base_shape}, got {fp.shape}", 'error')
            return False
    return True

# Generates and validates fingerprints from a video using feature extraction and sparse representation.
def generate_fingerprints(video_path, model1, model2, model3, pca, dictionary, fixed_length):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fingerprints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.all((gray_frame == 0)):
            write_log(f"Frame {frame_count} in {video_path} is completely black, skipping", 'debug')
            frame_count += 1
            continue

        descriptors = extract_features(frame, model1, model2, model3, fixed_length=fixed_length, pca=pca)
        if descriptors is None:
            write_log(f"No descriptors found for frame {frame_count} in {video_path}", 'debug')
            frame_count += 1
            continue
        sparse_rep = sparse_representation(descriptors, dictionary, fixed_length)
        fingerprints.append(sparse_rep)
        frame_count += 1

    cap.release()
    if not fingerprints:
        write_log(f"No valid fingerprints found for video {video_path}. Skipping video.", 'debug')
        return None
    if not validate_fingerprints(fingerprints):
        raise ValueError("Fingerprint dimension mismatch detected.")
    return fingerprints

# Compares a test signature against database signatures using Euclidean distance and cosine similarity.
def compare_signatures(test_signature, database_signatures, dynamic_threshold=False):
    write_log(f"Comparing test signature of shape {test_signature.shape} with database signatures", 'debug')
    test_signature = normalize_vector(test_signature)
    distances = []
    cosine_similarities = []
    for idx, ref_signature in enumerate(database_signatures):
        ref_signature = normalize_vector(ref_signature)
        if ref_signature.shape != test_signature.shape:
            handle_error(f"Skipping due to shape mismatch: {ref_signature.shape} vs {test_signature.shape}", 'error')
            continue
        distance = euclidean(test_signature, ref_signature)
        cosine_sim = cosine(test_signature, ref_signature)
        distances.append(distance)
        cosine_similarities.append(cosine_sim)

    if not distances:
        write_log("No valid distances calculated. Returning None values for comparison results.")
        return None, None, None, None

    if dynamic_threshold:
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        dynamic_threshold_value = mean_distance + dynamic_threshold_factor * std_distance
        write_results(f"Dynamic Threshold: {dynamic_threshold_value}")
    else:
        dynamic_threshold_value = None

    is_authentic_dynamic = any(distance < dynamic_threshold_value for distance in distances) if dynamic_threshold_value is not None else None
    is_authentic_static = any(distance < static_threshold_value for distance in distances)

    min_distance = min(distances) if distances else None
    max_cosine_sim = max(cosine_similarities) if cosine_similarities else None

    write_results(f"Is Authentic (Dynamic Threshold): {is_authentic_dynamic}, Min Distance: {min_distance}")
    write_results(f"Is Authentic (Static Threshold): {is_authentic_static}, Min Distance: {min_distance}")
    write_results(f"Max Cosine Similarity: {max_cosine_sim}")

    return is_authentic_dynamic, is_authentic_static, min_distance, max_cosine_sim

# Checks if a file is a valid image.
def is_image_file(file_path):
    try:
        img = cv2.imread(file_path)
        return img is not None
    except Exception as e:
        return False

# Checks if a file is a valid video.
def is_video_file(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception as e:
        return False

# Calculates the optical flow between two video frames using the Farneback method.
def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Using the Farneback method for optical flow calculation
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if mag.size == 0:
        write_log("Optical flow magnitude size is zero, skipping analysis to avoid division by zero.")
        return 0, 0
    return mag, ang

# Analyzes test results for a file, computing metrics such as accuracy, precision, anomalies, and reliability.
def analyze_new_tests(results, file_path, thresholds):
    try:
        write_log(f"Analyzing new tests for {file_path}. Results keys: {list(results.keys())}")

        # Compute accuracy as the ratio of authentic results to total results
        authentic_results = sum([
            results['is_authentic_lighting'],
            results['is_authentic_frequency_domain'],
            results['is_authentic_prnu'],
            results['is_authentic_metadata'],
            results['is_authentic_jpeg_artifacts'],
            results['is_authentic_noise']
        ])
        total_tests = 6  # Total number of tests conducted
        accuracy = authentic_results / total_tests
        results['accuracy'] = accuracy
        write_log(f"Accuracy for {file_path}: {accuracy}")

        # Compute precision as the ratio of true positives to all positives
        true_positives = authentic_results  # Assuming all authentic results are true positives
        false_positives = total_tests - authentic_results  # Assuming remaining are false positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        results['precision'] = precision
        write_log(f"Precision for {file_path}: {precision}")

        # Check and log anomalies
        anomaly_scores = [
            results['lighting_consistency_score'],
            results['frequency_domain_score'],
            results['prnu_score'],
            results['metadata_score'],
            results['jpeg_artifacts_score'],
            results['noise_score']
        ]
        high_anomalies = [score for score in anomaly_scores if score > thresholds.get('high_anomaly', 0.9)]
        num_high_anomalies = len(high_anomalies)
        results['num_high_anomalies'] = num_high_anomalies
        write_log(f"Number of high anomaly scores for {file_path}: {num_high_anomalies}")
        if num_high_anomalies > 1:  # Example threshold for concern
            write_log(f"Warning: High number of anomalies detected for {file_path}")
            perform_detailed_analysis(file_path, "High anomalies")

        # Compute reliability as the inverse of the average anomaly score
        average_anomaly_score = sum(anomaly_scores) / total_tests
        reliability = 1 / average_anomaly_score if average_anomaly_score > 0 else 0
        results['reliability'] = reliability
        write_log(f"Reliability for {file_path}: {reliability}")

        write_log(f"Analysis complete for {file_path}")

    except Exception as e:
        handle_error(f"Error in new tests analysis for {file_path}: {e}", e)
        raise

# Flags a file for further review with a specified reason. (Or might, in the future.)
def flag_for_review(file_path, reason):
    write_results(f"Flagging {file_path} for review: {reason}")

# Escalates an issue with a specified reason. (Or might, in the future.)
def escalate_issue(file_path, reason):
    write_results(f"Escalating issue for {file_path}: {reason}")

# More detailed testing with a specified reason. (Or might, in the future.)
def perform_detailed_analysis(file_path, reason):
    write_results(f"Performing detailed analysis for {file_path}: {reason}")

# Processes a video to calculate the average optical flow magnitude.      
def process_video_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    optical_flow_magnitudes = []

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames // 2, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, next_frame = cap.read()
            if not ret:
                break
            if frame_count % 2 == 0:  
                try:
                    mag, _ = calculate_optical_flow(prev_frame, next_frame)
                    if mag is not None:
                        avg_mag = np.mean(mag)
                        optical_flow_magnitudes.append(avg_mag)
                except Exception as e:
                    handle_error(f"Error calculating optical flow for frame {frame_count} in {video_path}: {e}", e)
                pbar.update(1)
            prev_frame = next_frame
            frame_count += 1

    cap.release()

    if not optical_flow_magnitudes:
        write_log(f"No valid optical flow magnitudes for video {video_path}. Returning 0.0.")
        return 0.0

    avg_optical_flow_magnitude = np.mean(optical_flow_magnitudes)
    write_log(f'Average Optical Flow Magnitude: {avg_optical_flow_magnitude}')
    return avg_optical_flow_magnitude

# Processes a video to calculate the average texture score.
def process_video_texture(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    texture_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        score = texture_analysis(frame)
        if score is not None:
            texture_scores.append(score)
        frame_count += 1

    cap.release()
    avg_score = np.mean(texture_scores)
    write_log(f'Average Texture Score: {avg_score}')
    return avg_score

# Analyzes the texture of an image using the variance of the Laplacian.
def texture_analysis(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        handle_error(f"Failed to convert image to grayscale: {e}", e)
        return None
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Detects liveness in a video by analyzing eye aspect ratio and landmark distances.
def detect_liveness(video_path, EYE_AR_THRESH):
    #EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    write_log(f"Loading facial landmark predictor for video: {video_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        write_log(f"Error opening video file: {video_path}", 'error')
        write_results(f"Error opening video file: {video_path}")
        return None

    blink_count = 0
    total_frames = 0
    landmark_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        total_frames += 1

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                blink_count += 1

            # Calculate distances between key landmarks
            try:
                distances = calculate_landmark_distances(shape)
                if distances:
                    landmark_distances.append(distances)
                    write_log(f"Frame {total_frames}: Landmarks detected and distances calculated.")
                else:
                    write_log(f"Frame {total_frames}: Error in calculating distances.")
            except IndexError as e:
                write_log(f"IndexError in landmark distance calculation: {str(e)}", 'error')
                return None
            except Exception as e:
                write_log(f"Error in landmark distance calculation: {str(e)}", 'error')
                return None

    cap.release()

    # Check for consistency in landmark distances
    if not check_consistency(landmark_distances):
        write_log(f"Video {video_path} has inconsistent facial landmarks. Possible face-warping artifact detected.")
        write_results(f"Video {video_path} has inconsistent facial landmarks. Possible face-warping artifact detected.")
        return False

    if blink_count > EYE_AR_CONSEC_FRAMES:
        write_log(f"Video {video_path} is likely live. Blink count: {blink_count}")
        write_results(f"Video {video_path} is likely live. Blink count: {blink_count}")
        return True
    else:
        write_log(f"Video {video_path} is likely a forgery. Blink count: {blink_count}")
        write_results(f"Video {video_path} is likely a forgery. Blink count: {blink_count}")
        return False

# Calculates distances between key facial landmarks.
def calculate_landmark_distances(shape):
    try:
        distances = []
        for (i, (x, y)) in enumerate(shape):
            for (j, (x2, y2)) in enumerate(shape[i+1:]):
                distance = np.linalg.norm(np.array((x, y)) - np.array((x2, y2)))
                distances.append(distance)
        return distances
    except IndexError as e:
        write_log(f"IndexError in calculate_landmark_distances: {str(e)}", 'error')
        return []
    except Exception as e:
        write_log(f"Error in calculate_landmark_distances: {str(e)}", 'error')
        return []

# Checks the consistency of landmark distances within a specified tolerance.
def check_consistency(landmark_distances):
    try:
        for distances in landmark_distances:
            if len(distances) != len(landmark_distances[0]):
                write_log("Inconsistent number of distances detected.", 'error')
                return False
        return True
    except IndexError as e:
        write_log(f"IndexError in check_consistency: {str(e)}", 'error')
        return False
    except Exception as e:
        write_log(f"Error in check_consistency: {str(e)}", 'error')
        return False

# Calculates the eye aspect ratio from eye landmarks.
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Normalizes a vector to unit length.
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        write_log("Vector norm is zero, skipping normalization")
        return vector
    return vector / norm

# Loads a database from a specified path.
def load_database(database_path):
    try:
        db = np.load(database_path, allow_pickle=True).item()
        write_log(f"Loaded Database with {len(db['sparse_representations'])} sparse representations and {len(db['video_fingerprints'])} video fingerprints")
        return db
    except FileNotFoundError:
        write_log(f"Database file not found: {database_path}")
        return None

# Saves a PCA model to a specified path.
def save_pca_model(pca_model, pca_model_path):
    joblib.dump(pca_model, pca_model_path)

# Loads a PCA model from a specified path.
def load_pca_model(pca_model_path):
    try:
        pca = joblib.load(pca_model_path)
        write_log("PCA model loaded successfully.")
        return pca
    except Exception as e:
        write_log(f"Failed to load PCA model: {e}")
        return None

# Check if the given file is a JPEG by reading its file signature.
def is_jpeg(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_signature = f.read(2)
        return file_signature == b'\xff\xd8'
    except Exception as e:
        write_log(f"Error checking JPEG signature: {e}")
        return False

# Extract quantization tables from the JPEG file's EXIF data.
def extract_quantization_tables(file_path):
    try:
        pil_image = Image.open(file_path)
        exif_data = pil_image.getexif()
        if exif_data:
            quantization_tables = exif_data.get(0x0200)
            return quantization_tables
        else:
            write_log(f"No EXIF data found in image {file_path}")
            return None
    except Exception as e:
        write_log(f"Error extracting quantization tables: {e}")
        return None
        
# Process each image in the suspect corpus directory and analyze JPEG artifacts.
def jpeg_artifacts_analysis(file_path):
    # Log the file path and its type
    write_log(f"File path received in jpeg_artifacts_analysis: {file_path} (type: {type(file_path)})")

    try:
        # Attempt to extract quantization tables from EXIF data
        img = Image.open(file_path)
        exif_data = img._getexif()
        
        if exif_data:
            for tag, value in exif_data.items():
                if isinstance(tag, int):
                    decoded = TAGS.get(tag, tag)
                    if decoded == 'JPEGQTables':
                        quantization_tables = value
                        if quantization_tables:
                            write_log(f"Quantization tables found in EXIF data of {file_path}")
                            return analyze_quantization_tables(quantization_tables)
        
        # If no quantization tables in EXIF, calculate from the image
        img_arr = np.array(img)
        quantization_tables = calculate_quantization_tables(img_arr)
        write_log(f"Quantization tables calculated from image data of {file_path}")
        return analyze_quantization_tables(quantization_tables)
    except Exception as e:
        write_log(f"Failed to analyze JPEG artifacts: {e}")
        return float('inf')

# Analyze the Q tables if we have them
def analyze_quantization_tables(quantization_tables):
    high_freq_coeff_count = 0
    total_tables = len(quantization_tables)
    
    if total_tables == 0:
        return float('inf')  # Prevent division by zero
    
    for table in quantization_tables:
        high_freq_coeff_count += sum(1 for coeff in table if coeff > 50)
    
    artifacts_score = high_freq_coeff_count / total_tables
    return artifacts_score

# If we can't extract the Q tables, calculated them in DCT
def calculate_quantization_tables(img_arr):
    write_log("Calculating quantization tables from image array")
    
    # Ensure the image is in grayscale
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        gray_image = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img_arr

    # Initialize the quantization tables
    quantization_tables = []

    # Divide the image into 8x8 blocks and compute DCT for each block
    for i in range(0, gray_image.shape[0], 8):
        for j in range(0, gray_image.shape[1], 8):
            block = gray_image[i:i+8, j:j+8]

            # Check if the block is 8x8
            if block.shape == (8, 8):
                # Compute the DCT
                dct_block = cv2.dct(np.float32(block))

                # Compute the quantization table
                quantization_table = np.round(dct_block / 8) * 8  # Simplistic quantization table approximation
                quantization_tables.append(quantization_table)

    quantization_tables = np.array(quantization_tables)

    # Simplify to a single representative table (e.g., average)
    representative_table = np.mean(quantization_tables, axis=0)
    
    write_log(f"Quantization table calculated: {representative_table}")

    return representative_table

# Analyzes noise in an image using wavelet-based noise estimation.
def noise_analysis(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sigma_est = estimate_sigma(gray, channel_axis=-1)  # Replace multichannel with channel_axis
        
        if isinstance(sigma_est, (list, np.ndarray)):
            sigma_est = np.mean(sigma_est)
        
        noise_level = sigma_est ** 2
        return noise_level
    except ImportError:
        write_log("PyWavelets is not installed. Please ensure it is installed in order to use this function.")
    except Exception as e:
        write_log(f"Failed to analyze noise: {e}")
    return None

# Analyzes the consistency of shadows and lighting in an image using HOG features and a light direction model.
def shadow_lighting_consistency(image):
    try:
        if light_direction_model is None:
            handle_error("Light direction model is not loaded. Skipping shadow and lighting consistency analysis.", 'error')
            return None

        preprocessed_image = preprocess_image_for_light_model(image)  # Ensure consistent preprocessing
        features = hog(preprocessed_image, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

        model_light_directions = light_direction_model.predict([features])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
        light_directions = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    light_directions.append(angle)

        if light_directions:
            light_directions = np.array(light_directions)
            combined_light_directions = np.concatenate((light_directions, model_light_directions))
        else:
            combined_light_directions = model_light_directions

        consistency_score = np.std(combined_light_directions)
        return consistency_score
    except Exception as e:
        handle_error(f"Failed to analyze shadow and lighting consistency: {e}", e)
        return None

# Analyzes the metadata of an image, checking for expected EXIF tags.
def metadata_analysis(image_path):
    from PIL.ExifTags import TAGS as TiffTags  # Ensure import within the function scope
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            write_log(f"No EXIF data found for image: {image_path}")
            return None  # Return 0 to indicate no metadata

        metadata_score = 0.0
        for tag, value in exif_data.items():
            if tag in TiffTags:
                tag_name = TiffTags[tag]
                write_log(f"tag: {tag_name} ({tag}) - value: {value}")

                # Customize scoring logic based on metadata attributes
                if tag_name in ["Make", "Model", "DateTime"]:
                    if value:  # Adjust conditions based on requirements
                        metadata_score += 0.2
        
        metadata_score = min(metadata_score, 1.0)
        write_log(f"Metadata score for image {image_path}: {metadata_score}")
        return metadata_score
    except Exception as e:
        write_log(f"Failed to analyze metadata: {e}")
        return None  # Return 0 to indicate analysis failure

# Analyzes the frequency domain of an image using FFT and calculates the high-frequency ratio.  
def frequency_domain_analysis(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)

        rows, cols = gray.shape
        crow, ccol = rows // 2 , cols // 2
        high_freq_magnitude = np.sum(magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30])
        total_magnitude = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_magnitude / total_magnitude

        frequency_score = high_freq_ratio
        return frequency_score
    except Exception as e:
        write_log(f"Failed to analyze frequency domain: {e}")
    return None

# Extracts the Photo-Response Non-Uniformity (PRNU) pattern from an image.
def extract_prnu(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        prnu = gray - cv2.GaussianBlur(gray, (3, 3), 0)
        prnu /= np.std(prnu)
        prnu = cv2.resize(prnu, (256, 384))  # Resize PRNU to match database patterns
        return prnu
    except Exception as e:
        write_log(f"Failed to extract PRNU pattern: {e}")
        return None

# Analyzes an image for PRNU patterns, comparing against known patterns.
def prnu_analysis(image, prnu_patterns):
    try:
        write_log("Starting PRNU analysis")

        if not prnu_patterns:
            write_log("PRNU analysis skipped: prnu_patterns are empty.")
            return None

        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Resize the gray image to match PRNU pattern dimensions
        prnu_shape = prnu_patterns[0].shape
        gray_image = cv2.resize(gray_image, (prnu_shape[1], prnu_shape[0]))

        # Normalize the grayscale image
        gray_image = gray_image.astype(np.float32) / 255.0
        write_log(f"Gray image shape: {gray_image.shape}, dtype: {gray_image.dtype}")

        # Denoise the image to obtain the noise residual
        denoised_image = cv2.fastNlMeansDenoising(gray_image.astype(np.uint8), None, 10, 7, 21)
        noise_residual = gray_image - denoised_image.astype(np.float32)

        # Standardize the noise residual
        std_noise_residual = np.std(noise_residual)
        if std_noise_residual == 0:
            write_log("Standard deviation of noise residual is zero, cannot normalize.")
            return None
        noise_residual /= std_noise_residual
        write_log(f"Noise residual shape: {noise_residual.shape}, dtype: {noise_residual.dtype}")

        # Resize noise_residual to match PRNU pattern dimensions
        noise_residual = cv2.resize(noise_residual, (prnu_shape[1], prnu_shape[0]))

        # Compute correlations with PRNU patterns
        correlations = []
        for prnu in prnu_patterns:
            if prnu.shape != noise_residual.shape:
                write_log(f"Shape mismatch: noise_residual shape {noise_residual.shape}, prnu shape {prnu.shape}")
                continue
            correlation = np.corrcoef(noise_residual.flatten(), prnu.flatten())[0, 1]
            correlations.append(correlation)

        if not correlations:
            write_log("PRNU analysis failed: No valid correlations found due to shape mismatches.")
            return None

        # Calculate the mean correlation score
        prnu_score = np.mean(correlations)
        write_log(f"PRNU analysis complete with score: {prnu_score:.12f}")

        return prnu_score
    except cv2.error as e:
        handle_error(f"Error in PRNU analysis: OpenCV error - {e}", e)
        return None
    except Exception as e:
        handle_error(f"Error in PRNU analysis: {e}", e)
        return None

# Checks if all provided arrays have the same shape.
def check_array_shapes(arrays):
    base_shape = arrays[0].shape
    for array in arrays:
        if array.shape != base_shape:
            return False
    return True

# Processes a file (image or video) to extract features, analyze various aspects, and log the results.
def get_file_type(file_path):
    # Check if the path is a file
    if not os.path.isfile(file_path):
        write_log(f"Path is not a file: {file_path}", 'error')
        return None
    
    try:
        kind = filetype.guess(file_path)
        if kind is None:
            write_log(f"Unable to determine file type for {file_path}", 'error')
            return None
        return kind.mime
    except PermissionError as e:
        write_log(f"Permission denied: {file_path}. Error: {e}", 'error')
        return None
    except Exception as e:
        write_log(f"Error guessing file type: {file_path}. Error: {e}", 'error')
        return None

def process_file(file_path, dictionary, database, model1, model2, model3, pca, fixed_length, thresholds):
    file_type = get_file_type(file_path)
    
    if file_type is None:
        logging.warning(f"Unable to determine file type for {file_path}")
        return
    
    try:
        write_log(f"Processing file: {file_path}")
        log_resource_usage()  # Log resource usage at the beginning of file processing

        results = {}  # Initialize results dictionary
        predicted_label = None  # Initialize predicted_label

        write_log(f"Threshold values before processing: {thresholds}", 'debug')
        write_results(f"Threshold values before processing: {thresholds}")

        if is_image_file(file_path):
            write_log(f"Loaded image: {file_path}")
            image = load_image(file_path)
            write_log(f"Calling extract_features with pca: {pca is not None}")
            descriptors = process_with_timeout(extract_features, args=(image, model1, model2, model3, fixed_length, pca), timeout_duration=600)
            write_log(f"Descriptors for image {file_path}: {descriptors[:10] if descriptors is not None else 'None'}")  # Log first 10 descriptors
            log_resource_usage()  # Log resource usage after feature extraction

            if descriptors is not None:
                try:
                    sparse_rep = process_with_timeout(sparse_representation, args=(descriptors, dictionary, fixed_length), timeout_duration=600)
                    write_log(f"Sparse representation for image {file_path}: {sparse_rep[:10]}")
                except Exception as e:
                    handle_error(f"Error in sparse representation for {file_path}: {e}", e)
                    return None

                write_log(f"Comparing signatures for image {file_path}")
                is_authentic_dynamic, is_authentic_static, distance, cosine_sim = process_with_timeout(compare_signatures, args=(sparse_rep, database['sparse_representations'], thresholds['dynamic_threshold_factor']), timeout_duration=600)
                write_log(f"Comparison results for image {file_path}: Dynamic: {is_authentic_dynamic}, Static: {is_authentic_static}, Distance: {distance}, Cosine Sim: {cosine_sim}")
                log_resource_usage()  # Log resource usage after signature comparison

                results['is_authentic_static'] = is_authentic_static
                results['is_authentic_dynamic'] = is_authentic_dynamic
                results['distance'] = distance
                results['cosine_sim'] = cosine_sim

                # Determine the final predicted label
                if is_authentic_static or is_authentic_dynamic:
                    predicted_label = 'authentic'
                else:
                    predicted_label = 'forgery'
            else:
                write_log(f'No descriptors found or descriptor shape mismatch for: {file_path}')
                write_results(f'File {file_path} processing failed due to descriptor shape mismatch.')
                return None
            # Individual tests
            write_log(f"Running shadow lighting consistency for image {file_path}")
            lighting_consistency_score = process_with_timeout(shadow_lighting_consistency, args=(image,), timeout_duration=600)
            if lighting_consistency_score is None:
                lighting_consistency_score = float('inf')  
            results['lighting_consistency_score'] = lighting_consistency_score
            write_log(f"Lighting consistency score for image {file_path}: {lighting_consistency_score}")
            results['is_authentic_lighting'] = lighting_consistency_score < thresholds.get('lighting_consistency', float('inf'))

            write_log(f"Running frequency domain analysis for image {file_path}")
            frequency_domain_score = process_with_timeout(frequency_domain_analysis, args=(image,), timeout_duration=600)
            if frequency_domain_score is None:
                frequency_domain_score = float('inf')  
            results['frequency_domain_score'] = frequency_domain_score
            write_log(f"Frequency domain score for image {file_path}: {frequency_domain_score}")
            results['is_authentic_frequency_domain'] = frequency_domain_score < thresholds.get('frequency_domain', float('inf'))

            write_log(f"Running PRNU analysis for image {file_path}")
            prnu_score = process_with_timeout(prnu_analysis, args=(image, database.get('prnu_patterns', [])), timeout_duration=600)
            if prnu_score is None:
                prnu_score = float('inf')  
            results['prnu_score'] = prnu_score
            write_log(f"PRNU score for image {file_path}: {prnu_score}")
            results['is_authentic_prnu'] = prnu_score > thresholds.get('prnu', float('inf'))

            write_log(f"Running metadata analysis for image {file_path}")
            metadata_score = process_with_timeout(metadata_analysis, args=(file_path,), timeout_duration=600)
            if metadata_score is None:
                metadata_score = 0  
            results['metadata_score'] = metadata_score
            write_log(f"Metadata score for image {file_path}: {metadata_score}")
            results['is_authentic_metadata'] = metadata_score < thresholds.get('metadata', 0)

            write_log(f"Running JPEG artifacts analysis for image {file_path}")
            jpeg_artifacts_score = process_with_timeout(jpeg_artifacts_analysis, args=(file_path,), timeout_duration=600)
            if jpeg_artifacts_score is None:
                jpeg_artifacts_score = float('inf')  
            results['jpeg_artifacts_score'] = jpeg_artifacts_score
            write_log(f"JPEG artifacts score for image {file_path}: {jpeg_artifacts_score}")
            results['is_authentic_jpeg_artifacts'] = jpeg_artifacts_score < thresholds.get('jpeg_artifacts', float('inf'))

            write_log(f"Running noise analysis for image {file_path}")
            noise_score = process_with_timeout(noise_analysis, args=(image,), timeout_duration=600)
            if noise_score is None:
                noise_score = float('inf')  
            results['noise_score'] = noise_score
            write_log(f"Noise analysis score for image {file_path}: {noise_score}")
            results['is_authentic_noise'] = noise_score < thresholds.get('noise', float('inf'))

            write_log(f"Running CFA analysis for image {file_path}")
            cfa_score = process_with_timeout(cfa_analysis, args=(image,), timeout_duration=600)
            if cfa_score is None:
                cfa_score = float('inf')  
            results['cfa_score'] = cfa_score
            write_log(f"CFA score for image {file_path}: {cfa_score}")
            results['is_authentic_cfa'] = cfa_score < thresholds.get('cfa_threshold', float('inf'))
            
            write_log(f"Analyzing new tests for image {file_path}")
            process_with_timeout(analyze_new_tests, args=(results, file_path, database), timeout_duration=600)

            # Log all results in the results file
            write_results(f"Results for image {file_path}:")
            for key, value in results.items():
                write_results(f"  {key}: {value}")

            # Final verdict
            if results.get('is_authentic_static'):
                write_results(f'File {file_path} is likely authentic with static threshold. Distance: {results["distance"]}, Cosine Similarity: {results["cosine_sim"]}')
            else:
                write_results(f'File {file_path} is likely a forgery with static threshold. No match found.')

            if results.get('is_authentic_dynamic'):
                write_results(f'File {file_path} is likely authentic with dynamic threshold. Distance: {results["distance"]}, Cosine Similarity: {results["cosine_sim"]}')
            else:
                write_results(f'File {file_path} is likely a forgery with dynamic threshold. No match found.')

            # Individual test results for new tests
            write_results(f'Lighting Consistency Result: {"Authentic" if results["is_authentic_lighting"] else "Forgery"}')
            write_results(f'Frequency Domain Result: {"Authentic" if results["is_authentic_frequency_domain"] else "Forgery"}')
            write_results(f'PRNU Result: {"Authentic" if results["is_authentic_prnu"] else "Forgery"}')
            write_results(f'Metadata Result: {"Authentic" if results["is_authentic_metadata"] else "Forgery"}')
            write_results(f'JPEG Artifacts Result: {"Authentic" if results["is_authentic_jpeg_artifacts"] else "Forgery"}')
            write_results(f'Noise Result: {"Authentic" if results["is_authentic_noise"] else "Forgery"}')
            write_results(f'CFA Result: {"Authentic" if results["is_authentic_cfa"] else "Forgery"}')
        elif is_video_file(file_path):
            write_log(f"Loaded video: {file_path}")
            optical_flow_score = process_with_timeout(process_video_optical_flow, args=(file_path,), timeout_duration=5400)
            texture_score = process_with_timeout(process_video_texture, args=(file_path,), timeout_duration=5400)
            liveness_result = process_with_timeout(detect_liveness, args=(file_path,thresholds.get('cfa_threshold', float('inf'))), timeout_duration=5400)
            
            # Extract audio features
            audio_features = process_with_timeout(extract_audio_features, args=(file_path,), timeout_duration=5400)
            
            # Check audio-video consistency
            av_consistency = check_av_consistency(audio_features, optical_flow_score, texture_score)
            
            # Ensure av_consistency is not None before proceeding
            if av_consistency is not None and av_consistency.all():
                # Proceed with further processing if av_consistency is valid
                logging.info(f"AV consistency check passed for {file_path}. Consistency: {av_consistency}.")
            else:
                logging.warning(f"AV consistency check failed for {file_path}")
            
            # Check temporal inconsistencies
            temporal_threashold = thresholds.get('temporal_consistency_threshold', float('inf'))
            temporal_inconsistencies = check_temporal_inconsistencies(file_path, temporal_consistency_threshold)
            
            # DeepFakeDetection check
            deepfake_threshold = thresholds.get('deepfake_detection', float('inf'))
            deepfake_model_path = 'G:\\My Drive\\Classes\\Dissertation\\Models\\deepfake_model.h5'
            deepfake_check = deepfake_detection_check(file_path, deepfake_model_path , deepfake_threshold)

            if optical_flow_score is not None:
                write_log(f"Optical Flow Score for video {file_path}: {optical_flow_score}")
                if optical_flow_score < thresholds.get('optical_flow', float('inf')):
                    predicted_label = 'authentic'
                    write_results(f'Video {file_path} is likely authentic based on optical flow analysis. Score: {optical_flow_score}')
                else:
                    predicted_label = 'forgery'
                    write_results(f'Video {file_path} is likely a forgery based on optical flow analysis. Score: {optical_flow_score}')
            else:
                write_log(f"Failed to process optical flow for video {file_path}")
                write_results(f'Video {file_path} processing failed during optical flow analysis.')

            if texture_score is not None:
                write_log(f"Texture Score for video {file_path}: {texture_score}")
                if texture_score > thresholds.get('texture_treshold', float('-inf')):
                    predicted_label = 'authentic'
                    write_results(f'Video {file_path} is likely authentic based on texture analysis. Score: {texture_score}')
                else:
                    predicted_label = 'forgery'
                    write_results(f'Video {file_path} is likely a forgery based on texture analysis. Score: {texture_score}')
            else:
                write_log(f"Failed to process texture for video {file_path}")
                write_results(f'Video {file_path} processing failed during texture analysis.')

            if liveness_result is not None:
                write_log(f"Liveness Result for video {file_path}: {liveness_result}")
                if isinstance(liveness_result, bool):
                    if liveness_result:
                        predicted_label = 'authentic'
                        write_results(f'Video {file_path} is likely authentic based on liveness detection. Details: {liveness_result}')
                    else:
                        predicted_label = 'forgery'
                        write_results(f'Video {file_path} is likely a forgery based on liveness detection. Details: {liveness_result}')
                else:
                    predicted_label = 'forgery'
                    write_results(f'Video {file_path} is likely a forgery based on liveness detection. Details: {liveness_result}')
            else:
                write_log(f"Failed to process liveness for video {file_path}")
                write_results(f'Video {file_path} processing failed during liveness detection.')
            # Check AV consistency
            if av_consistency.all():
                write_log(f"AV consistency check passed for video {file_path}.")
                write_results(f'Video {file_path} passed AV consistency check. Score: {av_consistency}')
            else:
                predicted_label = 'forgery'
                write_log(f"AV consistency check failed for video {file_path}.")
                write_results(f'Video {file_path} failed AV consistency check. Score: {av_consistency}')

            # Check temporal inconsistencies
            if temporal_inconsistencies:
                write_log(f"Temporal inconsistencies check passed for video {file_path}.")
                write_results(f'Video {file_path} passed temporal inconsistencies check. Score: {temporal_inconsistencies}')
            else:
                predicted_label = 'forgery'
                write_log(f"Temporal inconsistencies check failed for video {file_path}.")
                write_results(f'Video {file_path} failed temporal inconsistencies check. Score: {temporal_inconsistencies}')

            # DeepFakeDetection check
            if deepfake_check:
                write_log(f"DeepFakeDetection check passed for video {file_path}.")
                write_results(f'Video {file_path} passed DeepFakeDetection check.')
            else:
                predicted_label = 'forgery'
                write_log(f"DeepFakeDetection check failed for video {file_path}.")
                write_results(f'Video {file_path} failed DeepFakeDetection check.')

    except Exception as e:
        write_results(f'Error processing file {file_path}: {str(e)}')
        handle_error(f'Error processing file {file_path}: {str(e)}', e)

    return predicted_label

# Creates and saves a dictionary from image descriptors using MiniBatchDictionaryLearning.
def create_and_save_dictionary(image_descriptors, dictionary_path, pca):
    try:
        write_log(f"Creating dictionary with descriptors of shape {image_descriptors.shape}")
        dictionary_learner = MiniBatchDictionaryLearning(n_components=1024, batch_size=100, verbose=1)
        
        write_log("Starting to fit the dictionary learner in batches. This step can take significant time depending on the size of the dataset.")
        start_time = time.time()
        
        for i in range(0, image_descriptors.shape[0], dictionary_learner.batch_size):
            batch = image_descriptors[i:i + dictionary_learner.batch_size]
            dictionary_learner.partial_fit(batch)
            write_log(f"Processed batch {i // dictionary_learner.batch_size + 1}/{image_descriptors.shape[0] // dictionary_learner.batch_size + 1}")
        
        end_time = time.time()
        write_log(f"Dictionary learner fit complete. Time taken: {end_time - start_time} seconds")
        
        dictionary = dictionary_learner.components_
        write_log("Saving the dictionary to disk")
        np.save(dictionary_path, dictionary)
        write_log(f"Dictionary created and saved to {dictionary_path}")
        
        return dictionary
    except Exception as e:
        handle_error(f"Failed to create and save dictionary: {e}", e)
        return None

# Loads a dictionary from a specified path.
def load_dictionary(dictionary_path):
    try:
        dictionary = np.load(dictionary_path, allow_pickle=True)
        write_log(f"Dictionary loaded from {dictionary_path}")
        return dictionary
    except Exception as e:
        handle_error(f"Failed to load dictionary from {dictionary_path}: {e}", e)
        return None

# Checks if a file exists and logs an error if it doesn't.       
def check_file_exists(file_path, file_type):
    if not os.path.exists(file_path):
        handle_error(f"{file_type} could not be created. Exiting.", level='error')
        exit(1)
    else:
        write_log(f"{file_type} exists at {file_path}.")

# Builds a database from authentic images and videos, extracting features and analyzing various aspects.
def build_database(authentic_images_directory, authentic_videos_directory, model1, model2, model3, dictionary_path, database_path):
    write_log("Beginning DB build.")
    database = {
        'sparse_representations': [],
        'video_fingerprints': [],
        'optical_flow_magnitudes': [],
        'lighting_consistency_scores': [],
        'frequency_domain_scores': [],
        'prnu_patterns': [],
        'metadata_scores': [],
        'jpeg_artifacts_scores': [],
        'noise_scores': [],
        'dynamic_threshold_scores': []
    }

    write_log("Processing images.")
    sample_data = []
    image_paths = [os.path.join(authentic_images_directory, file_name) for file_name in os.listdir(authentic_images_directory) if is_image_file(os.path.join(authentic_images_directory, file_name))]

    with tqdm(total=len(image_paths), desc="Processing images") as pbar:
        for file_path in image_paths:
            write_log(f"Processing image file: {file_path}")
            try:
                image = load_image(file_path)
                if image is not None:
                    descriptors = extract_features(image, model1, model2, model3, fixed_length=None, pca=None) 
                    if descriptors is not None:
                        sample_data.append(descriptors)
                pbar.update(1)
            except Exception as e:
                handle_error(f"Error processing image {file_path}: {str(e)}", e)
                continue

    sample_data = np.array(sample_data)
    if sample_data.ndim == 1:
        sample_data = sample_data.reshape(1, -1)
    elif sample_data.ndim == 2:
        sample_data = sample_data
    else:
        handle_error(f"Unexpected dimensions in sample_data: {sample_data.shape}", 'error')
        return None

    # Validate and clean sample_data
    if np.isnan(sample_data).any():
        handle_error(f"Sample data contains NaN values. Cleaning the data.", 'error')
        sample_data = sample_data[~np.isnan(sample_data).any(axis=1)]

    write_log(f"Shape of sample_data: {sample_data.shape}", 'debug')

    # Calculate the number of PCA components
    n_components = calculate_cumulative_explained_variance(sample_data, variance_threshold=0.95)

    try:
        if not os.path.exists(dictionary_path):
            write_log("Creating and saving dictionary with PCA-transformed sample data.")
            transformed_data = pca.transform(sample_data)
            dictionary = create_and_save_dictionary(transformed_data, dictionary_path, pca)
            check_file_exists(dictionary_path, "Dictionary")
        else:
            dictionary = load_dictionary(dictionary_path)
            write_log(f"Dictionary file already exists at {dictionary_path}. Skipping dictionary creation.")
    except Exception as e:
        handle_error(f"Error creating or loading dictionary: {str(e)}", e)
        return None

    try:
        write_log("Training PCA model with sample data.")
        pca = train_pca_model(sample_data, n_components=n_components)
        save_pca_model(pca, pca_model_path)
        check_file_exists(pca_model_path, "PCA Model")
    except Exception as e:
        handle_error(f"Error training PCA model: {str(e)}", e)
        return None

    with tqdm(total=len(image_paths), desc="Extracting features and building database") as pbar:
        for file_path in image_paths:
            write_log(f"Processing image file for database: {file_path}")
            try:
                image = load_image(file_path)
                if image is not None:
                    descriptors = extract_features(image, model1, model2, model3, fixed_length=n_components, pca=pca) 
                    if descriptors.shape[0] == pca.n_components:
                        sparse_rep = sparse_representation(descriptors, dictionary, pca.n_components)
                        database['sparse_representations'].append(sparse_rep)
                        prnu = extract_prnu(image)
                        database['prnu_patterns'].append(prnu)
                        lighting_score = shadow_lighting_consistency(image)
                        database['lighting_consistency_scores'].append(lighting_score)
                        frequency_score = frequency_domain_analysis(image)
                        database['frequency_domain_scores'].append(frequency_score)
                        metadata_score = metadata_analysis(file_path)
                        database['metadata_scores'].append(metadata_score)
                        jpeg_artifacts_score = jpeg_artifacts_analysis(image)
                        write_log(f"JPEG artifacts score for {file_path}: {jpeg_artifacts_score}")
                        database['jpeg_artifacts_scores'].append(jpeg_artifacts_score)
                        noise_score = noise_analysis(image)
                        write_log(f"Noise score for {file_path}: {noise_score}")
                        database['noise_scores'].append(noise_score)
                        dynamic_threshold_score = calculate_dynamic_threshold_score(image)
                        database['dynamic_threshold_scores'].append(dynamic_threshold_score)
                    else:
                        handle_error(f"Inconsistent descriptor shape for {file_path}: {descriptors.shape}", 'error')
                pbar.update(1)
            except Exception as e:
                handle_error(f"Error processing image {file_path} for database: {str(e)}", e)
                continue

    write_log("Processing videos.")
    video_paths = [os.path.join(authentic_videos_directory, file_name) for file_name in os.listdir(authentic_videos_directory) if is_video_file(os.path.join(authentic_videos_directory, file_name))]

    with tqdm(total=len(video_paths), desc="Processing videos") as pbar:
        for file_path in video_paths:
            write_log(f"Processing video file: {file_path}")
            try:
                fingerprints = generate_fingerprints(file_path, model1, model2, model3, pca, dictionary, n_components) 
                database['video_fingerprints'].append(fingerprints)
                optical_flow_magnitude = process_video_optical_flow(file_path)
                database['optical_flow_magnitudes'].append(optical_flow_magnitude)
                pbar.update(1)
            except Exception as e:
                handle_error(f"Error processing video {file_path}: {str(e)}", e)
                continue

    write_log("Saving database.")
    try:
        np.save(database_path, database)
        check_file_exists(database_path, "Database")
        write_log(f"Database saved to {database_path}")
    except Exception as e:
        handle_error(f"Error saving database: {str(e)}", e)
        return None

    return database

# Saves the database
def save_database(database, database_path):
    try:
        np.save(database_path, database)
        write_log(f"Database saved to {database_path}")
    except Exception as e:
        handle_error(f"Error saving database: {str(e)}", e)

# Loads or creates a database, building it from scratch if necessary.
def load_or_create_database(database_path, authentic_images_directory, authentic_videos_directory, dictionary, model1, model2, model3, fixed_length):
    if os.path.exists(database_path):
        database = load_database(database_path)
        if database is not None:
            write_log("Database loaded successfully.")
            return database
        else:
            write_log("Database is invalid or not found. Rebuilding database...", 'info')
    else:
        write_log("Database path not found. Building database from scratch...", 'info')
    
    database = build_database(authentic_images_directory, authentic_videos_directory, model1, model2, model3, dictionary_path, database_path)

    save_database(database, database_path)
    check_file_exists(database_path, "Database")
    return database

# Loads or creates a PCA model, building it from scratch if necessary.
def load_or_create_pca_model(pca_model_path, sample_data, n_components):
    if os.path.exists(pca_model_path) and validate_pca_model(pca_model_path):
        return load_pca_model(pca_model_path)
    else:
        write_log("PCA model not found or invalid. Building PCA model from scratch...", 'info')
        pca = train_pca_model(sample_data, n_components)
        save_pca_model(pca, pca_model_path)
        check_file_exists(pca_model_path, "PCA Model")
        return pca

def check_temporal_inconsistencies(video_path, temporal_threshold):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        prev_frame = None
        temporal_inconsistencies = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                # Calculate the absolute difference between the current frame and the previous frame
                frame_diff = cv2.absdiff(prev_frame, gray_frame)
                # Calculate the mean of the differences
                mean_diff = np.mean(frame_diff)
                temporal_inconsistencies.append(mean_diff)

            prev_frame = gray_frame

        cap.release()

        # Calculate the overall temporal inconsistency score
        inconsistency_score = np.mean(temporal_inconsistencies)
        is_consistent = inconsistency_score < temporal_threshold

        write_results(f"Temporal inconsistency score for video {video_path}: {inconsistency_score}")
        return is_consistent

    except Exception as e:
        handle_error(f"Error checking temporal inconsistencies for {video_path}: {e}", e)
        return False

def cfa_analysis(image):
    try:
        logging.info("Performing CFA analysis")
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform a Sobel filter to detect edges
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of the gradient
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize the magnitude
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold the magnitude to create a binary image
        _, binary_image = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate the consistency of the CFA pattern
        cfa_consistency = np.mean(binary_image)
        
        write_results(f"CFA consistency score: {cfa_consistency}")
        return cfa_consistency
    except Exception as e:
        handle_error(f"Error performing CFA analysis: {e}", e)
        return float('inf')

def extract_audio_features(file_path):
    audio_path = file_path + '.wav'  # Temporarily store audio as .wav
    try:
        audio_clip = AudioFileClip(file_path)
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le')  # Specify the codec here
        audio_features = process_audio(audio_path)
        audio_clip.close()

        # Ensure the audio file is closed before removing it
        for _ in range(5):
            try:
                os.remove(audio_path)
                break
            except PermissionError:
                time.sleep(1)  # Wait for 1 second before retrying
        return audio_features
    except Exception as e:
        logging.error(f"Error extracting audio features from {file_path}: {str(e)}")
        return None

def process_audio(audio_path):
    try:
        import librosa

        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs, axis=1)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = np.mean(chroma, axis=1)
        
        # Extract spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast = np.mean(spectral_contrast, axis=1)
        
        # Concatenate the features into a single feature vector
        audio_features = np.concatenate((mfccs, chroma, spectral_contrast))
        
        return audio_features
    except Exception as e:
        logging.error(f"Error processing audio features from {audio_path}: {e}")
        return None

def check_av_consistency(audio_features, optical_flow_score, texture_score):
    try:
        if audio_features is None:
            raise ValueError("No valid audio features found.")

        # Ensure audio_features is a valid array
        if not isinstance(audio_features, np.ndarray):
            raise ValueError("Invalid audio features array.")

        # Log the shape of the audio features array
        logging.debug(f"Shape of audio features: {audio_features.shape}")

        # Assuming audio_features is a 1D array, calculate some consistency metric
        avg_mfcc = np.mean(audio_features)

        return avg_mfcc
    except ValueError as e:
        logging.error(f"Error checking AV consistency: {e}")
        return np.zeros(1)  # Return a default value to avoid NoneType

# Use the custom_objects dictionary when loading the model
def load_model_with_fixed_reduction(model_path, custom_objects=None):
    with h5py.File(model_path, 'r') as f:
        # Load model configuration
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model configuration found in the file.')
        model_config = json.loads(model_config)
        model = model_from_json(json.dumps(model_config), custom_objects=custom_objects)

        # Load model weights
        model.load_weights(model_path)

        # Load training configuration
        training_config = f.attrs.get('training_config')
        if training_config is None:
            raise ValueError('No training configuration found in the file.')
        training_config = json.loads(training_config)

        # Load optimizer configuration
        optimizer_config = training_config.get('optimizer_config')
        if optimizer_config is None:
            raise ValueError('No optimizer configuration found in the file.')

        # Get optimizer class and configuration
        optimizer_class = optimizer_config.get('class_name')
        optimizer_params = optimizer_config.get('config')

        # Filter out unsupported arguments
        unsupported_keys = ['jit_compile', 'is_legacy_optimizer']
        for key in unsupported_keys:
            if key in optimizer_params:
                del optimizer_params[key]

        # Create optimizer instance
        if optimizer_class == 'Custom>Adam':
            optimizer = CustomAdam.from_config(optimizer_params)
        elif optimizer_class == 'Adam':
            optimizer = Adam.from_config(optimizer_params)
        else:
            raise ValueError(f'Unsupported optimizer class: {optimizer_class}')

        # Compile model with the loaded optimizer
        model.compile(optimizer=optimizer, loss=training_config['loss'], metrics=training_config['metrics'])

        return model

def get_optimizer(optimizer_config):
    config = optimizer_config['config']
    # Filter out unrecognized arguments
    recognized_keys = set(Adam.__init__.__code__.co_varnames)
    filtered_config = {k: v for k, v in config.items() if k in recognized_keys}
    return Adam.from_config(filtered_config)

def deepfake_detection_check(file_path, deepfake_model_path, threshold):
    try:
        logging.info(f"Performing DeepFakeDetection check for {file_path}")
        #deepfake_model_path = 'G:\\My Drive\\Classes\\Dissertation\\Models\\deepfake_model.h5'

        if not os.path.exists(deepfake_model_path):
            handle_error(f"Deepfake model file not found: {deepfake_model_path}", None)
            return False

        # Load the pre-trained deepfake detection model
        deepfake_model = load_model_with_fixed_reduction(deepfake_model_path, custom_objects={
            'Custom>CastToFloat32': CustomCastToFloat32,
            'Custom>Adam': CustomAdam,
            'Functional': CustomFunctional
        })
        write_log("Deepfake model loaded successfully.")

        # Load the video
        cap = cv2.VideoCapture(file_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Preprocess frames and perform deepfake detection
        is_deepfake = False
        for frame in frames:
            # Preprocess the frame
            frame_resized = cv2.resize(frame, (64, 64))  # Resize to 64x64 as expected by the model
            frame_normalized = frame_resized / 255.0  # Normalize the frame
            frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

            # Predict with the model
            prediction = deepfake_model.predict(frame_expanded)
            # Assuming the model returns a single value
            if prediction[0] > threshold:
                is_deepfake = True
                break

        # Decision logic based on model predictions
        if is_deepfake:
            write_log(f"DeepFakeDetection check for {file_path}: Detected as deepfake")
            write_results(f"DeepFakeDetection check for {file_path}: Detected as deepfake")
            return True
        else:
            write_log(f"DeepFakeDetection check for {file_path}: Not detected as deepfake")
            write_results(f"DeepFakeDetection check for {file_path}: Not detected as deepfake")
            return False
        
    except Exception as e:
        handle_error(f"Error performing DeepFakeDetection check for {file_path}: {e}", e)
        return False

def train_model():
    # Define a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Assuming you have a directory structure for your dataset
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'path_to_train_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'path_to_validation_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32
    )

    # Save the model
    model.save('deepfake_model.h5')

def prompt_for_threshold(key, calc_value, default_value=None):
    if calc_value == 0:
        if default_value is not None:
            prompt = f"Calculated {key} threshold is zero. The default value is {default_value}. Enter the value you would like to use: "
        else:
            prompt = f"Calculated {key} threshold is zero. There is no default value found in config.ini. Please enter a threshold value: "
    elif default_value is not None and abs(calc_value - default_value) / default_value > 0.01:  # 1% variance check
        prompt = f"Calculated {key} threshold differs significantly from the default value. Default: {default_value}, Calculated: {calc_value}. Enter the value you would like to use: "
    else:
        return calc_value

    user_input = input(prompt)
    try:
        return float(user_input)
    except ValueError:
        print("Invalid input. Using calculated value.")
        return calc_value

def calculate_thresholds(database, config):
    def safe_mean(values):
        filtered_values = [v for v in values if isinstance(v, (int, float))]
        if not filtered_values:
            return 0
        try:
            return np.mean(filtered_values)
        except TypeError as e:
            handle_error(f"Error calculating mean for values {filtered_values}: {e}", e)
            return 0

    def safe_std(values):
        filtered_values = [v for v in values if isinstance(v, (int, float))]
        if not filtered_values:
            return 0
        try:
            return np.std(filtered_values)
        except TypeError as e:
            handle_error(f"Error calculating std for values {filtered_values}: {e}", e)
            return 0

    def get_safe_values(key):
        values = database.get(key, [])
        try:
            float_values = [float(v) for v in values if v is not None and isinstance(v, (int, float, str))]
            return float_values
        except ValueError as e:
            handle_error(f"Error converting values to float for key {key}: {e}", e)
            return []

    thresholds = {
        'lighting_consistency': safe_mean(get_safe_values('lighting_consistency_scores')) + safe_std(get_safe_values('lighting_consistency_scores')),
        'frequency_domain': safe_mean(get_safe_values('frequency_domain_scores')) + safe_std(get_safe_values('frequency_domain_scores')),
        'metadata': safe_mean(get_safe_values('metadata_scores')) + safe_std(get_safe_values('metadata_scores')),
        'jpeg_artifacts': safe_mean(get_safe_values('jpeg_artifacts_scores')) + safe_std(get_safe_values('jpeg_artifacts_scores')),
        'noise': safe_mean(get_safe_values('noise_scores')) + safe_std(get_safe_values('noise_scores')),
        'cfa': safe_mean(get_safe_values('cfa_scores_upper')) + safe_std(get_safe_values('cfa_scores')),
        'prnu': safe_mean(get_safe_values('prnu_scores')) + safe_std(get_safe_values('prnu_scores')),
        'dynamic_threshold_factor': safe_mean(get_safe_values('dynamic_threshold_factor_scores')) + safe_std(get_safe_values('dynamic_threshold_factor_scores')),
        'texture': safe_mean(get_safe_values('texture_scores')) + safe_std(get_safe_values('texture_scores')),
        'optical_flow': safe_mean(get_safe_values('optical_flow_scores')) + safe_std(get_safe_values('optical_flow_scores')),
        'temporal_consistency': safe_mean(get_safe_values('temporal_consistency_scores')) + safe_std(get_safe_values('temporal_consistency_scores')),
        'static_value': safe_mean(get_safe_values('static_value_scores')) + safe_std(get_safe_values('static_value_scores')),
        'ear_value': safe_mean(get_safe_values('ear_value_scores')) + safe_std(get_safe_values('ear_value_scores')),
        'deepfake_detection': safe_mean(get_safe_values('deepfake_value_scores')) + safe_std(get_safe_values('deepfake_value_scores'))
    }

    for key, calc_value in thresholds.items():
        if calc_value is None or np.isnan(calc_value):
            calc_value = 0  # Assign a default value to None entries
        write_log(f"Calculated {key} threshold: {calc_value} (Type: {type(calc_value)})")
        default_value = config.getfloat('Thresholds', f'{key}_threshold', fallback=0.1 if key == 'dynamic_threshold_factor' else None)
        thresholds[key] = prompt_for_threshold(key, calc_value, default_value)

    for key, value in thresholds.items():
        write_log(f"Threshold for {key}: {value} (Type: {type(value)})")
        
    return thresholds

# Function to calculate accuracy metrics
def calculate_accuracy(tp, fp, tn, fn):
    write_log(f"Calculating accuracy with: TP={tp}, FP={fp}, TN={tn}, FN={fn}", 'debug')
    total = tp + fp + tn + fn
    if total == 0:
        return 0, 0, 0, 0
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    write_log("Starting MMIFD")
    write_results("MMIFD - Starting Detection Script")

    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    database_path = config['Paths']['database_path']
    pca_model_path = config['Paths']['pca_model_path']
    light_direction_model_path = config['Paths']['light_direction_model_path']
    dictionary_path = config['Paths']['dictionary_path']
    authentic_images_directory = config['Paths']['authentic_images_directory']
    authentic_videos_directory = config['Paths']['authentic_videos_directory']
    corpus_directory = config['Paths']['corpus_directory']
    ground_truth_labels_path = config['Paths']['ground_truth_labels_path']
    ground_truth_labels = load_ground_truth_labels(ground_truth_labels_path)
    fixed_length = config.getint('ModelParameters', 'fixed_length')
    lighting_consistency_threshold = config.getfloat('Thresholds', 'lighting_consistency_threshold')
    frequency_domain_threshold = config.getfloat('Thresholds', 'frequency_domain_threshold')
    prnu_threshold = config.getfloat('Thresholds', 'prnu_threshold')
    metadata_threshold = config.getfloat('Thresholds', 'jpeg_artifacts_threshold')
    jpeg_artifacts_threshold = config.getfloat('Thresholds', 'jpeg_artifacts_threshold')
    noise_threshold = config.getfloat('Thresholds', 'noise_threshold')
    cfa_threshold = config.getfloat('Thresholds', 'cfa_threshold')
    texture_threshold = config.getfloat('Thresholds', 'texture_threshold')
    optical_flow_threshold = config.getfloat('Thresholds', 'optical_flow_threshold')
    temporal_consistency_threshold = config.getfloat('Thresholds', 'temporal_consistency_threshold')
    dynamic_threshold_factor = config.getfloat('Thresholds', 'dynamic_threshold_factor')
    static_threshold_value = config.getfloat('Thresholds', 'static_threshold_value')
    deepfake_detection_threshold = config.getfloat('Thresholds', 'deepfake_detection_threshold')
    eye_ar_threshold = config.getfloat('Thresholds', 'eye_ar_threshold')
        
    # Initialize models for feature extraction
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    inception_model = InceptionV3(weights='imagenet', include_top=False)

    # Log model initialization
    write_log("VGG16 model initialized successfully.")
    write_log("ResNet50 model initialized successfully.")
    write_log("InceptionV3 model initialized successfully.")

    # Load or create the light direction model
    try:
        if not os.path.exists(light_direction_model_path):
            write_log("Light direction model not found. Building the model...", 'info')
            light_directions_db = build_light_direction_database(authentic_images_directory)
            light_direction_model = train_light_direction_model_with_db(light_directions_db)
            if light_direction_model is not None:
                joblib.dump(light_direction_model, light_direction_model_path)
                check_file_exists(light_direction_model_path, "Light Direction Model")
            else:
                handle_error("Light direction model training failed.", 'error')
        else:
            light_direction_model = load_light_direction_model(light_direction_model_path)
    except Exception as e:
        handle_error(f"Error training or loading light direction model: {str(e)}", e)
        exit(1)

    # Load or create PCA model
    pca = None
    n_components = None
    try:
        write_log("Loading PCA model...", 'info')
        if os.path.exists(pca_model_path) and validate_pca_model(pca_model_path):
            pca = load_pca_model(pca_model_path)
            n_components = pca.n_components_  # Ensure n_components is set from the loaded PCA model
        else:
            write_log("PCA model not found or invalid. Building PCA model from scratch...", 'info')
            sample_data = extract_features_from_directory(authentic_images_directory, vgg16_model, resnet_model, inception_model, fixed_length=None)
            n_components = calculate_cumulative_explained_variance(sample_data, variance_threshold=0.95)
            pca = train_pca_model(sample_data, n_components)
            save_pca_model(pca, pca_model_path)
            check_file_exists(pca_model_path, "PCA Model")
    except Exception as e:
        handle_error(f"Error loading or creating PCA model: {str(e)}", e)
        exit(1)

    # Load or create dictionary
    try:
        if not os.path.exists(dictionary_path):
            write_log("Dictionary not found. Creating a new dictionary...")
            image_descriptors = extract_features_from_directory(authentic_images_directory, vgg16_model, resnet_model, inception_model, fixed_length=None)
            transformed_data = pca.transform(image_descriptors)
            if transformed_data is not None:
                dictionary = create_and_save_dictionary(transformed_data, dictionary_path, pca)
                check_file_exists(dictionary_path, "Dictionary")
        else:
            dictionary = load_dictionary(dictionary_path)
            write_log(f"Dictionary file already exists at {dictionary_path}. Skipping dictionary creation.")
            if not validate_dictionary_with_pca(dictionary, pca):
                write_log("Invalid dictionary. Rebuilding dictionary...", 'info')
                image_descriptors = extract_features_from_directory(authentic_images_directory, vgg16_model, resnet_model, inception_model, fixed_length=None)
                transformed_data = pca.transform(image_descriptors)
                dictionary = create_and_save_dictionary(transformed_data, dictionary_path, pca)
                check_file_exists(dictionary_path, "Dictionary")
    except Exception as e:
        handle_error(f"Error creating or loading dictionary: {str(e)}", e)
        exit(1)

    # Load or create database
    try:
        write_log("Loading database...", 'info')
        database = load_or_create_database(database_path, authentic_images_directory, authentic_videos_directory, dictionary, vgg16_model, resnet_model, inception_model, fixed_length=n_components)
    except Exception as e:
        handle_error(f"Error loading or creating database: {str(e)}", e)
        exit(1)

    # Calculate thresholds at runtime
    thresholds = calculate_thresholds(database, config)
    write_log(f"Threshold values before processing: {thresholds}")

    # Initialize counters
    total_files_processed = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Process each test file
    for test_file in os.listdir(corpus_directory):
        test_file_path = os.path.join(corpus_directory, test_file)
        total_files_processed += 1
        predicted_label = process_file(test_file_path, dictionary, database, vgg16_model, resnet_model, inception_model, pca, fixed_length, thresholds)

        if predicted_label:
            write_log(f"File: {test_file}, Predicted Label: {predicted_label}", 'debug')
            write_results(f"File: {test_file}, Predicted Label: {predicted_label}")
            actual_label = ground_truth_labels.get(test_file.upper())
            if actual_label:
                write_results(f"File: {test_file}, Predicted Label: {predicted_label}, Actual Label: {actual_label}")

                if predicted_label == 'authentic' and actual_label == 'authentic':
                    true_positives += 1
                elif predicted_label == 'forgery' and actual_label == 'forgery':
                    true_negatives += 1
                elif predicted_label == 'authentic' and actual_label == 'forgery':
                    false_positives += 1
                elif predicted_label == 'forgery' and actual_label == 'authentic':
                    false_negatives += 1

                # Calculate accuracy metrics
                accuracy, precision, recall, f1_score = calculate_accuracy(true_positives, false_positives, true_negatives, false_negatives)

                # Log the final accuracy metrics
                write_results(f'True Positives: {true_positives}')
                write_results(f'False Positives: {false_positives}')
                write_results(f'True Negatives: {true_negatives}')
                write_results(f'False Negatives: {false_negatives}')
                write_results(f'Accuracy: {accuracy:.2f}')
                write_results(f'Precision: {precision:.2f}')
                write_results(f'Recall: {recall:.2f}')
                write_results(f'F1 Score: {f1_score:.2f}')
        else:
            write_log(f"File: {test_file}, no labels", 'debug')

write_results(f'Total Files Processed: {total_files_processed}')
write_log("Processing complete.")