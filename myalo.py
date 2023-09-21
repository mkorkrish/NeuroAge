import os
import shutil
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn import image
from nilearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nilearn.plotting as niplot
import matplotlib.pyplot as plt
import time
import ctypes

# Constants
template = datasets.fetch_icbm152_2009()
t1_template = template.t1
PROCESSED_SUFFIX = "_processed.nii.gz"
DISK_SPACE_THRESHOLD = 10 * 1024 * 1024 * 1024

# Disk Management Functions

def ensure_free_space(required_space):
    # Ensure free disk space by setting OneDrive folders to online-only if needed.
    total, used, free = shutil.disk_usage('/')
    if free < required_space:
        onedrive_folders = ["Testing", "Training"]
        for onedrive_folder in onedrive_folders:
            print(f"Ensuring space by setting {onedrive_folder} folder to online-only...")
            for folder in os.listdir(onedrive_folder):
                folder_path = os.path.join(onedrive_folder, folder)
                set_online_only(folder_path)

def set_online_only(path):
    # Set a folder to online-only to free up disk space.
    FILE_ATTRIBUTE_UNPINNED = 0x00001000
    try:
        ctypes.windll.kernel32.SetFileAttributesW(path, FILE_ATTRIBUTE_UNPINNED)
    except Exception as e:
        print(f"Error while setting {path} to online-only: {e}")

# MRI Processing Functions

def compute_mean_std_intensity(voxel_data):
    # Compute the mean and standard deviation of the intensity of brain voxels.
    brain_voxels = voxel_data[voxel_data > 0]
    mean_intensity = np.mean(brain_voxels)
    std_intensity = np.std(brain_voxels)
    return mean_intensity, std_intensity

def load_mri_data(mri_file):
    # Load MRI data from a file.
    print(f"Loading MRI data from: {mri_file}")
    return nib.load(mri_file).get_fdata()

def preprocess_mri(input_filename):
    # Preprocess MRI data by extracting brain voxels, normalizing, and resampling.
    processed_filename = input_filename.replace('.nii.gz', PROCESSED_SUFFIX)
    if os.path.exists(processed_filename):
        return nib.load(processed_filename)

    print(f"Preprocessing MRI: {input_filename}")
    print(f"Loading and processing file: {input_filename}")
    img = nib.load(input_filename)
    img_data = img.get_fdata()
    if img_data.ndim == 4:
        img_data = np.mean(img_data, axis=-1)

    mask = compute_epi_mask(img)
    brain_extracted_data = img_data * mask.get_fdata()
    mean_intensity, std_intensity = compute_mean_std_intensity(brain_extracted_data)
    normalized_data = (brain_extracted_data - mean_intensity) / std_intensity
    normalized_data += abs(np.min(normalized_data))
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)
    template_img = nib.load(t1_template)
    resampled_img = image.resample_to_img(normalized_img, template_img)
    nib.save(resampled_img, processed_filename)

    # Delete the original unprocessed MRI file
    print(f"Deleting unprocessed MRI file: {input_filename}")
    os.remove(input_filename)

    return resampled_img

def extract_features_from_preprocessed_mri_files(t1w_file=None, t2w_file=None, flair_file=None, dwi_file=None, bold_file=None, swi_file=None, asl_file=None):
    # Extract features from preprocessed MRI files.
    features = []
    for mri_file in [t1w_file, t2w_file, flair_file, dwi_file, bold_file, swi_file, asl_file]:
        if mri_file:
            preprocessed_img = preprocess_mri(mri_file)
            mri_data = preprocessed_img.get_fdata()
            mean_intensity, std_intensity = compute_mean_std_intensity(mri_data)
            features.extend([mean_intensity, std_intensity])
        else:
            features.extend([0, 0])
    return features

# Data Loading and Training Functions

def load_data(directory):
    # Load and preprocess MRI data from a directory.
    print(f"Loading and preprocessing MRI data from {directory} directory...")
    features, labels = [], []
    for age in range(42, 98):
        age_path = os.path.join(directory, f"Age_{age}")
        if os.path.isdir(age_path):
            print(f"Processing data for Age {age}...")
            subjects = set([f.split('_')[0] for f in os.listdir(age_path)])
            for subject in subjects:
                mri_files = {"t1w": None, "t2w": None, "flair": None, "dwi": None, "bold": None, "swi": None, "asl": None}
                for mri_type in mri_files.keys():
                    mri_files[mri_type] = next((f for f in os.listdir(age_path) if f.startswith(subject) and f"{mri_type}.nii.gz" in f), None)
                feature_vector = extract_features_from_preprocessed_mri_files(
                    os.path.join(age_path, mri_files["t1w"]) if mri_files["t1w"] else None,
                    os.path.join(age_path, mri_files["t2w"]) if mri_files["t2w"] else None,
                    os.path.join(age_path, mri_files["flair"]) if mri_files["flair"] else None,
                    os.path.join(age_path, mri_files["dwi"]) if mri_files["dwi"] else None,
                    os.path.join(age_path, mri_files["bold"]) if mri_files["bold"] else None,
                    os.path.join(age_path, mri_files["swi"]) if mri_files["swi"] else None,
                    os.path.join(age_path, mri_files["asl"]) if mri_files["asl"] else None
                )
                features.append(feature_vector)
                labels.append(age)
            set_online_only(age_path)
    return np.array(features), np.array(labels)

def train_and_evaluate_model(features, labels):
    # Train and evaluate a Random Forest regressor model.
    print("Training model...")
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Evaluating model on validation set...")
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    mse = mean_squared_error(y_val, val_predictions)
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation MSE: {mse:.2f}")
    return model, X_val, y_val, 

# ------------------ MAIN EXECUTION ------------------

def main_execution(training_directory, testing_directory):
    start_time = time.time()

    print("Ensuring sufficient disk space for training data...")
    ensure_free_space(DISK_SPACE_THRESHOLD)

    training_features, training_labels = load_data(training_directory)
    trained_model, X_val, y_val, val_predictions = train_and_evaluate_model(training_features, training_labels)
    plot_results(y_val, val_predictions, title="Validation Set: True Age vs. Predicted Brain Age")

    print("Ensuring sufficient disk space for testing data...")
    ensure_free_space(DISK_SPACE_THRESHOLD)

    testing_features, testing_labels = load_data(testing_directory)
    print("Predicting brain ages on testing data...")
    predicted_ages = trained_model.predict(testing_features)
    mae = mean_absolute_error(testing_labels, predicted_ages)
    mse = mean_squared_error(testing_labels, predicted_ages)
    print(f"Testing MAE: {mae:.2f}")
    print(f"Testing MSE: {mse:.2f}")
    plot_results(testing_labels, predicted_ages, title="Testing Set: True Age vs. Predicted Brain Age")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

TRAINING_DIR = "Training"
TESTING_DIR = "Testing"
main_execution(TRAINING_DIR, TESTING_DIR)
