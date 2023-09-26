import os
import json
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn import image, datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time

# Fetch the standard ICBM 152 brain template
template = datasets.fetch_icbm152_2009()
t1_template = template.t1

# List of valid MRI types
VALID_FILES = ["bold", "T1w", "T2w", "dwi", "minIP", "swi", "T2star", "angio", "asl", "fieldmap", "FLAIR", "GRE"]
# Expected feature length is three times the number of valid MRI types (mean intensity, std intensity, magnetic strength)
EXPECTED_FEATURE_LENGTH = 3 * len(VALID_FILES)

# Initialize a global counter for the number of ignored images
ignored_images_count = 0

def compute_mean_std_intensity(voxel_data):
    brain_voxels = voxel_data[voxel_data > 0]
    return np.mean(brain_voxels), np.std(brain_voxels)

def preprocess_mri(input_filename):
    backup_filename = input_filename.replace('.nii.gz', "_original.nii.gz")
    if not os.path.exists(backup_filename):
        os.rename(input_filename, backup_filename)
    else:
        return nib.load(input_filename)

    img = nib.load(backup_filename)
    img_data = img.get_fdata()

    if img_data.ndim == 4:
        img_data = np.mean(img_data, axis=-1)

    mask = compute_epi_mask(img)
    brain_extracted_data = img_data * mask.get_fdata()

    normalized_data = (brain_extracted_data - np.mean(brain_extracted_data)) / np.std(brain_extracted_data)
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)
    resampled_img = image.resample_to_img(normalized_img, nib.load(t1_template))
    nib.save(resampled_img, input_filename)

    return resampled_img

def extract_features_from_preprocessed_mri_files(**mri_files):
    global ignored_images_count
    features = []
    for mri_type, mri_file in mri_files.items():
        if mri_file:
            json_filename = mri_file.replace('.nii.gz', '.json')
            if os.path.exists(json_filename):
                metadata = load_mri_metadata(json_filename)
                quality = check_mri_quality(metadata)
                if not quality:
                    ignored_images_count += 1
                    continue
                magnetic_strength = float(metadata.get("MagneticFieldStrength", "0"))
                features.append(magnetic_strength)
            else:
                features.extend([0])
            
            preprocessed_img = preprocess_mri(mri_file)
            mean_intensity, std_intensity = compute_mean_std_intensity(preprocessed_img.get_fdata())
            features.extend([mean_intensity, std_intensity])
        else:
            features.extend([0, 0, 0])
    return features

def load_mri_metadata(json_filename):
    with open(json_filename, 'r') as file:
        metadata = json.load(file)
    return metadata

def check_mri_quality(metadata):
    required_magnet_strength = "3T"
    return metadata.get("MagneticFieldStrength", "") == required_magnet_strength

def load_data(directory):
    features, labels = [], []
    for age in range(42, 98):
        age_path = os.path.join(directory, f"Age_{age}")
        if os.path.isdir(age_path):
            subjects = set([f.split('_')[0] for f in os.listdir(age_path)])
            for subject in subjects:
                mri_files = {mri_type: os.path.join(age_path, filename) 
                             for mri_type, filename in [(f.split('_')[1].split('.')[0], f) 
                             for f in os.listdir(age_path) if f.startswith(subject) and not f.endswith('.json')]}
                feature_vector = extract_features_from_preprocessed_mri_files(**mri_files)
                while len(feature_vector) < EXPECTED_FEATURE_LENGTH:
                    feature_vector.append(0)
                if len(feature_vector) == EXPECTED_FEATURE_LENGTH:
                    features.append(feature_vector)
                    labels.append(age)

    return np.array(features), np.array(labels)

def train_and_evaluate_model(features, labels):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    return model, mean_absolute_error(y_val, val_predictions), mean_squared_error(y_val, val_predictions)

def main_execution(training_directory, testing_directory):
    global ignored_images_count
    start_time = time.time()

    training_features, training_labels = load_data(training_directory)
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)

    model, train_mae, train_mse = train_and_evaluate_model(training_features, training_labels)
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training MSE: {train_mse:.2f}")

    testing_features, testing_labels = load_data(testing_directory)
    testing_features = scaler.transform(testing_features)

    test_predictions = model.predict(testing_features)
    test_mae = mean_absolute_error(testing_labels, test_predictions)
    test_mse = mean_squared_error(testing_labels, test_predictions)
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")

    print(f"Total number of ignored images due to low quality: {ignored_images_count}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# The main_execution function is ready to be called when you want to run the pipeline.


if __name__ == "__main__":
    main_execution("Data\Training", "Data\Testing")
