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

def compute_mean_std_intensity(voxel_data):
    """
    Compute the mean and standard deviation of the intensity of brain voxels.
    Args:
    - voxel_data: numpy array of MRI voxel intensities
    
    Returns:
    - Tuple of mean and standard deviation of brain voxels
    """
    brain_voxels = voxel_data[voxel_data > 0]
    return np.mean(brain_voxels), np.std(brain_voxels)

def preprocess_mri(input_filename):
    """
    Preprocess an MRI file.
    The preprocessing involves:
    - Loading the MRI
    - Handling 4D data by averaging over the last dimension
    - Extracting brain regions using a mask
    - Normalizing voxel intensities
    - Resampling the MRI to match a standard template
    
    Args:
    - input_filename: String, path to the MRI file
    
    Returns:
    - Resampled and preprocessed MRI image
    """
    backup_filename = input_filename.replace('.nii.gz', "_original.nii.gz")
    # Check if preprocessing was done before by looking for a backup file
    if not os.path.exists(backup_filename):
        os.rename(input_filename, backup_filename)
    else:
        return nib.load(input_filename)

    # Load the MRI data
    img = nib.load(backup_filename)
    img_data = img.get_fdata()

    # Handle 4D MRI data by averaging over the last dimension
    if img_data.ndim == 4:
        img_data = np.mean(img_data, axis=-1)

    # Compute the brain mask and extract brain regions
    mask = compute_epi_mask(img)
    brain_extracted_data = img_data * mask.get_fdata()

    # Normalize voxel intensities
    normalized_data = (brain_extracted_data - np.mean(brain_extracted_data)) / np.std(brain_extracted_data)
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)

    # Resample the MRI to match a standard template
    resampled_img = image.resample_to_img(normalized_img, nib.load(t1_template))
    nib.save(resampled_img, input_filename)

    return resampled_img

def extract_features_from_preprocessed_mri_files(**mri_files):
    """
    Extract features from preprocessed MRI data.
    The features include:
    - Mean and standard deviation of voxel intensities
    - MRI metadata such as magnetic strength
    
    Args:
    - mri_files: Dictionary containing MRI types as keys and file paths as values
    
    Returns:
    - List of extracted features
    """
    features = []
    for mri_type, mri_file in mri_files.items():
        if mri_file:
            # Preprocess the MRI and compute mean and std intensity
            preprocessed_img = preprocess_mri(mri_file)
            mean_intensity, std_intensity = compute_mean_std_intensity(preprocessed_img.get_fdata())
            features.extend([mean_intensity, std_intensity])
            
            # Load associated JSON metadata and extract additional features
            json_filename = mri_file.replace('.nii.gz', '.json')
            if os.path.exists(json_filename):
                metadata = load_mri_metadata(json_filename)
                quality = check_mri_quality(metadata)
                if not quality:
                    print(f"Warning: MRI {mri_file} might be of low quality!")
                magnetic_strength = float(metadata.get("MagneticFieldStrength", "0"))
                features.append(magnetic_strength)
            else:
                # Default value if JSON metadata is not present
                features.extend([0])  
        else:
            # Default values if MRI type is not present
            features.extend([0, 0, 0])
    return features

def load_mri_metadata(json_filename):
    """
    Load the metadata of an MRI from its associated JSON file.
    
    Args:
    - json_filename: String, path to the JSON file
    
    Returns:
    - Dictionary containing MRI metadata
    """
    with open(json_filename, 'r') as file:
        metadata = json.load(file)
    return metadata

def check_mri_quality(metadata):
    """
    Check the quality of an MRI based on its metadata.
    
    Args:
    - metadata: Dictionary containing MRI metadata
    
    Returns:
    - Boolean indicating whether the MRI meets the quality criteria
    """
    required_magnet_strength = "3T"
    return metadata.get("MagneticFieldStrength", "") == required_magnet_strength

def load_data(directory):
    """
    Load and preprocess MRI data from a given directory.
    This function extracts features and labels for each MRI in the directory.
    
    Args:
    - directory: String, path to the directory containing MRI data
    
    Returns:
    - Tuple of numpy arrays containing features and labels
    """
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
                
                # Pad the feature vector if its length is less than the expected length
                while len(feature_vector) < EXPECTED_FEATURE_LENGTH:
                    feature_vector.append(0)  # Append default value
                
                # Debugging: Print out the length of feature vectors and the corresponding MRI filenames
                if len(feature_vector) != EXPECTED_FEATURE_LENGTH:
                    print(f"Unexpected feature vector length for {subject} in Age_{age}: {len(feature_vector)}")
                features.append(feature_vector)
                labels.append(age)
    return np.array(features), np.array(labels)

def train_and_evaluate_model(features, labels):
    """
    Train a RandomForestRegressor model on the features and labels.
    Evaluate the model's performance using mean absolute error (MAE) and mean squared error (MSE).
    
    Args:
    - features: numpy array of extracted features
    - labels: numpy array of target labels (ages)
    
    Returns:
    - Tuple containing the trained model, MAE, and MSE
    """
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    return model, mean_absolute_error(y_val, val_predictions), mean_squared_error(y_val, val_predictions)

def main_execution(training_directory, testing_directory):
    """
    Main execution function to load, preprocess, train, and evaluate MRI data.
    
    Args:
    - training_directory: String, path to the directory containing training data
    - testing_directory: String, path to the directory containing testing data
    """
    start_time = time.time()

    # Load and preprocess training data
    training_features, training_labels = load_data(training_directory)
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)

    # Train and evaluate the model on training data
    model, train_mae, train_mse = train_and_evaluate_model(training_features, training_labels)
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training MSE: {train_mse:.2f}")

    # Load and preprocess testing data
    testing_features, testing_labels = load_data(testing_directory)
    testing_features = scaler.transform(testing_features)

    # Evaluate the model on testing data
    test_predictions = model.predict(testing_features)
    test_mae = mean_absolute_error(testing_labels, test_predictions)
    test_mse = mean_squared_error(testing_labels, test_predictions)
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main_execution("Training", "Testing")
