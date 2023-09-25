import os
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn import image, datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# ------------------ CONSTANTS ------------------
template = datasets.fetch_icbm152_2009()
t1_template = template.t1

# ------------------ MRI PROCESSING FUNCTIONS ------------------

def compute_mean_std_intensity(voxel_data):
    """Compute the mean and standard deviation of the intensity of brain voxels."""
    brain_voxels = voxel_data[voxel_data > 0]
    return np.mean(brain_voxels), np.std(brain_voxels)

def preprocess_mri(input_filename):
    """Preprocess MRI data for model training and evaluation."""
    processed_filename = input_filename.replace('.nii.gz', "_processed.nii.gz")
    if os.path.exists(processed_filename):
        return nib.load(processed_filename)

    img = nib.load(input_filename)
    img_data = img.get_fdata()
    if img_data.ndim == 4:  # Handle 4D data by averaging over the last dimension
        img_data = np.mean(img_data, axis=-1)

    mask = compute_epi_mask(img)
    brain_extracted_data = img_data * mask.get_fdata()
    normalized_data = (brain_extracted_data - np.mean(brain_extracted_data)) / np.std(brain_extracted_data)
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)
    resampled_img = image.resample_to_img(normalized_img, nib.load(t1_template))
    nib.save(resampled_img, processed_filename)

    return resampled_img

def extract_features_from_preprocessed_mri_files(**mri_files):
    """Extract features (mean and std intensity) from preprocessed MRI data."""
    features = []
    for mri_type, mri_file in mri_files.items():
        if mri_file:
            preprocessed_img = preprocess_mri(mri_file)
            mean_intensity, std_intensity = compute_mean_std_intensity(preprocessed_img.get_fdata())
            features.extend([mean_intensity, std_intensity])
        else:
            features.extend([0, 0])  # Default values if MRI type not present
    return features

# ------------------ DATA LOADING AND TRAINING FUNCTIONS ------------------

def load_data(directory):
    """Load and preprocess MRI data from a given directory."""
    features, labels = [], []
    for age in range(42, 98):
        age_path = os.path.join(directory, f"Age_{age}")
        if os.path.isdir(age_path):
            subjects = set([f.split('_')[0] for f in os.listdir(age_path)])
            for subject in subjects:
                mri_files = {mri_type: os.path.join(age_path, filename) 
                             for mri_type, filename in [(f.split('_')[1].split('.')[0], f) 
                             for f in os.listdir(age_path) if f.startswith(subject)]}
                features.append(extract_features_from_preprocessed_mri_files(**mri_files))
                labels.append(age)
    return np.array(features), np.array(labels)

def train_and_evaluate_model(features, labels):
    """Train a model on the features and labels and evaluate its performance."""
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    return model, mean_absolute_error(y_val, val_predictions), mean_squared_error(y_val, val_predictions)

# ------------------ MAIN EXECUTION ------------------

def main_execution(training_directory, testing_directory):
    """Main execution function."""
    start_time = time.time()

    # Load and preprocess training data
    training_features, training_labels = load_data(training_directory)
    model, train_mae, train_mse = train_and_evaluate_model(training_features, training_labels)
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training MSE: {train_mse:.2f}")

    # Load and preprocess testing data
    testing_features, testing_labels = load_data(testing_directory)
    test_predictions = model.predict(testing_features)
    test_mae = mean_absolute_error(testing_labels, test_predictions)
    test_mse = mean_squared_error(testing_labels, test_predictions)
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main_execution("Training", "Testing")
