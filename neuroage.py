import os
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask

def preprocess_mri(input_filename, output_prefix):
    # Load the MRI scan using nibabel
    img = nib.load(input_filename)
    img_data = img.get_fdata()
    
    # Create a brain mask using nilearn
    mask = compute_epi_mask(img)
    mask_data = mask.get_fdata()
    
    # Apply the mask to get the brain-extracted data
    brain_extracted_data = img_data * mask_data
    
    # Create a new NIFTI image using the extracted data
    brain_extracted = nib.Nifti1Image(brain_extracted_data, img.affine)
    
    # Save the brain-extracted image
    brain_extracted_filename = f"{output_prefix}_brain_extracted.nii.gz"
    nib.save(brain_extracted, brain_extracted_filename)
    
    # For now, let's stop here and ensure this step works correctly
    return brain_extracted_filename

# Input files based on provided data
input_files = [
    r"Data\OAS30194_MR_d5837\scans\anat1-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-01_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat2-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-02_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat3-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-01_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat4-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-02_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat5-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-03_T1w.nii.gz"
]

def preprocess_mri(input_filename, output_prefix):
    # Load the MRI scan using nibabel
    img = nib.load(input_filename)
    img_data = img.get_fdata()
    
    # Create a brain mask using nilearn
    mask = compute_epi_mask(img)
    mask_data = mask.get_fdata()
    
    # Apply the mask to get the brain-extracted data
    brain_extracted_data = img_data * mask_data

    # Intensity normalization (z-score normalization)
    mean_intensity = np.mean(brain_extracted_data[brain_extracted_data > 0])
    std_intensity = np.std(brain_extracted_data[brain_extracted_data > 0])
    normalized_data = (brain_extracted_data - mean_intensity) / std_intensity

    # Create a new NIFTI image using the normalized data
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)

    # Save the normalized image
    normalized_filename = f"{output_prefix}_normalized.nii.gz"
    nib.save(normalized_img, normalized_filename)
    
    return normalized_filename

# Process each file
for input_file in input_files:
    output_prefix = os.path.splitext(input_file)[0]  # Use input filename as the base for output
    preprocess_mri(input_file, output_prefix)
