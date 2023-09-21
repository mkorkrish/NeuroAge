import requests
import os
import time
import ctypes
import shutil

# ---------------------- CONSTANTS ----------------------

BASE_URL = 'https://central.xnat.org/data'
AUTH = ('', '')
RETRIES = 3
DELAY = 5  # Delay in seconds between retries
DISK_SPACE_THRESHOLD = 10 * 1024 * 1024 * 1024  # 10GB

# ---------------------- Request and Download Helpers ----------------------

def robust_request(url):
    """Make a GET request to the specified URL with retries."""
    for i in range(RETRIES):
        try:
            response = session.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if i < RETRIES - 1:
                time.sleep(DELAY)
                continue
            else:
                raise e

def download_file(url, output_path):
    """Download a file from a URL and save it to a specified path."""
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return
    response = robust_request(url)
    with open(output_path, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                out_file.write(chunk)

# ---------------------- Data Extraction Helpers ----------------------

def get_subject_experiment_for_mr_id(mr_id):
    """Retrieve the subject and experiment ID for a specific MR ID."""
    experiments_response = robust_request(f"{BASE_URL}/experiments?xsiType=xnat:imageSessionData&columns=ID,project,subject_ID,label&label={mr_id}")
    try:
        experiments = experiments_response.json()['ResultSet']['Result']
    except Exception as e:
        print(f"Error parsing JSON for MR ID {mr_id}. Response content: {experiments_response.text}")
        return None, None
    if experiments:
        return experiments[0]['subject_ID'], experiments[0]['ID']
    return None, None

# ---------------------- Cleanup Helpers ----------------------

def cleanup_files_in_directory(directory):
    """Remove files not of specific MRI types."""
    valid_extensions = ("T1w.nii", "T1w.nii.gz", "T2w.nii", "T2w.nii.gz", "FLAIR.nii", "FLAIR.nii.gz", 
                        "bold.nii", "bold.nii.gz", "dwi.nii", "dwi.nii.gz", "swi.nii", "swi.nii.gz", "asl.nii", "asl.nii.gz")
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(valid_extensions):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

def set_online_only(path):
    """Set the file or folder at 'path' as online-only."""
    FILE_ATTRIBUTE_UNPINNED = 0x00001000
    try:
        ctypes.windll.kernel32.SetFileAttributesW(path, FILE_ATTRIBUTE_UNPINNED)
    except Exception as e:
        print(f"Error setting {path} as online-only. Error: {e}")

# ---------------------- DISK MANAGEMENT FUNCTIONS ----------------------

def ensure_free_space(required_space):
    total, used, free = shutil.disk_usage('/')
    if free < required_space:
        onedrive_folders = ["Testing", "Training"]
        for onedrive_folder in onedrive_folders:
            print(f"Ensuring space by setting {onedrive_folder} folder to online-only...")
            for folder in os.listdir(onedrive_folder):
                folder_path = os.path.join(onedrive_folder, folder)
                set_online_only(folder_path)

# ---------------------- Main Script ----------------------

if __name__ == "__main__":
    start_time = time.time()
    session = requests.Session()
    session.auth = AUTH

    # Ensure enough free space before starting the download process
    print("Ensuring sufficient disk space before starting downloads...")
    ensure_free_space(DISK_SPACE_THRESHOLD)

    training_path = "Training"
    testing_path = "Testing"

    # Initial cleanup
    cleanup_files_in_directory(training_path)
    cleanup_files_in_directory(testing_path)

    # Load MR IDs
    
        lines = file.readlines()
    mr_ids = [line.split(',')[0].strip() for line in lines if not line.startswith(("Training Data", "Testing Data", "\n"))]
    mr_id_to_info = {}
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(("Training Data", "Testing Data")):
            data_type = stripped_line.replace(" Data:", "")
        elif stripped_line and "," in stripped_line:
            values = stripped_line.split(',')
            mr_id = values[0].strip()
            age = values[1].strip()
            mr_id_to_info[mr_id] = {"type": data_type, "age": age}

    # Process MR IDs
    for mr_id in mr_ids:
        subject_id, experiment_id = get_subject_experiment_for_mr_id(mr_id)
        if subject_id and experiment_id:
            scans_response = robust_request(f"{BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans")
            scans = scans_response.json()['ResultSet']['Result']
            for scan in scans:
                scan_label = scan['ID']
                files_response = robust_request(f"{BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/files")
                files = files_response.json()['ResultSet']['Result']
                for file in files:
                    file_name = file['Name']
                    if file_name.endswith(('.nii', '.nii.gz')):
                        file_url = f"{BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/resources/{file['collection']}/files/{file_name}"
                        base_path = training_path if mr_id_to_info[mr_id]["type"] == "Training" else testing_path
                        age_folder = os.path.join(base_path, mr_id_to_info[mr_id]["age"].replace("Age: ", "Age_"))
                        if not os.path.exists(age_folder):
                            os.makedirs(age_folder)
                        output_path = os.path.join(age_folder, file_name)
                        download_file(file_url, output_path)
                cleanup_files_in_directory(age_folder)
                set_online_only(age_folder)
            
            # Ensure free space after processing all scans for the current MRI ID
            ensure_free_space(DISK_SPACE_THRESHOLD)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
