import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

VALID_FILES = [
    "bold", "T1w", "T2w", "dwi", "swi", "asl", "FLAIR"
]

class MRIFileHandler:
    """Handles MRI file operations including downloading, storage, and cleanup."""
    
    BASE_URL = 'https://central.xnat.org/data'
    RETRIES = 3  # Number of retries for failed requests
    DELAY = 5  # Delay in seconds between retries
    CHUNK_SIZE = 8192  # Chunk size for downloads
    MAX_WORKERS = 10  # Number of parallel downloads

    def __init__(self, auth):
        """Initialize MRIFileHandler with authentication credentials."""
        self.session = requests.Session()
        self.session.auth = auth

    def robust_request(self, url):
        """Make a GET request with retries on failure."""
        for i in range(self.RETRIES):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if i < self.RETRIES - 1:
                    time.sleep(self.DELAY)
                    continue
                else:
                    print(f"Error with URL {url}: {e}")
                    return None

    def download_file(self, url, output_path):
        """Download a file from a URL and save it to the specified path."""
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
            return
        response = self.robust_request(url)
        if response:
            with open(output_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    out_file.write(chunk)
            print(f"Downloaded to: {output_path}")

    def cleanup_files(self, directory):
        """Remove files that don't match specific MRI types from the directory."""
        print(f"Cleaning up directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if not any(valid_file in file for valid_file in VALID_FILES):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")

    def get_subject_experiment_for_mr_id(self, mr_id):
        """Retrieve subject and experiment ID for a specific MR ID."""
        url = f"{self.BASE_URL}/experiments?xsiType=xnat:imageSessionData&columns=ID,project,subject_ID,label&label={mr_id}"
        response = self.robust_request(url)
        if response:
            try:
                experiments = response.json()['ResultSet']['Result']
            except Exception as e:
                print(f"Error parsing JSON for MR ID {mr_id}. Response content: {response.text}")
                return None, None
            if experiments:
                return experiments[0]['subject_ID'], experiments[0]['ID']
        return None, None

    def download_files_in_parallel(self, download_list):
        """Download files in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(self.download_file, url, output_path) for url, output_path in download_list]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred during download: {e}")

if __name__ == "__main__":
    # CONSTANTS
    AUTH = ('mkorkrish', 'vujpex-fupsut-1tajJe')

    handler = MRIFileHandler(auth=AUTH)

    # Initial cleanup of files not matching the valid MRI types
    print("Starting initial cleanup...")
    handler.cleanup_files("Data\Training")
    handler.cleanup_files("Data\Testing")
    print("Initial cleanup completed.")

    # Load MR IDs from the file
    print("Loading MR IDs...")
    with open(r"C:\Users\mriga\SynologyDrive\Myalo\Lists\Sorted_MR_IDs_Ages_Train_Test.txt", "r") as file:
        lines = file.readlines()

    data_type = None
    mr_id_to_info = {}
    for line in lines:
        stripped_line = line.strip()
        if "Training Data:" in stripped_line:
            data_type = "Data\Training"
        elif "Testing Data:" in stripped_line:
            data_type = "Data\Testing"
        elif stripped_line and "," in stripped_line:
            values = stripped_line.split(',')
            mr_id = values[0].strip()
            age = values[1].strip()
            mr_id_to_info[mr_id] = {"type": data_type, "age": age}

    print(f"Loaded {len(mr_id_to_info)} MR IDs.")

    # Process each MR ID to extract relevant URLs and save paths
    download_list = []
    for mr_id, info in mr_id_to_info.items():
        print(f"Processing MR ID: {mr_id}")
        subject_id, experiment_id = handler.get_subject_experiment_for_mr_id(mr_id)
        if subject_id and experiment_id:
            url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans"
            scans_response = handler.robust_request(url)
            if scans_response:
                try:
                    scans = scans_response.json()['ResultSet']['Result']
                except Exception as e:
                    print(f"Error parsing scans JSON for MR ID {mr_id}. Response content: {scans_response.text}")
                    continue

                for scan in scans:
                    scan_label = scan['ID']
                    files_url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/files"
                    files_response = handler.robust_request(files_url)
                    if files_response:
                        try:
                            files = files_response.json()['ResultSet']['Result']
                        except Exception as e:
                            print(f"Error parsing files JSON for MR ID {mr_id}, Scan {scan_label}. Response content: {files_response.text}")
                            continue

                        for file in files:
                            file_name = file['Name']
                            if any(valid_file in file_name for valid_file in VALID_FILES):
                                file_url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/resources/{file['collection']}/files/{file_name}"
                                base_path = info["type"]
                                age_folder = os.path.join(base_path, info["age"].replace("Age: ", "Age_"))
                                if not os.path.exists(age_folder):
                                    os.makedirs(age_folder)
                                output_path = os.path.join(age_folder, file_name)
                                download_list.append((file_url, output_path))

    print("Processing completed. Starting parallel downloads...")
    handler.download_files_in_parallel(download_list)
    print("All downloads completed.")
