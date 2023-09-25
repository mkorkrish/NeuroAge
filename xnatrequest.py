import os
import time
import requests


class MRIFileHandler:
    """Class to handle MRI file download, storage, and cleanup."""
    
    BASE_URL = 'https://central.xnat.org/data'
    RETRIES = 3
    DELAY = 5  # delay in seconds
    CHUNK_SIZE = 8192  # Increased chunk size for faster downloads

    def __init__(self, auth):
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
                raise e

    def download_file(self, url, output_path):
        """Download file from URL and save to specified path."""
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
            return
        response = self.robust_request(url)
        with open(output_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                out_file.write(chunk)

    def cleanup_files(self, directory, valid_extensions):
        """Remove files not matching specific MRI types in directory."""
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(valid_extensions):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")

    def get_subject_experiment_for_mr_id(self, mr_id):
        """Retrieve the subject and experiment ID for a specific MR ID."""
        url = f"{self.BASE_URL}/experiments?xsiType=xnat:imageSessionData&columns=ID,project,subject_ID,label&label={mr_id}"
        response = self.robust_request(url)
        try:
            experiments = response.json()['ResultSet']['Result']
        except Exception as e:
            print(f"Error parsing JSON for MR ID {mr_id}. Response content: {response.text}")
            return None, None
        if experiments:
            return experiments[0]['subject_ID'], experiments[0]['ID']
        return None, None


if __name__ == "__main__":
    # CONSTANTS
    AUTH = ('mkorkrish', 'vujpex-fupsut-1tajJe')
    VALID_EXTENSIONS = (
        "T1w.nii", "T1w.nii.gz", "T2w.nii", "T2w.nii.gz", "FLAIR.nii", "FLAIR.nii.gz", 
        "bold.nii", "bold.nii.gz", "dwi.nii", "dwi.nii.gz", "swi.nii", "swi.nii.gz", 
        "asl.nii", "asl.nii.gz", "minIP.nii", "minIP.nii.gz", "T2star.nii", "T2star.nii.gz", 
        "angio.nii", "angio.nii.gz", "fieldmap.nii", "fieldmap.nii.gz", "GRE.nii", "GRE.nii.gz"
    )

    handler = MRIFileHandler(auth=AUTH)

    # Initial cleanup
    handler.cleanup_files("Training", VALID_EXTENSIONS)
    handler.cleanup_files("Testing", VALID_EXTENSIONS)

    # Load MR IDs
    with open(r"C:\Users\mriga\SynologyDrive\Myalo\Technical\ML Team\Data\Programs\Lists\Sorted_MR_IDs_Ages_Train_Test.txt", "r") as file:
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
        subject_id, experiment_id = handler.get_subject_experiment_for_mr_id(mr_id)
        if subject_id and experiment_id:
            url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans"
            scans_response = handler.robust_request(url)
            scans = scans_response.json()['ResultSet']['Result']
            for scan in scans:
                scan_label = scan['ID']
                files_url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/files"
                files_response = handler.robust_request(files_url)
                files = files_response.json()['ResultSet']['Result']
                for file in files:
                    file_name = file['Name']
                    if file_name.endswith(('.nii', '.nii.gz')):
                        file_url = f"{handler.BASE_URL}/projects/OASIS3/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_label}/resources/{file['collection']}/files/{file_name}"
                        base_path = "Training" if mr_id_to_info[mr_id]["type"] == "Training" else "Testing"
                        age_folder = os.path.join(base_path, mr_id_to_info[mr_id]["age"].replace("Age: ", "Age_"))
                        if not os.path.exists(age_folder):
                            os.makedirs(age_folder)
                        output_path = os.path.join(age_folder, file_name)
                        handler.download_file(file_url, output_path)
                handler.cleanup_files(age_folder, VALID_EXTENSIONS)
