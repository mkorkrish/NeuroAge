from concurrent.futures import as_completed

def download_files_in_parallel(self, download_list):
    """Download files in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
        # Submit tasks and store the resulting futures in a list
        futures = [executor.submit(self.download_file, url, output_path) for url, output_path in download_list]
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                # If there's a result (which there isn't in this case), you could retrieve it with future.result()
                future.result()
            except Exception as e:
                print(f"Error occurred during download: {e}")