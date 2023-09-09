import os
import wget

def download(url: str, destination: str, force: bool = False) -> None:
    """
    Downloads a file from a URL to a destination. 
    If the file already exists, it can optionally overwrite it.
    
    Parameters:
        url (str): The URL of the file to download.
        destination (str): The local path where the file should be saved.
        force (bool, optional): Whether to force download and overwrite. Default is False.
        
    Returns:
        None
    """
    if os.path.exists(destination):
        if force:
            print(f"Overwriting existing file {destination}...")
            os.remove(destination)
        else:
            print(f"The file {destination} already exists. Skipping download.")
            return

    print(f"Downloading file from {url} to {destination}...")
    wget.download(url, destination)
    print(f"\nDownload complete.")