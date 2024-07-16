import os
import tarfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub import login

# modified original file for downloading noise data:
# (login and additional repo info were necessary)
# https://github.com/DBD-research-group/BirdSet/blob/main/resources/utils/download_background_noise.py
def main():
    repo_id = "DBD-research-group/BirdSet"
    filenames = ["dcase18_shard_0001.tar.gz", "dcase18_shard_0002.tar.gz"]
    subfolder = "dcase18"
    revision = "data"

    local_dir = "./noise_data"
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    # Download the files
    for filename in filenames:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            revision=revision,
            force_download=True
        )

    output_directory = os.path.join(local_dir, "background_noise")
    os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(filenames, desc="Extracting files"):
        file_path = os.path.join(local_dir, subfolder, filename)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_directory)
        os.remove(file_path)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
