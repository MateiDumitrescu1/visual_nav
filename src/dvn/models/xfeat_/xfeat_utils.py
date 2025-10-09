import os
import numpy as np
import torch

def save_mkpts_to_file(mkpts, output_folder, filename):
    """
    Save matched keypoints to a text file. Each line contains 'x y' coordinates.
    mkpts: iterable of (x, y) pairs (e.g. numpy array shape (N,2))
    output_folder: directory to write the file into
    filename: name of the file ('.txt' will be appended if missing)
    """
    # ensure .txt extension
    if not filename.lower().endswith('.txt'):
        filename = filename + '.txt'

    # create folder if missing
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)

    try:
        with open(file_path, 'w') as f:
            for pt in mkpts:
                # format with 6 decimal places
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        print(f"Saved keypoints to {file_path}")
    except Exception as e:
        print(f"Error saving keypoints to file {file_path}: {e}")

def save_features_to_folder(features: dict, output_folder: str, filename_prefix: str) -> None:
    """
    Save computed XFeat features to a folder as numpy arrays.

    ### Parameters:
        - features: Dictionary containing 'keypoints', 'descriptors', 'scores', and optionally 'image_size'
        - output_folder: Directory path where features will be saved
        - filename_prefix: Prefix for the saved files (e.g., 'image1', 'frame_001')

    ### Saves:
        - {filename_prefix}_keypoints.npy: Keypoint coordinates
        - {filename_prefix}_descriptors.npy: Feature descriptors
        - {filename_prefix}_scores.npy: Feature scores
        - {filename_prefix}_metadata.npz: Additional metadata (image_size, etc.)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Convert torch tensors to numpy if needed
        keypoints = features['keypoints']
        descriptors = features['descriptors']
        scores = features['scores']

        if torch.is_tensor(keypoints):
            keypoints = keypoints.cpu().numpy()
        if torch.is_tensor(descriptors):
            descriptors = descriptors.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()

        # Save each component
        np.save(os.path.join(output_folder, f"{filename_prefix}_keypoints.npy"), keypoints)
        np.save(os.path.join(output_folder, f"{filename_prefix}_descriptors.npy"), descriptors)
        np.save(os.path.join(output_folder, f"{filename_prefix}_scores.npy"), scores)

        # Save metadata (image_size and any other non-array data)
        metadata = {}
        if 'image_size' in features:
            metadata['image_size'] = features['image_size']

        np.savez(os.path.join(output_folder, f"{filename_prefix}_metadata.npz"), **metadata)

        print(f"Saved features to {output_folder}/{filename_prefix}_*.npy")

    except Exception as e:
        raise RuntimeError(f"Error saving features to {output_folder}: {e}")

def load_features_from_folder(folder_path: str, filename_prefix: str, device: str = 'cpu') -> dict:
    """
    Load computed XFeat features from a folder.

    ### Parameters:
        - folder_path: Directory path where features are stored
        - filename_prefix: Prefix of the saved files (e.g., 'image1', 'frame_001')
        - device: Device to load tensors to ('cpu' or 'cuda')

    ### Returns:
        Dictionary containing 'keypoints', 'descriptors', 'scores', and optionally 'image_size'

    ### Raises:
        FileNotFoundError: If required feature files are not found
        RuntimeError: If there's an error loading the files
    """
    try:
        # Load numpy arrays
        keypoints_path = os.path.join(folder_path, f"{filename_prefix}_keypoints.npy")
        descriptors_path = os.path.join(folder_path, f"{filename_prefix}_descriptors.npy")
        scores_path = os.path.join(folder_path, f"{filename_prefix}_scores.npy")
        metadata_path = os.path.join(folder_path, f"{filename_prefix}_metadata.npz")

        # Check if files exist
        if not os.path.exists(keypoints_path):
            raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
        if not os.path.exists(descriptors_path):
            raise FileNotFoundError(f"Descriptors file not found: {descriptors_path}")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Scores file not found: {scores_path}")

        # Load arrays
        keypoints = np.load(keypoints_path)
        descriptors = np.load(descriptors_path)
        scores = np.load(scores_path)

        # Convert to torch tensors
        keypoints_tensor = torch.from_numpy(keypoints).to(device)
        descriptors_tensor = torch.from_numpy(descriptors).to(device)
        scores_tensor = torch.from_numpy(scores).to(device)

        # Create features dictionary
        features = {
            'keypoints': keypoints_tensor,
            'descriptors': descriptors_tensor,
            'scores': scores_tensor,
        }

        # Load metadata if exists
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True)
            if 'image_size' in metadata:
                features['image_size'] = tuple(metadata['image_size'])

        print(f"Loaded features from {folder_path}/{filename_prefix}_*.npy")
        return features

    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error loading features from {folder_path}: {e}")