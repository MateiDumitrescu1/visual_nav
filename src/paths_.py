import os
import sys
from enum import StrEnum
# This file defines project-relative paths used across the package.
# The paths are intentionally not validated at import time so a fresh public
# clone can import the package before local datasets and model files are added.

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', ))

def check_dir_exists(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

#! data dir
data_dir = os.path.abspath(os.path.join(project_root_dir, 'data'))

#! main dirs in the `data` folder
video_dir = os.path.abspath(os.path.join(data_dir, 'videos'))
images_dir = os.path.abspath(os.path.join(data_dir, 'images'))
models_dir = os.path.abspath(os.path.join(data_dir, 'models'))
output_dir = os.path.abspath(os.path.join(data_dir, 'output'))

demo_output_dir = os.path.abspath(os.path.join(output_dir, 'demo_output'))

#! test image sets dir
test_image_sets_dir = os.path.abspath(os.path.join(images_dir, 'test_image_sets'))


class TEST_SET(StrEnum):
    OPTICAL_FLOW_TEST = 'optical_flow_test'
    FEATURE_MATCHING_TEST = 'feature_matching_test'
class PathLogic:
        
    @staticmethod
    def get_test_image_sets(test_root_dir: TEST_SET) -> dict[str, list[str]]:
        """
        ### Params:
        test_root_dir: str
            Name of the subfolder in `test_image_sets_dir` containing test image sets.
            Example values: 'optical_flow_test', 'feature_matching_test'
        ### Returns a dictionary where keys are the names of the test sets (subfolder names in `test_image_sets_dir`) 
        and the values are lists of file paths of images in those subfolders.
        """
        if test_root_dir not in TEST_SET:
            raise ValueError(f"Invalid test_root_dir: {test_root_dir}. Must be one of {[e.value for e in TEST_SET]}")
        
        test_dir = test_image_sets_dir + '/' + test_root_dir
        check_dir_exists(test_dir)
        test_sets = {}
        for entry in os.listdir(test_dir):
            entry_path = os.path.join(test_dir, entry)
            if os.path.isdir(entry_path):
                image_files = [os.path.join(entry_path, f) for f in os.listdir(entry_path)
                            if os.path.isfile(os.path.join(entry_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                test_sets[entry] = sorted(image_files)
        return test_sets

#! ---------------- TESTING ----------------
def test_get_test_image_sets():
    of_test_sets = PathLogic.get_test_image_sets(TEST_SET.OPTICAL_FLOW_TEST)
    feature_matching_test_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    print(of_test_sets)
    print(feature_matching_test_sets)
    
def test_():
    print(f"current dir { current_dir } ")
    print(f"project root dir { project_root_dir } ")
    print(f"data dir { data_dir } ")
    print(f"video dir { video_dir } ")
    print(f"data images dir { images_dir } ")
    test_get_test_image_sets()

if __name__ == '__main__':
    test_()