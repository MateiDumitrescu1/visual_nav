import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', ))
# Add root directory to Python path so this module can be imported from anywhere

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

data_dir = os.path.abspath(os.path.join(project_root_dir, 'data'))

video_dir = os.path.abspath(os.path.join(data_dir, 'videos'))
images_dir = os.path.abspath(os.path.join(data_dir, 'images'))

# print(f"current dir { current_dir } ")
# print(f"project root dir { project_root_dir } ")
# print(f"data dir { data_dir } ")
# print(f"video dir { video_dir } ")
# print(f"data images dir { images_dir } ")
