import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, 'data'))
data_images_dir = os.path.abspath(os.path.join(data_dir, 'images'))
test_image_pairs_dir = os.path.abspath(os.path.join(data_images_dir, 'test_image_pairs')) # contains image pairs for testing

print(f"data dir { data_dir } ")
print(f"data images dir { data_images_dir } ")
print(f"test image pairs dir { test_image_pairs_dir } ")