from paths_ import PathLogic, TEST_SET
from dvn.utils.image_utils.plot_images import plot_1_image
from dvn.models.xfeat_.xfeat_methods import XFeatModel, XFEAT_MODELS
from dvn.models.xfeat_.xfeat_utils import save_features_to_folder, load_features_from_folder
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches
import cv2
import numpy as np
import os
import tempfile

def test_():
    xfeat_model = XFeatModel(top_k=4096)
    xfeat_model_steerer = XFeatModel(top_k=4096, model_name=XFEAT_MODELS.STEERER_PRETRAINED)

    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    test_imgs = test_img_sets.get("xfeat_example", None)

    if test_imgs is None:
        raise ValueError("No test images found in the 'xfeat_example' test set.")

    # Load test images
    img1: np.ndarray = cv2.imread(test_imgs[0]) # type: ignore[assignment]
    img2: np.ndarray = cv2.imread(test_imgs[1]) # type: ignore[assignment]

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both test images.")

    def test_saving_and_loading_features():
        """
        Extract features from the test images, save them to disk, then load back, then match the features and visualize the result.
        This is to test the save and load functions and make sure they work correctly.
        """
        # Create a temporary directory for saving features
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            # Step 1: Extract features from both images
            print("\n=== Step 1: Extracting features ===")
            feat1 = xfeat_model.xfeat_detect_and_compute(img1, top_k=4096)
            feat2 = xfeat_model.xfeat_detect_and_compute(img2, top_k=4096)

            print(f"Image 1 features: {feat1['keypoints'].shape[0]} keypoints")
            print(f"Image 2 features: {feat2['keypoints'].shape[0]} keypoints")

            # Step 2: Save features to disk
            print("\n=== Step 2: Saving features to disk ===")
            save_features_to_folder(feat1, temp_dir, "img1")
            save_features_to_folder(feat2, temp_dir, "img2")

            # Step 3: Load features back from disk
            print("\n=== Step 3: Loading features from disk ===")
            loaded_feat1 = load_features_from_folder(temp_dir, "img1", device='cuda')
            loaded_feat2 = load_features_from_folder(temp_dir, "img2", device='cuda')

            print(f"Loaded Image 1 features: {loaded_feat1['keypoints'].shape[0]} keypoints")
            print(f"Loaded Image 2 features: {loaded_feat2['keypoints'].shape[0]} keypoints")

            # Step 4: Verify that loaded features match original features
            print("\n=== Step 4: Verifying feature integrity ===")
            # Check keypoints
            keypoints_match = np.allclose(
                feat1['keypoints'].cpu().numpy(),
                loaded_feat1['keypoints'].cpu().numpy()
            )
            # Check descriptors
            descriptors_match = np.allclose(
                feat1['descriptors'].cpu().numpy(),
                loaded_feat1['descriptors'].cpu().numpy()
            )
            # Check scores
            scores_match = np.allclose(
                feat1['scores'].cpu().numpy(),
                loaded_feat1['scores'].cpu().numpy()
            )

            print(f"Keypoints match: {keypoints_match}")
            print(f"Descriptors match: {descriptors_match}")
            print(f"Scores match: {scores_match}")

            if not (keypoints_match and descriptors_match and scores_match):
                raise ValueError("Loaded features do not match original features!")

            # Step 5: Match features using the loaded data
            print("\n=== Step 5: Matching loaded features ===")
            mkpts_0, mkpts_1 = xfeat_model.xfeat_match_sparse_default(loaded_feat1, loaded_feat2)

            print(f"Number of matches: {len(mkpts_0)}")

            # Step 6: Visualize the matches
            print("\n=== Step 6: Visualizing matches ===")
            output_canvas = warp_corners_and_draw_matches(
                mkpts_0,
                mkpts_1,
                img1,
                img2,
                draw_match_lines=True
            )

            if output_canvas is None:
                print("Visualization failed")
                return

            # Convert BGR to RGB and plot
            output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
            plot_1_image(
                image=output_rgb,
                title=f'Save/Load Test: {len(mkpts_0)} matches',
                tight_layout=True
            )

            print("\n=== Test completed successfully! ===")

    def test_saving_and_loading_features_DENSE():
        """
        Extract DENSE features from the test images, save them to disk, then load back,
        then match the features using the steerer model and visualize the result.
        This tests that dense features can be saved/loaded correctly (especially without scores).
        """
        # Create a temporary directory for saving features
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\nUsing temporary directory: {temp_dir}")

            # Step 1: Extract DENSE features from both images using the steerer model
            print("\n=== Step 1: Extracting DENSE features ===")
            feat1 = xfeat_model_steerer.xfeat_detect_and_compute_DENSE(img1, top_k=8000)
            feat2 = xfeat_model_steerer.xfeat_detect_and_compute_DENSE(img2, top_k=8000)

            print(f"Image 1 DENSE features: {feat1['keypoints'].shape[0]} keypoints")
            print(f"Image 2 DENSE features: {feat2['keypoints'].shape[0]} keypoints")
            print(f"Image 1 has 'scores' key: {'scores' in feat1}")
            print(f"Image 2 has 'scores' key: {'scores' in feat2}")
            print(f"Image 1 has 'scales' key: {'scales' in feat1}")
            print(f"Image 2 has 'scales' key: {'scales' in feat2}")

            # Step 2: Save features to disk (will test handling of missing scores)
            print("\n=== Step 2: Saving DENSE features to disk ===")
            save_features_to_folder(feat1, temp_dir, "img1_dense")
            save_features_to_folder(feat2, temp_dir, "img2_dense")

            # Step 3: Load features back from disk
            print("\n=== Step 3: Loading DENSE features from disk ===")
            loaded_feat1 = load_features_from_folder(temp_dir, "img1_dense", device='cuda')
            loaded_feat2 = load_features_from_folder(temp_dir, "img2_dense", device='cuda')

            print(f"Loaded Image 1 features: {loaded_feat1['keypoints'].shape[0]} keypoints")
            print(f"Loaded Image 2 features: {loaded_feat2['keypoints'].shape[0]} keypoints")
            print(f"Loaded Image 1 has 'scores' key: {'scores' in loaded_feat1}")
            print(f"Loaded Image 2 has 'scores' key: {'scores' in loaded_feat2}")
            print(f"Loaded Image 1 has 'scales' key: {'scales' in loaded_feat1}")
            print(f"Loaded Image 2 has 'scales' key: {'scales' in loaded_feat2}")

            # Step 4: Verify that loaded features match original features
            print("\n=== Step 4: Verifying feature integrity ===")
            # Check keypoints
            keypoints_match = np.allclose(
                feat1['keypoints'].cpu().numpy(),
                loaded_feat1['keypoints'].cpu().numpy()
            )
            # Check descriptors
            descriptors_match = np.allclose(
                feat1['descriptors'].cpu().numpy(),
                loaded_feat1['descriptors'].cpu().numpy()
            )
            # Check scales if present
            scales_match = True
            if 'scales' in feat1 and 'scales' in loaded_feat1:
                scales_match = np.allclose(
                    feat1['scales'].cpu().numpy(),
                    loaded_feat1['scales'].cpu().numpy()
                )

            print(f"Keypoints match: {keypoints_match}")
            print(f"Descriptors match: {descriptors_match}")
            print(f"Scales match: {scales_match}")

            if not (keypoints_match and descriptors_match and scales_match):
                raise ValueError("Loaded DENSE features do not match original features!")

            # Step 5: Match features using the loaded data with steerer model
            print("\n=== Step 5: Matching loaded DENSE features ===")
            mkpts_0, mkpts_1, rot = xfeat_model_steerer.xfeat_match_semi_dense_steerer(
                loaded_feat1,
                loaded_feat2,
                min_cossim=0.82
            )

            print(f"Number of matches: {len(mkpts_0)}, rotation: {rot}")

            # Step 6: Visualize the matches
            print("\n=== Step 6: Visualizing matches ===")
            output_canvas = warp_corners_and_draw_matches(
                mkpts_0,
                mkpts_1,
                img1,
                img2,
                draw_match_lines=True
            )

            if output_canvas is None:
                print("Visualization failed")
                return

            # Convert BGR to RGB and plot
            output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
            plot_1_image(
                image=output_rgb,
                title=f'DENSE Save/Load Test: {len(mkpts_0)} matches (rot={rot})',
                tight_layout=True
            )

            print("\n=== DENSE Test completed successfully! ===")

    # Run the tests
    # test_saving_and_loading_features()
    test_saving_and_loading_features_DENSE()

if __name__ == '__main__':
    test_()