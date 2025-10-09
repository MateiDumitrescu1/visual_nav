import time
from typing import Sequence
import cv2
import numpy as np
import onnxruntime
from dvn.models.optical_flow.neuflow_v2_.neuflow_utils import draw_flow, check_model
from paths_ import PathLogic, TEST_SET, models_dir
class NeuFlowV2:

    def __init__(self, path: str):
        check_model(path)

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def estimate_flow(self, img_prev: np.ndarray, img_now: np.ndarray) -> np.ndarray:
        input_tensors = self.prepare_inputs(img_prev, img_now)

        # Perform inference on the image
        outputs = self.inference(input_tensors)

        return self.process_output(outputs[0])

    def prepare_inputs(self, img_prev: np.ndarray, img_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.img_height, self.img_width = img_now.shape[:2]

        input_prev = self.prepare_input(img_prev)
        input_now = self.prepare_input(img_now)

        return input_prev, input_now

    def prepare_input(self, img: np.ndarray) -> np.ndarray:
        # Resize input image
        input_img = cv2.resize(img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensors: tuple[np.ndarray, np.ndarray]) -> Sequence:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensors[0],
                                                       self.input_names[1]: input_tensors[1]})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output) -> np.ndarray:
        flow = output.squeeze().transpose(1, 2, 0)

        return cv2.resize(flow, (self.img_width, self.img_height))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


#! ---------------- TESTING ----------------
def test_estimate_flow():
    """Test NeuFlowV2 optical flow estimation on test images."""
    # Initialize model
    model_path = f"{models_dir}/neuflow_sintel.onnx"
    neuflow = NeuFlowV2(model_path)

    # Load images
    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.OPTICAL_FLOW_TEST)
    test_imgs = test_img_sets.get("0", None)

    if test_imgs is None or len(test_imgs) < 2:
        raise ValueError("Not enough test images found in the '0' test set.")

    # Read the images
    img1 = cv2.imread(test_imgs[0])
    img2 = cv2.imread(test_imgs[1])

    if img1 is None or img2 is None:
        raise ValueError("Failed to load test images.")

    # Estimate optical flow
    flow = neuflow.estimate_flow(img1, img2)

    # Plot flow visualization
    flow_viz = draw_flow(flow, img1)

    # Display results
    cv2.imshow("Optical Flow", flow_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return flow

if __name__ == '__main__':
    test_estimate_flow()