import os
import sys
import numpy as np
from skimage import morphology
import mediapipe as mp
import warnings
import torch
import cv2
from torchvision.transforms import Compose
from pathlib import Path
lama_path = Path(__file__).resolve().parent.parent.parent / "AnyEdit_Collection/other_modules"
sys.path.insert(0, str(lama_path))
from AnyEdit_Collection.other_modules.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
import glob
from AnyEdit_Collection.other_modules.DPT.util import io
from AnyEdit_Collection.other_modules.DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from termcolor import cprint
from AnyEdit_Collection.other_modules.uniformer.mmseg.datasets.pipelines import Compose
from AnyEdit_Collection.other_modules.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from AnyEdit_Collection.other_modules.uniformer.mmseg.core.evaluation import get_palette
from AnyEdit_Collection.other_modules.HED import HEDdetector

warnings.filterwarnings("ignore")


def img2sketch(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply edge detection
    edges = cv2.Canny(gaussian_blur, 50, 150)
    # Invert the colors of the edges
    edges = cv2.bitwise_not(edges)
    # Save the result
    cv2.imwrite(output_path, edges)

def run_depth(input_image, output_image, model):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input image
        output_path (str): path to output image
        model_path (str): path to saved model
    """

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    img = io.read_image(input_image)

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    cprint(output_image, 'red')
    io.write_depth(output_image, prediction, bits=2, absolute_depth=False)


def img2depth(image_path, output_path, depth_anything = None):
    '''
    change to depth anything V2
    '''
    

    if os.path.isfile(image_path):
        if image_path.endswith('txt'):
            with open(image_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [image_path]
    else:
        filenames = glob.glob(os.path.join(image_path, '**/*'), recursive=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(output_path, depth)


def img2seg(image_path, output_path, model):
    # Load the image
    img = cv2.imread(image_path)
    result = inference_segmentor(model, img)
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    # Save the result
    cv2.imwrite(output_path, res_img)
