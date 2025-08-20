import os
import sys
import time
import cv2
import torch
import argparse
import numpy as np
from mmedit.apis import init_model, restoration_inference

#Add the parent directory to sys.path to locate segment_anything
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='../clipiqa_attribute_test.py', help='Test config file path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument('--device', type=int, default=1, help='CUDA device ID')
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    image_root = ''
    sam_checkpoint = ''
    sodmaskpath = ''
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Update this list based on dataset to process
    dataset_lists = ['VT821', 'VT1000', 'VT5000/Test']

    for dataset in dataset_lists:
        args = parse_args()
        model = init_model(args.config, args.checkpoint, device=torch.device('cuda', args.device))

        rgb_folder = os.path.join(image_root, dataset, 'RGB')
        names = [f for f in os.listdir(rgb_folder) if f.endswith('.jpg')]

        total_time = 0
        frame_count = 0

        for index, filename in enumerate(names, 1):
            png_name = filename.replace('.jpg', '.png')
            mask_path = os.path.join(sodmaskpath, dataset, png_name)

            img = cv2.imread(mask_path)
            if img is None:
                continue  # Skip if mask image is not found

            rgb_path = os.path.join(rgb_folder, filename)
            RGB = cv2.imread(rgb_path)
            if RGB is None:
                continue  # Skip if RGB image is not found

            RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)

            t_path = os.path.join(image_root, dataset, 'T', filename)
            T = cv2.imread(t_path)
            if T is None:
                continue  # Skip if thermal image is not found

            T = cv2.cvtColor(T, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray1 = gray.copy()

            start_time = time.time()
            output, attributes = restoration_inference(model, rgb_path, return_attributes=True)
            attributes = attributes.float().detach().cpu().numpy()[0]
            end_time = time.time()

            # Update total inference time
            total_time += (end_time - start_time)
            frame_count += 1

            savepath = os.path.join(sodmaskpath, 'result719unalign', dataset)
            os.makedirs(savepath, exist_ok=True)
            print(f'Saving: {dataset}, {filename}, Index: {index}')

            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = np.concatenate(contours, axis=0)
                x, y, w, h = cv2.boundingRect(max_contour)
            else:
                x, y, w, h = 0, 0, 0, 0

            input_box = np.array([x, y, x + w, y + h])
            gray_padded = np.pad(thresh, ((0, 160), (0, 0)), 'constant', constant_values=(0))
            gray_resized = np.array(cv2.resize(gray_padded, (256, 256)), dtype=np.float32) / 4 - 32

            # Determine whether to use T or RGB image based on attribute thresholds
            predictor.set_image(T if attributes[1] < 0.01 or attributes[4] > 0.86 else RGB)

            masks, _, _ = predictor.predict(
                point_coords=None,
                box=input_box[None, :],
                mask_input=gray_resized[None, :, :],
                multimask_output=False,
            )

            mask = masks[0]
            gray1[gray1 < 250] = 0
            mask = sigmoid(mask) * 255

            gray1 = cv2.resize(gray1, (mask.shape[1], mask.shape[0]))
            fmask = np.maximum(gray1, mask)

            save_file_path = os.path.join(savepath, filename)
            cv2.imwrite(save_file_path, fmask)

        # Calculate and print FPS
        if frame_count > 0:
            avg_time_per_frame = total_time / frame_count
            fps = 1.0 / avg_time_per_frame
            print(f'FPS for {dataset}: {fps:.2f}')
        else:
            print(f'No valid frames processed in {dataset}')


if __name__ == "__main__":
    main()