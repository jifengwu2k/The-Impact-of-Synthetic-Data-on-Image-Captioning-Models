import argparse
import logging
import os

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline

from tqdm import tqdm


def find_longest_concatenated_string(
        strings: list[str],
        limit: int = 77,
        concatenate_by: str = ', '
):
    concatenated_components, concatenated_length = helper_function(
        strings,
        limit,
        ('hyper realistic photograph',),
        26,
        concatenate_by
    )

    return concatenate_by.join(concatenated_components)


def helper_function(
        strings: list[str],
        limit: int,
        current_concatenated_components: tuple[str, ...],
        current_concatenated_length: int,
        concatenate_by: str
):
    if not strings:
        return current_concatenated_components, current_concatenated_length
    else:
        current_string, remaining_strings = strings[0], strings[1:]

        # Try adding the current string to the current_concatenated
        new_concatenated_components = current_concatenated_components + (current_string,)

        if current_concatenated_components:
            new_concatenated_length = current_concatenated_length + len(concatenate_by) + len(current_string)
        else:
            new_concatenated_length = current_concatenated_length + len(current_string)

        # If adding the current string does not exceed the limit, recurse
        if new_concatenated_length <= limit:
            with_current, with_current_length = helper_function(
                remaining_strings,
                limit,
                new_concatenated_components,
                new_concatenated_length,
                concatenate_by
            )

            without_current, without_current_length = helper_function(
                remaining_strings,
                limit,
                current_concatenated_components,
                current_concatenated_length,
                concatenate_by
            )

            # Compare the results with and without the current string
            if with_current_length >= without_current_length:
                return with_current, with_current_length
            else:
                return without_current, without_current_length
        else:
            # If adding the current string exceeds the limit, skip it and recurse
            return helper_function(
                remaining_strings,
                limit,
                current_concatenated_components,
                current_concatenated_length,
                concatenate_by
            )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file-names-and-captions-csv-path', type=str, required=True)
    parser.add_argument('--real-image-dir', type=str, required=True)
    parser.add_argument('--synthetic-image-dir', type=str, required=True)
    args = parser.parse_args()

    train_file_names_and_captions_csv_path = args.train_file_names_and_captions_csv_path
    train_file_names_and_captions_dataframe = pd.read_csv(train_file_names_and_captions_csv_path)

    real_image_dir = args.real_image_dir
    synthetic_image_dir = args.synthetic_image_dir

    pipe_ = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pipe_on_device = pipe_.to(device)

    # if one wants to set `leave=False`
    pipe_on_device.set_progress_bar_config(leave=False)

    # if one wants to disable `tqdm`
    pipe_on_device.set_progress_bar_config(disable=True)

    os.makedirs(synthetic_image_dir, exist_ok=True)

    # https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3
    negative_prompt = ('blurry, plastic, grainy, duplicate, deformed , disfigured, poorly drawn, bad anatomy, '
                       'wrong anatomy, extra limb, missing limb, floating limb, disconnected limb, mutated hands and '
                       'fingers, text, name, signature, watermark, worst quality, jpeg artifacts, boring composition, '
                       'uninteresting')

    for file_name, dataframe_with_file_name in tqdm(train_file_names_and_captions_dataframe.groupby('file_name')):
        assert isinstance(file_name, str)
        synthetic_image_path = os.path.join(synthetic_image_dir, file_name)

        if os.path.exists(synthetic_image_path):
            continue

        prompt = find_longest_concatenated_string(list(dataframe_with_file_name['caption']))

        image = pipe_on_device(
          prompt,
          negative_prompt=negative_prompt
        ).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
        # Now to display an image you can either save it such as:

        image.save(synthetic_image_path)

