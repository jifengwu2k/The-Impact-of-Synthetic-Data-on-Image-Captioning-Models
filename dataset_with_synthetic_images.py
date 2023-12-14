import os.path

import nltk
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

import random
import typing


nltk.download('punkt')


def sample(sequence: typing.Sequence[typing.Any], k: int) -> list[typing.Any]:
    """
    Sample k items from sequence without replacement.
    """
    n = len(sequence)
    if k <= n:
        return random.sample(sequence, k)
    else:
        return random.sample(sequence, n) + sample(sequence, k - n)


class SampleFromDatasetWithSyntheticData:
    def __init__(
            self,
            length_of_original_dataset: int,
            percentage_of_original_dataset: float,
            percentage_of_synthetic_dataset: float,
    ):
        self.length_of_original_dataset = length_of_original_dataset

        self.apparent_length_of_original_dataset = int(
            self.length_of_original_dataset * percentage_of_original_dataset
        )

        self.sampled_indices_of_original_dataset = sample(
            sequence=range(self.length_of_original_dataset),
            k=self.apparent_length_of_original_dataset,
        )

        self.apparent_length_of_synthetic_dataset = int(
            self.length_of_original_dataset * percentage_of_synthetic_dataset
        )

        self.sampled_indices_of_synthetic_dataset = sample(
            sequence=range(self.length_of_original_dataset),
            k=self.apparent_length_of_synthetic_dataset,
        )

        self.apparent_length_of_dataset = self.apparent_length_of_original_dataset + self.apparent_length_of_synthetic_dataset

    def __len__(self):
        return self.apparent_length_of_dataset

    def __getitem__(self, idx):
        # Decide whether idx refers to an original image or a synthetic image
        # And map idx to index in dataframe
        if idx < self.apparent_length_of_original_dataset:
            index_in_original_dataset = self.sampled_indices_of_original_dataset[idx]
            is_synthetic = False
        else:
            index_in_original_dataset = self.sampled_indices_of_synthetic_dataset[
                idx - self.apparent_length_of_original_dataset]
            is_synthetic = True

        return index_in_original_dataset, is_synthetic


class CSVDatasetWithSyntheticImages(Dataset):
    """Dataset class for torch.utils.DataLoader"""

    def __init__(self,
                 vocab,
                 dataframe: pd.DataFrame,
                 original_image_dir: str,
                 original_image_ratio: float,
                 synthetic_image_dir: str,
                 synthetic_image_ratio: float,
                 transform
                 ):
        self.vocab = vocab

        self.dataframe = dataframe

        self.sampler = SampleFromDatasetWithSyntheticData(
            len(self.dataframe),
            original_image_ratio,
            synthetic_image_ratio
        )

        self.original_image_dir = original_image_dir
        self.synthetic_image_dir = synthetic_image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Decide whether idx refers to an original image or a synthetic image
        # And map idx to index in dataframe
        index_in_dataframe, is_synthetic_image = self.sampler[idx]

        file_name = self.dataframe.iloc[index_in_dataframe]['file_name']
        caption = self.dataframe.iloc[index_in_dataframe]['caption']

        if is_synthetic_image:
            image = Image.open(os.path.join(self.synthetic_image_dir, file_name)).convert('RGB')
            image_tensor: torch.Tensor = self.transform(image)
        else:
            image = Image.open(os.path.join(self.original_image_dir, file_name)).convert('RGB')
            image_tensor: torch.Tensor = self.transform(image)

        # 从caption得到tokens
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        # append start and end token
        caption_tensor = torch.Tensor(
            [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        )

        return image_tensor, caption_tensor
