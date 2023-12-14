import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from dataset_with_synthetic_images import CSVDatasetWithSyntheticImages
from encoder_decoder import Encoder, Decoder
from vocabulary import get_vocabulary_from_captions


def collate_fn(coco_data: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    """
    # Sort Descending by caption length
    coco_data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*coco_data)

    # turn images to a 4D tensor with specified batch_size
    stacked_images = torch.stack(images, 0)

    # do the same thing for caption. -> 2D tensor with equal length of max lengths in batch, padded with 0
    caption_length_list = [len(cap) for cap in captions]
    seq_length = max(caption_length_list)
    # Truncation
    if max(caption_length_list) > 100:
        seq_length = 100

    stacked_captions = torch.LongTensor(np.zeros((len(captions), seq_length)))
    for i, cap in enumerate(captions):
        length = caption_length_list[i]
        stacked_captions[i, :length] = cap[:length]

    return stacked_images, stacked_captions, caption_length_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file-names-and-captions-csv-path', type=str, required=True)
    parser.add_argument('--real-image-dir', type=str, required=True)
    parser.add_argument('--input-encoder-weights-path', type=str, required=True)
    parser.add_argument('--input-decoder-weights-path', type=str, required=True)
    parser.add_argument('--encoder-weights-path', type=str, required=True)
    parser.add_argument('--decoder-weights-path', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    args = parser.parse_args()

    train_file_names_and_captions_csv_path = args.train_file_names_and_captions_csv_path
    train_file_names_and_captions_dataframe = pd.read_csv(train_file_names_and_captions_csv_path)

    real_image_dir = args.real_image_dir
    input_encoder_weights_path = args.input_encoder_weights_path
    input_decoder_weights_path = args.input_decoder_weights_path
    encoder_weights_path = args.encoder_weights_path
    decoder_weights_path = args.decoder_weights_path
    num_epochs = args.num_epochs

    vocabulary = get_vocabulary_from_captions(
        train_file_names_and_captions_dataframe['caption'],
        4
    )

    dataset = CSVDatasetWithSyntheticImages(
        vocabulary,
        train_file_names_and_captions_dataframe,
        real_image_dir,
        1.0,
        real_image_dir,
        0.0,
        # Image preprocessing, normalization for the pretrained resnet
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Declare the encoder decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    encoder_ = Encoder(embed_size=512).to(device)
    decoder_ = Decoder(embed_size=512, hidden_size=512, vocab_size=len(vocabulary), num_layers=3, stateful=False).to(device)

    encoder_.load_state_dict(torch.load(input_encoder_weights_path))
    decoder_.load_state_dict(torch.load(input_decoder_weights_path))

    encoder_.train()
    decoder_.train()

    encoder_on_device = encoder_.to(device)
    decoder_on_device = decoder_.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # For encoder only train the last fc layer
    params = list(encoder_on_device.resnet.fc.parameters()) + list(decoder_on_device.parameters())

    optimizer = torch.optim.Adam(params, lr=0.001)

    # Train the model
    for epoch in range(80):
        # Create a tqdm progress bar
        pbar = tqdm(data_loader)
        for i, (stacked_images_, stacked_captions_, caption_length_list_) in enumerate(pbar):

            # Set mini-batch dataset
            stacked_images_on_device = stacked_images_.to(device)
            stacked_captions_on_device = stacked_captions_.to(device)
            targets = pack_padded_sequence(stacked_captions_on_device, caption_length_list_, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder_on_device(stacked_images_on_device)
            outputs = decoder_on_device(features, stacked_captions_on_device, caption_length_list_)
            loss = criterion(outputs, targets)

            decoder_on_device.zero_grad()
            encoder_on_device.zero_grad()

            loss.backward(retain_graph=True)

            optimizer.step()

            # Update the description (this will display the loss)
            pbar.set_description(
                f"Loss: {loss.item():.2f}, Perplexity: {np.exp(loss.item()):5.4f}"
            )

        if epoch % 10 == 0:
            torch.save(decoder_on_device.state_dict(), decoder_weights_path)
            torch.save(encoder_on_device.state_dict(), encoder_weights_path)

    torch.save(decoder_on_device.state_dict(), decoder_weights_path)
    torch.save(encoder_on_device.state_dict(), encoder_weights_path)
