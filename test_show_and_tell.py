import argparse
import logging
import os.path

import pandas as pd
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from bleu_score import bleu_score
from encoder_decoder import Encoder, Decoder
from vocabulary import get_vocabulary_from_captions


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file-names-and-captions-csv-path', type=str, required=True)
    parser.add_argument('--test-file-names-and-captions-csv-path', type=str, required=True)
    parser.add_argument('--test-image-dir', type=str, required=True)
    parser.add_argument('--encoder-weights-path', type=str, required=True)
    parser.add_argument('--decoder-weights-path', type=str, required=True)
    parser.add_argument('--bleu-scores-csv-path', type=str, required=True)
    args = parser.parse_args()

    train_file_names_and_captions_csv_path = args.train_file_names_and_captions_csv_path
    train_file_names_and_captions_dataframe = pd.read_csv(train_file_names_and_captions_csv_path)

    test_file_names_and_captions_csv_path = args.test_file_names_and_captions_csv_path
    test_file_names_and_captions_dataframe = pd.read_csv(test_file_names_and_captions_csv_path)

    test_image_dir = args.test_image_dir
    encoder_weights_path = args.encoder_weights_path
    decoder_weights_path = args.decoder_weights_path

    bleu_scores_csv_path = args.bleu_scores_csv_path

    vocabulary = get_vocabulary_from_captions(
        train_file_names_and_captions_dataframe['caption'],
        4
    )

    # Declare the encoder decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    encoder_ = Encoder(embed_size=512)
    decoder_ = Decoder(embed_size=512, hidden_size=512, vocab_size=len(vocabulary), num_layers=3, stateful=False)

    encoder_.load_state_dict(torch.load(encoder_weights_path))
    decoder_.load_state_dict(torch.load(decoder_weights_path))

    encoder_.eval()
    decoder_.eval()

    encoder_on_device = encoder_.to(device)
    decoder_on_device = decoder_.to(device)

    # Generate captions for validation images.

    # In generation phase, we need should random crop, just resize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    test_file_name_list: list[str] = []
    generated_score_list: list[float] = []
    theoretical_score_list: list[float] = []

    for test_file_name, dataframe_with_test_file_name in train_file_names_and_captions_dataframe.groupby('file_name'):
        assert isinstance(test_file_name, str)

        image_path: str = os.path.join(
            test_image_dir,
            test_file_name
        )

        image: Image = Image.open(image_path).convert('RGB').resize((224, 224), Image.LANCZOS)

        # Prepare an image (NCHW)
        image_tensor: torch.Tensor = transform(image).unsqueeze(0)
        image_tensor_on_device = image_tensor.to(device)

        # Generate a caption from the image
        feature_tensor_on_device = encoder_on_device(image_tensor_on_device)
        sampled_word_indices_tensor_on_device = decoder_on_device.sample(feature_tensor_on_device)
        sampled_word_indices_ndarray = sampled_word_indices_tensor_on_device[0].cpu().numpy()

        raw_sampled_caption_tokens_list: list[str] = []

        for sampled_word_index in sampled_word_indices_ndarray:
            word = vocabulary.idx2word[sampled_word_index]
            raw_sampled_caption_tokens_list.append(word)
            if word == '<<end>>':
                break

        sampled_caption_tokens_list = raw_sampled_caption_tokens_list[1:-1]  # Skip <<start>> and <<end>>

        test_caption_list = list(
            dataframe_with_test_file_name['caption']
        )

        generated_score, theoretical_score = bleu_score(
            sampled_caption_tokens_list,
            test_caption_list
        )

        test_file_name_list.append(test_file_name)
        generated_score_list.append(generated_score)
        theoretical_score_list.append(theoretical_score)

    df = pd.DataFrame({
        'file_name': test_file_name_list,
        'generated_score': generated_score_list,
        'theoretical_score': theoretical_score_list
    })

    df.to_csv(bleu_scores_csv_path, index=False)
