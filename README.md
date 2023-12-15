# The Impact of Synthetic Data on Image Captioning Models

All code for the paper "The Impact of Synthetic Data on Image Captioning Models".

We ran the code on Google Colab (although you can adapt the following notebooks to run locally). To replicate the results:

- Save the [Flickr8k](https://drive.google.com/file/d/1m480o1akbeJ6HYN9U9BBZUrrLw7TKKVQ/view?usp=sharing), [Pascal_VOC_2008](https://drive.google.com/file/d/1ZhUMgceAxpX-lb-0AcVhqXpVuQCRJbJ9/view?usp=sharing), [Flickr30k](https://drive.google.com/file/d/1JUrk-cHHXRzaVaFMX_hDN_Fg-c4Ajf6x/view?usp=sharing), and [COCO2014](https://drive.google.com/file/d/1rjLvp3YwSyeVp_9_GGGUWFRlbvhp3fHX/view?usp=sharing) datasets to the top level of your Google Drive. Each Dataset is a `.zip` file containing the following files and folders:
    - <name of the dataset>
        - images/
        - train_file_names_and_captions.csv (only in the Flickr8k dataset)
        - test_file_names_and_captions.csv
        - file_names_and_captions.csv
- Run [generate_with_stable_diffusion.ipynb](https://colab.research.google.com/drive/1hx8qLCvTAgtRfii48IawUjmYOwgwTtX9?usp=sharing) to generate synthetic images for the Flickr8k dataset. This saves `Flickr8k_generated_images.zip` to the top level of your Google Drive, which contains:
    - generated_images/
- Run [train_show_and_tell_with_synthetic_images.ipynb](https://colab.research.google.com/drive/105UpEKbdQ3WXXuKbZ4RDVauouIqiN-mY?usp=sharing). This saves all weights to the top level of your Google Drive.
- Run [fine_tune_show_and_tell.ipynb](https://colab.research.google.com/drive/1-p5GIYf6Z8XGHefAtGNCL2DUT7ohkK4a?usp=sharing). This saves all weights to the top level of your Google Drive.
- Run [test_show_and_tell.ipynb](https://colab.research.google.com/drive/1PZmLDJV_vi14AaJJkinxvwjIpPMc6glS?usp=sharing) and [test_fine_tuned_show_and_tell.ipynb](https://colab.research.google.com/drive/1ZQ2bOqaCEBRUpwJTEQy3IpG9f0UDQ7lP?usp=sharing). These saves all CSV files to the top level of your Google Drive.
- Run [print_tables.ipynb](https://colab.research.google.com/drive/1BlA4D5uXhm_L8_lNoDydX80nGIaDPWVS?usp=sharing). This prints LaTeX tables in our paper.
