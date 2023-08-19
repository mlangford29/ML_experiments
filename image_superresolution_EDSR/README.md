# EDSR Super-Resolution Flask Application

This Flask application serves as a web interface for users to upload images and then process them using a pre-trained EDSR (Enhanced Deep Super-Resolution) model to enhance their resolutions.

## Features

- Web UI for easy image upload
- Supports multiple image uploads at once
- Uses a pre-trained EDSR model for super-resolution

## Usage

1. Open the web application in a browser.
2. Use the "Choose multiple images" button to select one or more images you want to enhance.
3. Click on "Upload and Process". The app will process the images and enhance their resolution using the EDSR model.
4. Enhanced images will be saved in the `uploads` directory with a prefix `enhanced_`.

## Note

- Supported image formats: PNG, JPG
- Ensure that the Flask server has necessary read/write permissions for the `uploads` directory.
  
## Acknowledgements

- EDSR model is from the [EDSR PyTorch implementation](https://github.com/sanghyun-son/EDSR-PyTorch).