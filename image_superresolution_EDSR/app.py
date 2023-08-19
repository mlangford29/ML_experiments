from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import torch
import argparse
from EDSR_PyTorch.src.model import EDSR

app = Flask(__name__)

# Constants and configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'edsr_model/EDSR_x3.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize and load the EDSR model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR(args=args)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_super_resolution(image_path):
    with Image.open(image_path).convert('RGB') as img:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_image = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_image).squeeze().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (output * 255).astype('uint8')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('file')

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Apply super resolution using EDSR
            enhanced_image = apply_super_resolution(filepath)
            
            # Save the enhanced image
            im = Image.fromarray(enhanced_image)
            enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + filename)
            im.save(enhanced_path)
    
    return jsonify({'message': 'Files uploaded and enhanced successfully!'})

@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
