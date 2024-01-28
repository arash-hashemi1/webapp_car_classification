import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models, datasets
from torchvision.transforms import v2
from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from flask import Flask, render_template, request
from PIL import Image
import pandas as pd
from transform import transform_image
from resnet_classifier import ResNet101, ResNet50
from class_prediction import get_predicted_class


# setting device

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Specify specs file path
spec_path = 'data/specs-cleaned.csv'
spec_df = pd.read_csv(spec_path)
spec_df.set_index('Unnamed: 0', inplace=True)
spec_columns = spec_df.columns

# Load pre-trained model
hidden_1 = 1024
hidden_2 = 512
model_path = 'models/model_checkpoint_99.pth'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location)
model = ResNet50(hidden_1=1024, hidden_2=512, num_target_classes=175).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


from flask import Flask, render_template, request
import pickle


# Get class-to-index mapping
dict_path = 'models/class_to_idx_2023.pkl'

with open(dict_path, 'rb') as f:
  class_to_idx = pickle.load(f)
  idx_to_class = {v: k for k, v in class_to_idx.items()}

train_transforms = transform_image()
# Function to get predicted class
def get_top_predictions(model, image_tensor, class_to_idx, transform, spec_df, top_k=5):
    model.eval()

    transformed_image = transform(image_tensor).unsqueeze(0)

    with torch.no_grad():
        logits = model(transformed_image.to(device))
        probabilities = torch.softmax(logits, dim=1)

        top_probs, top_classes = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy().flatten()
        top_classes = top_classes.cpu().numpy().flatten()

        idx_to_class = {idx: car_class for car_class, idx in class_to_idx.items()}

        top_predictions = [idx_to_class[idx] for idx in top_classes]
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = idx_to_class[predicted_idx]
        df_slice = ' '.join([predicted_class, ''])

        spec_select = spec_df.loc[df_slice]
        spec_select = spec_select[['Make', 'Model', 'Year', 'MSRP', 'Gas Mileage', 'Engine', 'EPA Class']]
        df = pd.DataFrame(spec_select)

        return list(zip(top_predictions, top_probs)), df



# Initialize Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        image = Image.open(file).convert('RGB')

        # Get the top predictions
        top_predictions, df = get_top_predictions(model, image, class_to_idx, train_transforms, spec_df)

        best_match = f'Best Match: {top_predictions[0][0]} with {top_predictions[0][1]:.2%} confidence.'
        other_matches = [{'model': prediction[0], 'confidence': f'{prediction[1]:.2%}'} for prediction in top_predictions[1:]]
        static_folder = 'src/web_app/static'
        os.makedirs(static_folder, exist_ok=True)
        image_path = os.path.join(static_folder, 'uploaded_image.jpg')
        image.save(image_path)

        return render_template('index.html', message= best_match,
                               image_path='uploaded_image.jpg', df_table=df.to_html(classes='table table-striped'),
                               other_matches=other_matches)

# Run the app
if __name__ == '__main__':
    app.run()