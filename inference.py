import torch
sys.path.append(r"C:\Users\avant\R2GenCMN")
from models import BaseCMNModel
from modules.tokenizers import Tokenizer
import os
import pydicom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
from main import parse_agrs

#Process Data
#Load files
dicom_dir = r"C:\Users\avant\OneDrive\Desktop\Dicoms"
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dicom')]
print("DICOM files:", dicom_files)

#See data for one dicom file
dicom_path = dicom_files[0]
dicom_data = pydicom.dcmread(dicom_path)
print(dicom_data)

#Extract pixel data 
pixel_array = dicom_data.pixel_array
print("Shape of pixel array:", pixel_array.shape)

#visualize 
plt.imshow(pixel_array, cmap='gray')
plt.title('DICOM Image')
plt.axis('on')
plt.show()

# Function to load and preprocess a single DICOM image
def preprocess_dicom(dicom_file):
        # Read DICOM image
        dicom_data = pydicom.dcmread(dicom_file)
        if not hasattr(dicom_data, "PixelData"):
            print(f"Skipping {dicom_file}: No pixel data.")
            return None
        image = dicom_data.pixel_array.astype(np.uint8)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1) 
        
        # Normalize and transform the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        return image_tensor

dicom_file = r"C:\Users\avant\OneDrive\Desktop\Dicoms\00a7e1e81b2122e5a0f298155573e549.dicom"

# Preprocess the DICOM image
image_tensor = preprocess_dicom(dicom_file)

# Display the processed data
if image_tensor is not None:
    print("Preprocessed Image Tensor Shape:", image_tensor.shape)
    print("Preprocessed Image Tensor (Sample Data):", image_tensor[:, :5, :5])  # Show a small patch of the image tensor
else:
    print("Failed to preprocess the file.")

#Batch processing

dicom_folder = r"C:\Users\avant\OneDrive\Desktop\Dicoms"
image_tensors = []

# Process all `.dicom` files in the folder
for file_name in os.listdir(dicom_folder):
    if file_name.endswith('.dicom'):
        file_path = os.path.join(dicom_folder, file_name)
        image_tensor = preprocess_dicom(file_path)
        if file_name.lower().endswith('.dicom'):
            print(f"Processing file: {file_path}")
            image_tensor = preprocess_dicom(file_path)
            if image_tensor is not None:
                image_tensors.append(image_tensor)
                print(f"Processed {file_name}")
        else:
            print(f"Skipping file due to error: {file_path}")

# Combine all tensors into a batch
if image_tensors:
    image_batch = torch.stack(image_tensors)
    print("Batch of preprocessed images shape:", image_batch.shape)
else:
    print("No valid DICOM files found or processed.")

# Parse the arguments as per the repository setup (this should be done correctly according to your environment)
args = parse_agrs()
args.ann_path = "C:\\Users\\avant\\R2GenCMN\\annotation.json"
tokenizer = Tokenizer(args)

# Load the model
model_path = r"C:\Users\avant\R2GenCMN\checkpoints\model_mimic_cxr.pth" # Path to the model checkpoint
model = BaseCMNModel(args, tokenizer = tokenizer)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()} 
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming image_tensors is a list of individual image tensors
with torch.no_grad():
    
    for image_tensor in image_tensors:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)      
        # If the image is grayscale (1 channel), convert to RGB (3 channels)
            if image_tensor.size(1) == 1:  # Single channel (grayscale)
               image_tensor = image_tensor.repeat(1, 3, 1, 1)  # Convert to [batch_size, 3, height, width]
               print(f"Converted to RGB: {image_tensor.shape}")
        # Simulate multiple views: Repeat the image for each view (2 views by default)
        # For each view, we want a separate tensor
               view_1 = image_tensor
               view_2 = image_tensor
               print(f"View 1 shape: {view_1.shape}, View 2 shape: {view_2.shape}")
        # Stack them into a batch of 4D tensors [batch_size, channels, height, width]
               image_tensor = torch.stack([view_1, view_2], dim=1)
               image_tensor = image_tensor.squeeze()  # Shape: [2, 3, H, W]
               print(f"Processed image tensor shape: {image_tensor.shape}")
        
    outputs = []
#     sequence_length = 60  # Length of the sequence you want to generate

    for image_tensor in image_tensors:  # Loop through each image tensor
        image_tensor = image_tensor.cpu()  # Ensure tensor is on the CPU
        output, output_probs = model.forward_mimic_cxr(image_tensor.unsqueeze(0), mode='sample')
        print(f"Output shape: {output.shape}")
        outputs.append(output)


#     logits = output.squeeze(0)  # Assuming output shape is [1, seq_len, vocab_size]
    
#     # Go through each step in the sequence
#     sampled_ids = []
#     for seq_idx in range(sequence_length):  # For each token in the sequence
#         prob_dist = logits[seq_idx]  # Get the probability distribution for the current token
#         sampled_token = torch.argmax(prob_dist).item()  # Get the index of the highest probability token (deterministic)
#         sampled_ids.append(sampled_token)
    
#     # Decode the sampled tokens into text
#     decoded_text = tokenizer.decode(sampled_ids)
#     print(f"Generated Text: {decoded_text}")  # Display the decoded text
#     outputs.append(decoded_text)

# Display the outputs
for output_idx, output in enumerate(outputs):  # Enumerate to keep track of which image
    print(f"Outputs for image {output_idx + 1}:")  # Optional: Print which image is being processed
    # Loop through each sequence in the current output tensor
    for i in range(output.size(0)):  
        decoded_output = tokenizer.decode(output[i].tolist())  # Decode the output
        print(f"Output: {decoded_output}")  # Print the decoded output

import json

# Initialize a dictionary to store standardized outputs
standardized_outputs = []

for output_idx, output in enumerate(outputs):  
    decoded_output = []
    for i in range(output.size(0)):  
        # Decode the generated output
        decoded_output.append(tokenizer.decode(output[i].tolist()))

    # Add the decoded report for this image
    standardized_outputs.append({
        "image_index": output_idx + 1,  # Image index (1-based)
        "generated_report": " ".join(decoded_output)  # Combine all sequences for one output
    })

# Save the outputs to a JSON file
output_file = "standardized_output.json"
with open(output_file, "w") as f:
    json.dump(standardized_outputs, f, indent=4)

print(f"Standardized output saved to {output_file}")
#open JSON file 
with open("standardized_output.json") as f:
    data = json.load(f)
    print(data)

#Key word extraction 
import os

# Install spacy
os.system("pip install spacy")
import spacy
from collections import Counter
from itertools import chain

# Load English language model
os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def extract_keywords(report):
    doc = nlp(report)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return keywords

# Extract keywords for each report
keywords_per_report = []
for entry in data:
    keywords = extract_keywords(entry['generated_report'])
    keywords_per_report.append({
        'image_index': entry['image_index'],
        'keywords': list(set(keywords))  # Deduplicate keywords
    })

#Generate Summaries
from transformers import pipeline

# Initialize summarization model
summarizer = pipeline("summarization")

# Function to generate summary from extracted keywords
import json

def generate_summary(keywords):
    keyword_text = " ".join(keywords)
    summary = summarizer(keyword_text, max_length=25, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Initialize a list to store all summaries
summaries = []

# Generate summaries for each report
for report in keywords_per_report:
    summary = generate_summary(report['keywords'])
    summaries.append({
        "image_index": report['image_index'],
        "summary": summary
    })
    print(f"Image {report['image_index']} Summary: {summary}")

# Specify the path to save the file
output_file = r"C:\Users\avant\Documents\summaries.json"

# Save the summaries to a JSON file
try:
    with open(output_file, "w") as json_file:
        json.dump(summaries, json_file, indent=4)
    print(f"Summaries saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
summaries

# #Hugging Face Download 
# from datasets import load_dataset

# MT_data = load_dataset("Magneto/modified-medical-dialogue-soap-summary")
# #Convert to CSV
# for split, dataset in MT_data.items():
#     dataset.to_csv(f"MT_data_{split}.csv")
# #Obtain CSV file paths
# csv_files = [f"MT_data_{split}.csv" for split in MT_data.keys()]
# print("CSV files:", csv_files)
# import pandas as pd
# open_file = pd.read_csv("MT_data_train.csv")
# print(open_file.head())