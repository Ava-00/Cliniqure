# CliniQure: AI-Driven Diagnostic and Treatment Recommendations for Radiology

## Overview

**CliniQure** is an advanced, AI-driven diagnostic system designed to automate disease detection, improve interpretability, and streamline clinical workflows in radiology. It leverages cutting-edge machine learning techniques, including multi-task learning for object detection and disease classification, to deliver accurate diagnostic insights. This project aims to enhance diagnostic precision, especially in resource-limited settings, where access to expert interpretation of medical images is often constrained.

### Key Features
- **Object Detection**: Utilizes the Faster R-CNN architecture with a ResNet-50 backbone to accurately localize abnormalities in medical images.
- **Disease Classification**: Employs the SE-ResNet50 model for multi-label classification of diseases based on medical image analysis.
- **Diagnostic Report Generation**: Generates initial diagnostic reports using the R2GenCMN model and fine-tunes them with a large language model (LLM) to improve context and coherence in clinical reporting.
- **Seamless Integration**: Combines visual analysis with detailed textual output to provide a comprehensive diagnostic report.

## Technologies Used
- **Programming Languages**: Python
- **Libraries/Frameworks**: PyTorch, OpenCV, NumPy
- **Machine Learning Models**: Faster R-CNN (ResNet-50), SE-ResNet50, R2GenCMN, Fine-tuned Large Language Model (LLM) - OPT1.3B
- **Dataset**: VinDr-CXR dataset
  
