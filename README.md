# 2023 Car Model Classification 
<p align="justify">
This project focuses on advancing car model classification, departing from prior works primarily based on older datasets such as the [Stanford car dataset](https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf). Tailored for 2023 models, the car model classifier is trained on an extensive dataset comprising over 200,000 car images sourced from the web, spanning model years 2011 to 2024. The dataset encompasses more than 19,000 car model images, each accompanied by detailed specifications.

Implemented on the PyTorch framework, two deep learning models utilize EfficientNetB1, ResNet50, and ResNet101 architectures. The first model classifies interior-exterior car images, contributing to the creation of the primary dataset. The second model achieves a 97.3% training and 73.4% testing accuracy in classifying 2023 car models.

To facilitate user interaction, a Flask-based web app has been developed for deploying the classifier. This web application allows users to upload images of 2023 model-year cars and receive detailed information about the model and its specifications.


<p align="center">
  <img src="/images/introduction.png" alt="Image Alt text">
</p>

</p>

Table of contents
=================

- [File Description](#file-description)
- [Data Scrapping](#data-scrapping)
- [Model Training](#model-training)
- [Web App](#web-app)
- [Future Work](#future-work)
- [Classification Report](#classification-report)

# File Description


The primary files for this project can be found within the "src" folder:
| Folder | Description |
| --- | --- |
| web_scrapping |  Includes the scripts used for scrapping the data images/specification, cleaning the data frames, and postprocessing the data |
| model_classifier | Includes the scripts used for training the main car classifier, utilizing the ResNet50, ResNet101 architectures |
| interior_exterior_classifier |  Includes the scripts used for training the interior exterior classifier, utilizing the EfficientNet_b1 architecture |
| interior_exterior_classifier |  Includes the files used for training the interior exterior classifier, utilizing the EfficientNet_b1 architecture |
| web_app |  Includes the scripts used for creating the Flask car classifier web app |


# Data Scrapping 

<p align="justify"> 
This project involves extensive data scraping activities. Specifically, 200,000 car image URLs, accompanied by their corresponding specifications spanning over 19,000 models, are systematically scraped from the web. The gathered data undergoes a thorough cleaning process to format the specifications appropriately. Subsequently, the car images are extracted and saved using the obtained URLs, undergoing post-processing steps to ensure they are prepared for the subsequent training phases. The car data includes both exterior and interior images:
 </p>


<p align="center">
  <img src="/images/data.PNG" alt="Image Alt text">
</p>

# Model Training


To distinguish between exterior and interior images and prepare the exterior ones for the primary model training, a specialized model is developed using the EfficientNetB1 architecture. This compact model is designed to classify car images into either "Exterior" or "Interior" categories based on their visual characteristics.



<p align="center">
  <img src="/images/EfficientNet.PNG" alt="Image Alt text">
</p>



<p align="justify"> 
 
After preparing the main training data, the primary classifier undergoes training using exterior image data to discern various car models. This involves employing two architectures, ResNet50 and ResNet101. Transfer learning is applied to extract features from the models, and the fully-connected sections are retrained using exterior image data. To enhance the networks' specialization, fine-tuning is performed using the 2023 model year car models. 


<p align="center">
  <img src="/images/resnet.png" alt="Image Alt text">
</p>




The resulting model is proficient in classifying 175 categories of 2023 car models. The training outcomes are detailed below, revealing the model's convergence and its attainment of a **97.3% training and a 73.4% testing accuracy**:

<p align="center">
  <img src="/images/results.png" alt="Image Alt text">
</p>

# Web Application



To streamline the model deployment process, a web application has been created using the Flask framework in Python. This application is designed to accept an image of a 2023 car model and provide information such as the model itself, the Manufacturer's Suggested Retail Price (MSRP), and pertinent specifications. The initial version, v1.0.0, is currently operational on a server and can be accessed via the following link.

<p align="center">
  <img src="/images/web-app.png" alt="Image Alt text">
</p>


# Future Work



The upcoming phases for this project include the following considerations:

1. Enhance the testing accuracy of the model by exploring additional architectures and incorporating more fully-connected layers.

2. Expand the training dataset to encompass car images from model years spanning 2020 to 2024.

3. Revise and upgrade the design of the web application, extending its capabilities to display a broader range of exterior and interior images associated with the classified car model.

 </p>

# Classification Report


Below is the full classification report generated from the output of the best ResNet101 model.




