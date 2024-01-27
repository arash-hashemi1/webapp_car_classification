# 2023 Car Model Classification (scrapping + training + web-app)

This project focuses on advancing car model classification, departing from prior works primarily based on older datasets such as the [Stanford car dataset](http://vision.stanford.edu/pdf/3drr13.pdf). Tailored for 2023 models, the car model classifier is trained on an extensive dataset comprising over 200,000 car images sourced from the web, spanning model years 2011 to 2024. The dataset encompasses more than 19,000 car model images, each accompanied by detailed specifications.

Implemented on the PyTorch framework, two deep learning models utilize EfficientNet B1 and ResNet101 architectures. The first model classifies interior-exterior car images, contributing to the creation of the primary dataset. The second model achieves a 97.3% training and 73.4% testing accuracy in classifying 2023 car models.

To facilitate user interaction, a Flask-based web app has been developed for deploying the classifier. This web application allows users to upload images of 2023 model-year cars and receive detailed information about the model and its specifications.



![introduction](https://github.com/arash-hashemi1/webapp_car_classification/assets/48169508/96948d90-399f-41dd-9050-42d8e167e195)



Table of contents
=================

<!--ts-->
  * [File Description](#files)  
  * [Data Scrapping](#data)  
  * [Model Training](#model)
  * [Error Analysis](#error)
  * [Web App](#webapp)
  * [Future Work](#future)
  * [Confidence Report](#confidence)
<!--te--> 

File Description
================

The primary files for this project can be found within the "src" folder:
| Folder | Description |
| --- | --- |
| web_scrapping |  Includes the scripts used for scrapping the data images/specification, cleaning the data frames, and postprocessing the data |
| model_classifier | Includes the scripts used for training the main car classifier, utilizing the ResNet50, ResNet101 architectures |
| interior_exterior_classifier |  Includes the scripts used for training the interior exterior classifier, utilizing the EfficientNet_B1 architecture |
| interior_exterior_classifier |  Includes the files used for training the interior exterior classifier, utilizing the EfficientNet_B1 architecture |
| web_app |  Includes the scripts used for creating the Flask car classifier web app |

Data Scrapping
================

This project involves extensive data scraping activities. Specifically, 200,000 car image URLs, accompanied by their corresponding specifications spanning over 19,000 models, are systematically scraped from the web. The gathered data undergoes a thorough cleaning process to format the specifications appropriately. Subsequently, the car images are extracted and saved using the obtained URLs, undergoing post-processing steps to ensure they are prepared for the subsequent training phases.
