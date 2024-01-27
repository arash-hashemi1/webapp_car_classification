# 2023 Car Model Classification (scrapping + training + web-app)

This project focuses on advancing car model classification, departing from prior works primarily based on older datasets such as the [Stanford car dataset](https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf). Tailored for 2023 models, the car model classifier is trained on an extensive dataset comprising over 200,000 car images sourced from the web, spanning model years 2011 to 2024. The dataset encompasses more than 19,000 car model images, each accompanied by detailed specifications.

Implemented on the PyTorch framework, two deep learning models utilize EfficientNetB1, ResNet50, and ResNet101 architectures. The first model classifies interior-exterior car images, contributing to the creation of the primary dataset. The second model achieves a 97.3% training and 73.4% testing accuracy in classifying 2023 car models.

To facilitate user interaction, a Flask-based web app has been developed for deploying the classifier. This web application allows users to upload images of 2023 model-year cars and receive detailed information about the model and its specifications.



<p>![Image Alt text](/images/introduction.png)</p>



Table of contents
=================

<!--ts-->
  * [File Description](#files)  
  * [Data Scrapping](#data)  
  * [Model Training](#model)
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
| interior_exterior_classifier |  Includes the scripts used for training the interior exterior classifier, utilizing the EfficientNet_b1 architecture |
| interior_exterior_classifier |  Includes the files used for training the interior exterior classifier, utilizing the EfficientNet_b1 architecture |
| web_app |  Includes the scripts used for creating the Flask car classifier web app |

Data Scrapping
================
<p align="justify"> 
This project involves extensive data scraping activities. Specifically, 200,000 car image URLs, accompanied by their corresponding specifications spanning over 19,000 models, are systematically scraped from the web. The gathered data undergoes a thorough cleaning process to format the specifications appropriately. Subsequently, the car images are extracted and saved using the obtained URLs, undergoing post-processing steps to ensure they are prepared for the subsequent training phases. The car data includes both exterior and interior images:
 </p>



![Image Alt text](/images/data.PNG)

Model Training
================

To distinguish between exterior and interior images and prepare the exterior ones for the primary model training, a specialized model is developed using the EfficientNetB1 architecture. This compact model is designed to classify car images into either "Exterior" or "Interior" categories based on their visual characteristics.



![EfficientNet](https://github.com/arash-hashemi1/webapp_car_classification/assets/48169508/29d2eba2-cf8d-4ce5-b964-f49855acbd87)


<p align="justify"> 
 
After preparing the main training data, the primary classifier undergoes training using exterior image data to discern various car models. This involves employing two architectures, ResNet50 and ResNet101. Transfer learning is applied to extract features from the models, and the fully-connected sections are retrained using exterior image data. To enhance the networks' specialization, fine-tuning is performed using the 2023 model year car models. 



![resnet](https://github.com/arash-hashemi1/webapp_car_classification/assets/48169508/23b8d993-c10e-412a-bc2b-10e82eef6ff3)



The resulting model is proficient in classifying 175 categories of 2023 car models. The training outcomes are detailed below, revealing the model's convergence and its attainment of a **97.3% training and a 73.4% testing accuracy**:

<p align="center">
  <img width="460" height="300" src=![results](https://github.com/arash-hashemi1/webapp_car_classification/assets/48169508/f0fb2378-66eb-4186-9095-a81d1058104f)>
</p>
  



 </p>




