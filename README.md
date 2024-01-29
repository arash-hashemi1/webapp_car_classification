# 2023 Car Model Classification 

This project focuses on advancing car model classification, departing from prior works primarily based on older vehicle datasets such as the [Stanford car dataset](https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf). Extensive data scraping was employed in this project to collect over 200,000 car images meticulously sourced from the web. The dataset spans model years from 2011 to 2024, encompassing over 19,000 images for individual car models, each accompanied by detailed specifications. The car model classifier is meticulously crafted, focusing its training and customization exclusively on 2023 car models. This targeted approach ensures a heightened level of precision and accuracy in recognizing the distinctive features of vehicles from this specific model year.

Utilizing the PyTorch framework, this project employs two deep learning models. The initial model employs the EfficientNetB1 architecture to classify interior-exterior car images, thereby contributing to the formation of the primary dataset. The second model, responsible for the main car model classification, is trained on ResNet50 and ResNet101 architectures.

To facilitate user interaction, a Flask-based web app has been developed for deploying the classifier. This web application allows users to upload images of 2023 model-year cars and receive detailed information about the model and its specifications ([Web App Link](http://arashhsm.pythonanywhere.com/)) 
<br>
<br>
<br>
<br>

<p align="center">
  <img src="/images/introduction.png" alt="Image Alt text">
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

Below is the complete classification report for the best ResNet101 model
                           
|                                 |  precision  |  recall  |  f1-score  |  support |
|---------------------------------|-------------|----------|------------|--------- |
|2023 Acura Integra               |  0.4        |  0.33    |  0.36      |  6 |
|2023 Acura MDX                   |  0.25       |  0.25    |  0.25      |  8 |
|2023 Acura RDX                   |  0.5        |  0.29    |  0.36      |  7 |
|2023 Acura TLX                   |  0.42       |  0.62    |  0.5       |  8 |
|2023 Alfa Romeo Giulia           |  0.71       |  0.83    |  0.77      |  6 |
|2023 Alfa Romeo Stelvio          |  0.8        |  0.8     |  0.8       |  5 |
|2023 Audi A3                     |  1.0        |  0.71    |  0.83      |  7 |
|2023 Audi A4                     |  0.86       |  0.86    |  0.86      |  7 |
|2023 Audi A5                     |  0.6        |  0.38    |  0.46      |  8 |
|2023 Audi A6                     |  0.5        |  0.5     |  0.5       |  8 |
|2023 Audi A7                     |  0.42       |  0.56    |  0.48      |  9 |
|2023 Audi A8                     |  0.54       |  0.7     |  0.61      |  10 |
|2023 Audi E-Tron GT              |  0.67       |  0.8     |  0.73      |  5 |
|2023 Audi Q3                     |  0.62       |  0.83    |  0.71      |  6 |
|2023 Audi Q4 E-Tron              |  0.38       |  0.71    |  0.5       |  7 |
|2023 Audi Q5                     |  0.5        |  0.5     |  0.5       |  8 |
|2023 Audi Q7                     |  0.86       |  0.86    |  0.86      |  7 |
|2023 Audi R8                     |  0.3        |  0.33    |  0.32      |  9 |
|2023 Audi TT                     |  1.0        |  0.71    |  0.83      |  7 |
|2023 BMW 2-Series                |  0.0        |  0.0     |  0.0       |  2 |
|2023 BMW 3-Series                |  0.73       |  1.0     |  0.84      |  8 |
|2023 BMW 4-Series                |  0.43       |  0.75    |  0.55      |  8 |
|2023 BMW 5-Series                |  0.67       |  0.5     |  0.57      |  4 |
|2023 BMW 7-Series                |  0.89       |  0.73    |  0.8       |  11 |
|2023 BMW 8-Series                |  0.8        |  1.0     |  0.89      |  4 |
|2023 BMW X1                      |  0.71       |  1.0     |  0.83      |  10 |
|2023 BMW X3                      |  0.7        |  0.88    |  0.78      |  8 |
|2023 BMW X4                      |  0.86       |  0.86    |  0.86      |  7 |
|2023 BMW X5                      |  0.67       |  0.67    |  0.67      |  6 |
|2023 BMW X6                      |  0.6        |  0.75    |  0.67      |  8 |
|2023 BMW X7                      |  0.57       |  0.44    |  0.5       |  9 |
|2023 BMW Z4                      |  1.0        |  0.9     |  0.95      |  10 |
|2023 BMW i4                      |  1.0        |  0.75    |  0.86      |  4 |
|2023 BMW iX                      |  0.8        |  0.5     |  0.62      |  8 |
|2023 Buick Enclave               |  0.78       |  0.88    |  0.82      |  8 |
|2023 Buick Encore GX             |  0.5        |  0.83    |  0.62      |  6 |
|2023 Buick Envision              |  0.57       |  0.57    |  0.57      |  7 |
|2023 Cadillac CT4                |  0.6        |  0.5     |  0.55      |  6 |
|2023 Cadillac CT5                |  0.62       |  0.71    |  0.67      |  7 |
|2023 Cadillac Escalade           |  1.0        |  0.6     |  0.75      |  5 |
|2023 Cadillac Lyriq              |  1.0        |  0.25    |  0.4       |  4 |
|2023 Cadillac XT4                |  0.86       |  1.0     |  0.92      |  6 |
|2023 Cadillac XT5                |  0.75       |  0.86    |  0.8       |  7 |
|2023 Cadillac XT6                |  0.33       |  0.25    |  0.29      |  4 |
|2023 Chevrolet Blazer            |  0.8        |  0.57    |  0.67      |  7 |
|2023 Chevrolet Bolt EV           |  0.8        |  0.57    |  0.67      |  7 |
|2023 Chevrolet Camaro            |  0.6        |  0.6     |  0.6       |  5 |
|2023 Chevrolet Colorado          |  0.5        |  0.73    |  0.59      |  11 |
|2023 Chevrolet Corvette          |  0.5        |  0.43    |  0.46      |  7 |
|2023 Chevrolet Equinox           |  0.75       |  0.86    |  0.8       |  7 |
|2023 Chevrolet Malibu            |  0.89       |  1.0     |  0.94      |  8 |
|2023 Chevrolet Silverado 1500    |  0.67       |  0.5     |  0.57      |  4 |
|2023 Chevrolet Silverado 2500HD  |  0.62       |  0.56    |  0.59      |  9 |
|2023 Chevrolet Suburban          |  0.75       |  0.5     |  0.6       |  6 |
|2023 Chevrolet Tahoe             |  0.5        |  0.43    |  0.46      |  7 |
|2023 Chevrolet TrailBlazer       |  0.88       |  1.0     |  0.93      |  7 |
|2023 Chevrolet Traverse          |  0.83       |  1.0     |  0.91      |  5 |
|2023 Chrysler 300                |  0.83       |  0.71    |  0.77      |  7 |
|2023 Chrysler Pacifica           |  1.0        |  0.5     |  0.67      |  6 |
|2023 Dodge Challenger            |  0.58       |  0.78    |  0.67      |  9 |
|2023 Dodge Charger               |  1.0        |  0.29    |  0.44      |  7 |
|2023 Dodge Durango               |  0.0        |  0.0     |  0.0       |  1 |
|2023 Dodge Hornet                |  0.71       |  0.91    |  0.8       |  11 |
|2023 FIAT 500X                   |  0.8        |  0.89    |  0.84      |  9 |
|2023 Ford Bronco                 |  0.57       |  0.8     |  0.67      |  5 |
|2023 Ford Bronco Sport           |  0.5        |  0.56    |  0.53      |  9 |
|2023 Ford Edge                   |  0.71       |  0.83    |  0.77      |  6 |
|2023 Ford Escape                 |  0.72       |  0.93    |  0.81      |  14 |
|2023 Ford Expedition             |  1.0        |  0.67    |  0.8       |  3 |
|2023 Ford Explorer               |  1.0        |  0.94    |  0.97      |  16 |
|2023 Ford F-150                  |  1.0        |  0.5     |  0.67      |  4 |
|2023 Ford F-150 Lightning        |  0.57       |  0.67    |  0.62      |  6 |
|2023 Ford Maverick               |  0.85       |  0.94    |  0.89      |  18 |
|2023 Ford Mustang                |  0.8        |  0.67    |  0.73      |  6 |
|2023 Ford Mustang Mach-E         |  0.67       |  0.67    |  0.67      |  6 |
|2023 Ford Ranger                 |  0.89       |  0.89    |  0.89      |  9 |
|2023 GMC Acadia                  |  1.0        |  0.57    |  0.73      |  7 |
|2023 GMC Canyon                  |  0.67       |  0.67    |  0.67      |  3 |
|2023 GMC Sierra 1500             |  0.75       |  0.6     |  0.67      |  5 |
|2023 GMC Sierra 2500HD           |  0.8        |  0.8     |  0.8       |  5 |
|2023 GMC Terrain                 |  0.7        |  0.88    |  0.78      |  8 |
|2023 GMC Yukon                   |  1.0        |  0.75    |  0.86      |  4 |
|2023 Genesis G70                 |  0.6        |  0.38    |  0.46      |  8 |
|2023 Genesis G80                 |  1.0        |  0.33    |  0.5       |  3 |
|2023 Genesis G90                 |  1.0        |  0.8     |  0.89      |  5 |
|2023 Genesis GV60                |  0.67       |  0.4     |  0.5       |  5 |
|2023 Genesis GV70                |  0.83       |  1.0     |  0.91      |  15 |
|2023 Genesis GV80                |  0.86       |  0.86    |  0.86      |  14 |
|2023 Honda Accord                |  0.75       |  0.67    |  0.71      |  9 |
|2023 Honda CR-V                  |  0.75       |  0.5     |  0.6       |  6 |
|2023 Honda Civic                 |  0.88       |  1.0     |  0.93      |  7 |
|2023 Honda HR-V                  |  0.75       |  0.9     |  0.82      |  10 |
|2023 Honda Odyssey               |  0.85       |  1.0     |  0.92      |  11 |
|2023 Honda Passport              |  0.77       |  1.0     |  0.87      |  10 |
|2023 Honda Pilot                 |  0.67       |  0.4     |  0.5       |  5 |
|2023 Honda Ridgeline             |  0.67       |  0.29    |  0.4       |  7 |
|2023 Hyundai Elantra             |  0.75       |  1.0     |  0.86      |  15 |
|2023 Hyundai IONIQ 6             |  1.0        |  1.0     |  1.0       |  4 |
|2023 Hyundai Ioniq 5             |  0.86       |  0.67    |  0.75      |  9 |
|2023 Hyundai Kona                |  1.0        |  0.25    |  0.4       |  4 |
|2023 Hyundai Kona Electric       |  0.75       |  0.75    |  0.75      |  12 |
|2023 Hyundai Palisade            |  0.82       |  0.9     |  0.86      |  10 |
|2023 Hyundai Santa Cruz          |  1.0        |  0.5     |  0.67      |  6 |
|2023 Hyundai Santa Fe            |  1.0        |  1.0     |  1.0       |  12 |
|2023 Hyundai Sonata              |  0.67       |  0.57    |  0.62      |  7 |
|2023 Hyundai Tucson              |  0.89       |  0.94    |  0.91      |  17 |
|2023 Hyundai Venue               |  1.0        |  0.57    |  0.73      |  7 |
|2023 INFINITI Q50                |  0.75       |  0.43    |  0.55      |  7 |
|2023 INFINITI QX50               |  0.86       |  0.75    |  0.8       |  8 |
|2023 INFINITI QX80               |  1.0        |  0.83    |  0.91      |  6 |
|2023 Jaguar E-Pace               |  0.67       |  0.25    |  0.36      |  8 |
|2023 Jaguar F-Pace               |  0.67       |  1.0     |  0.8       |  14 |
|2023 Jaguar F-Type               |  1.0        |  0.33    |  0.5       |  6 |
|2023 Jaguar I-Pace               |  0.5        |  0.5     |  0.5       |  10 |
|2023 Jaguar XF                   |  0.8        |  0.67    |  0.73      |  6 |
|2023 Jeep Grand Cherokee         |  0.62       |  0.83    |  0.71      |  12 |
|2023 MINI Cooper                 |  1.0        |  0.67    |  0.8       |  6 |
|2023 Mercedes-Benz CLA Class     |  0.38       |  0.62    |  0.48      |  8 |
|2023 Mercedes-Benz CLS Class     |  0.6        |  0.38    |  0.46      |  8 |
|2023 Mercedes-Benz E Class       |  0.0        |  0.0     |  0.0       |  3 |
|2023 Mercedes-Benz EQS           |  0.43       |  0.38    |  0.4       |  8 |
|2023 Mercedes-Benz GLB Class     |  1.0        |  0.43    |  0.6       |  7 |
|2023 Mercedes-Benz GLC Class     |  0.57       |  0.8     |  0.67      |  10 |
|2023 Mercedes-Benz GLE Class     |  0.62       |  1.0     |  0.77      |  5 |
|2023 Mercedes-Benz GLS Class     |  0.44       |  0.5     |  0.47      |  8 |
|2023 Mercedes-Benz S Class       |  0.75       |  0.86    |  0.8       |  7 |
|2023 Mitsubishi Mirage           |  1.0        |  0.56    |  0.71      |  9 |
|2023 Mitsubishi Outlander        |  0.71       |  0.62    |  0.67      |  8 |
|2023 Nissan Altima               |  0.78       |  0.88    |  0.82      |  8 |
|2023 Nissan Ariya                |  0.71       |  0.71    |  0.71      |  7 |
|2023 Nissan Kicks                |  1.0        |  0.8     |  0.89      |  5 |
|2023 Nissan Leaf                 |  0.67       |  0.86    |  0.75      |  7 |
|2023 Nissan Murano               |  1.0        |  1.0     |  1.0       |  9 |
|2023 Nissan Sentra               |  1.0        |  0.71    |  0.83      |  7 |
|2023 Nissan Versa                |  1.0        |  0.67    |  0.8       |  6 |
|2023 Nissan Z                    |  0.8        |  0.89    |  0.84      |  9 |
|2023 Polestar 2                  |  0.82       |  1.0     |  0.9       |  9 |
|2023 Porsche 718                 |  0.71       |  0.83    |  0.77      |  6 |
|2023 Porsche 911                 |  0.62       |  0.71    |  0.67      |  7 |
|2023 Porsche Cayenne             |  0.75       |  0.5     |  0.6       |  6 |
|2023 Porsche Taycan              |  1.0        |  0.4     |  0.57      |  5 |
|2023 Ram 1500                    |  0.0        |  0.0     |  0.0       |  3 |
|2023 Ram 2500                    |  0.43       |  0.6     |  0.5       |  5 |
|2023 Rivian R1S                  |  0.8        |  1.0     |  0.89      |  4 |
|2023 Rivian R1T                  |  0.75       |  0.67    |  0.71      |  9 |
|2023 Subaru Crosstrek            |  0.62       |  0.71    |  0.67      |  7 |
|2023 Subaru Outback              |  0.0        |  0.0     |  0.0       |  2 |
|2023 Subaru WRX                  |  0.57       |  0.5     |  0.53      |  8 |
|2023 Tesla Model 3               |  0.8        |  0.8     |  0.8       |  5 |
|2023 Tesla Model S               |  0.56       |  0.56    |  0.56      |  9 |
|2023 Tesla Model X               |  0.86       |  0.67    |  0.75      |  9 |
|2023 Tesla Model Y               |  1.0        |  0.8     |  0.89      |  5 |
|2023 Toyota 4Runner              |  0.8        |  0.67    |  0.73      |  6 |
|2023 Toyota BZ4X                 |  0.8        |  1.0     |  0.89      |  8 |
|2023 Toyota Camry                |  0.85       |  1.0     |  0.92      |  11 |
|2023 Toyota Corolla              |  0.6        |  0.5     |  0.55      |  6 |
|2023 Toyota Crown                |  0.86       |  0.86    |  0.86      |  7 |
|2023 Toyota Highlander           |  0.0        |  0.0     |  0.0       |  3 |
|2023 Toyota Prius                |  0.67       |  0.8     |  0.73      |  10 |
|2023 Toyota Sequoia              |  0.75       |  0.86    |  0.8       |  7 |
|2023 Toyota Sienna               |  0.6        |  0.6     |  0.6       |  5 |
|2023 Toyota Tundra               |  1.0        |  0.4     |  0.57      |  5 |
|2023 Toyota Venza                |  0.73       |  1.0     |  0.84      |  8 |
|2023 Volkswagen Arteon           |  0.85       |  1.0     |  0.92      |  17 |
|2023 Volkswagen Atlas            |  0.82       |  1.0     |  0.9       |  9 |
|2023 Volkswagen Golf             |  0.75       |  0.6     |  0.67      |  5 |
|2023 Volkswagen ID               |  0.0        |  0.0     |  0.0       |  5 |
|2023 Volkswagen ID.4             |  0.0        |  0.0     |  0.0       |  5 |
|2023 Volkswagen Jetta            |  0.57       |  0.8     |  0.67      |  10 |
|2023 Volkswagen Taos             |  0.6        |  0.6     |  0.6       |  10 |
|2023 Volvo S60                   |  0.2        |  0.17    |  0.18      |  6 |
|2023 Volvo S90                   |  0.25       |  0.33    |  0.29      |  3 |
|2023 Volvo XC40                  |  0.57       |  0.67    |  0.62      |  6 |
|2023 Volvo XC60                  |  0.75       |  0.5     |  0.6       |  6 |
|2023 Volvo XC90                  |  0.67       |  0.75    |  0.71      |  8 |
|accuracy                         |             |          |  0.71      |  1284 |
|macro avg                        |  0.7        |  0.66    |  0.66      |  1284 |
|weighted avg                     |  0.72       |  0.71    |  0.7       |  1284 |





