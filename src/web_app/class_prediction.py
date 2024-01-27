# Function to get predicted class
import torch
import pandas as pd

def get_predicted_class(model, image_tensor, class_to_idx, transform, spec_df):

    model.eval()  # Set the model to evaluation mode

    # Apply the transformation to the image
    transformed_image = transform(image_tensor).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        logits = model(transformed_image)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        idx_to_class = {idx: car_class for car_class, idx in class_to_idx.items()}
        predicted_class = idx_to_class[predicted_idx]
        df_slice = ' '.join([predicted_class, ''])

        spec_select = spec_df.loc[df_slice]
        spec_select = spec_select[['Make', 'Model', 'Year', 'MSRP', 'Gas Mileage', 'Engine', 'EPA Class']]
        df = pd.DataFrame(spec_select)

    return predicted_class, df