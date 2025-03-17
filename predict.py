import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("final_banana_leaf_model.keras")

# Define class names (Ensure these match your dataset structure)
class_labels = ['born', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese', 'potassium', 'sulphur', 'zinc']

# Function to classify a new image
def predict_leaf_deficiency(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    print(f"Predicted Deficiency: {class_labels[predicted_class]} ({confidence*100:.2f}%)")

# Example usage
predict_leaf_deficiency("path_to_new_image.jpg")
