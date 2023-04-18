from face_extract import faceExtract
import numpy as np
import os




def check_path(PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)


def verify(model, APP_INP):

    APP_VERIFY = './data/app_data/verify_images'
    check_path(APP_INP)
    check_path(APP_VERIFY)

    # Build results array
    detection_threshold, verification_threshold = 0.9, 0.5
    results = []
    for image in os.listdir(APP_VERIFY):
        input_img = faceExtract(os.path.join(APP_INP,'input_image.jpg'))
        validation_img = faceExtract(os.path.join(APP_VERIFY, image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=0)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(APP_VERIFY)) 
    verified = verification > verification_threshold
    
    return verified

print("Function made successfully!!!!")