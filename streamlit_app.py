import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import firebase_admin
# from firebase_admin import credentials, firestore, storage
from datetime import datetime
# import uuid
# import io

# Initialize Firebase
# if not firebase_admin._apps:
#     cred = credentials.Certificate("/workspaces/rice-leaf-diseasedetection/firebasetoken/riceleafdiseases-87f44-firebase-adminsdk-fbsvc-bc35a716a3.json")
#     firebase_admin.initialize_app(cred, {
#         'storageBucket': 'riceleafdiseases-87f44'
#     })

# db = firestore.client()
# bucket = storage.bucket()

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLOv8 model"""
    model = YOLO('/workspaces/rice-leaf-diseasedetection/model/best.pt')
    return model

def process_image(image_pil):
    """
    Process image for YOLOv8 prediction:
    1. Convert to RGB if image has alpha channel
    2. Convert to numpy array
    """
    # Convert RGBA to RGB if necessary
    if image_pil.mode == 'RGBA':
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # Convert to numpy array
    return np.array(image_pil)

def get_severity(box_area, image_area):
    """Calculate severity based on affected area ratio"""
    ratio = box_area / image_area
    if ratio < 0.2:
        return "Low"
    elif ratio < 0.5:
        return "Medium"
    else:
        return "High"

def predict_disease(model, image):
    """Predict disease using YOLOv8 model"""
    try:
        # Process image
        img = process_image(image)
        
        # Get image dimensions for severity calculation
        img_width, img_height = image.size
        image_area = img_width * img_height
        
        # Run prediction
        results = model(img)
        
        # Process results
        predictions = []
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # Get box coordinates
            confidence = float(result.conf)  # Get confidence score
            class_id = int(result.cls)  # Get class ID
            class_name = model.names[class_id]  # Get class name
            
            # Calculate box area for severity
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            severity = get_severity(box_area, image_area)
            
            predictions.append({
                'disease': class_name,
                'confidence': confidence,
                'severity': severity,
                'box': box.tolist()
            })
        
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return []

def draw_boxes(image, predictions):
    """Draw bounding boxes and labels on the image using PIL"""
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create copy of image
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for pred in predictions:
        box = pred['box']
        # Draw rectangle
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline="green",
            width=2
        )
        
        # Add label
        label = f"{pred['disease']} ({pred['confidence']:.2f})"
        draw.text(
            (box[0], box[1] - 20),
            label,
            fill="green",
            font=font
        )
    
    return img_draw

# def save_to_firebase(original_image, annotated_image, predictions):
#     """Save detection results and images to Firebase"""
#     try:
#         # Generate unique ID
#         record_id = str(uuid.uuid4())
        
#         # Ensure images are in RGB mode before saving
#         if original_image.mode != 'RGB':
#             original_image = original_image.convert('RGB')
#         if annotated_image.mode != 'RGB':
#             annotated_image = annotated_image.convert('RGB')
        
#         # Save original image
#         original_path = f"leaf_images/original/{record_id}.jpg"
#         original_blob = bucket.blob(original_path)
#         img_byte_arr = io.BytesIO()
#         original_image.save(img_byte_arr, format='JPEG')
#         original_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/jpeg')
        
#         # Save annotated image
#         annotated_path = f"leaf_images/annotated/{record_id}.jpg"
#         annotated_blob = bucket.blob(annotated_path)
#         img_byte_arr = io.BytesIO()
#         annotated_image.save(img_byte_arr, format='JPEG')
#         annotated_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/jpeg')
        
#         # Save to Firestore
#         doc_ref = db.collection('disease_detections').document(record_id)
#         doc_ref.set({
#             'timestamp': datetime.now(),
#             'predictions': predictions,
#             'original_image_url': original_blob.public_url,
#             'annotated_image_url': annotated_blob.public_url
#         })
        
#         return True, record_id
#     except Exception as e:
#         st.error(f"Error saving to Firebase: {str(e)}")
#         return False, None

def main():
    st.title("Rice Leaf Disease Detection")
    st.write("Upload an image of a rice leaf to detect diseases")
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add a button to trigger detection
            if st.button('Detect Disease'):
                with st.spinner('Analyzing image...'):
                    # Get predictions
                    predictions = predict_disease(model, image)
                    
                    if predictions:
                        # Draw boxes on image
                        annotated_image = draw_boxes(image, predictions)
                        
                        # Save results to Firebase
                        # save_success, record_id = save_to_firebase(image, annotated_image, predictions)
                        
                        # Display results
                        st.success("Analysis Complete!")
                        st.image(annotated_image, caption='Detection Results', use_column_width=True)
                        
                        st.write("### Detected Diseases:")
                        for pred in predictions:
                            st.write(f"""
                            - **Disease:** {pred['disease']}
                            - **Confidence:** {pred['confidence']:.2%}
                            - **Severity:** {pred['severity']}
                            """)
                        
                        # if save_success:
                        #     st.info(f"Results saved successfully! Record ID: {record_id}")
                    else:
                        st.warning("No diseases detected in the image.")
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()