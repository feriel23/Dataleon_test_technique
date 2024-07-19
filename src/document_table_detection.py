from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import imghdr
import mimetypes

class DocumentTableDetector:
    def __init__(self):
        """Initialize the processor and model."""
        self.processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
        self.model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")
        self.model.eval()  # Set the model to evaluation mode
        
    def load_image(self, image_path):
        """Helper function to load an image from a URL or local path."""
        try:
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
            
    def predict(self, image_path):
        """Predict tables in an image."""
        # Check the image format using imghdr and mimetypes
        file_type = imghdr.what(image_path)
        mime_type, _ = mimetypes.guess_type(image_path)

        print(f"Detected file type: {file_type}")
        print(f"Detected MIME type: {mime_type}")

        if file_type not in ['jpeg', 'png', 'jpg'] and mime_type not in ['image/jpeg', 'image/png']:
            raise ValueError("Unsupported file format")

        try:
                image = self.load_image(image_path)
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.model(**inputs)

                # Converting outputs to COCO API and filtering by score threshold
                target_sizes = torch.tensor([image.size[::-1]])
                results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

                predictions = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    predictions.append(
                        {
                            "label": self.model.config.id2label[label.item()],
                            "confidence": round(score.item(), 3),
                            "box": box
                        }
                    )
                return predictions
        except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")
                
    def draw_boxes(self, predictions, image_path, show_scores=False):
        """
        Draw bounding boxes around detected tables on the image.
        Optionally, include detection confidence scores above the boxes.

        Args:
            predictions (list of dicts): List of predictions with scores, labels, and boxes.
            image_path (str): Path to the image file.
            show_scores (bool): If True, display the confidence scores above the bounding boxes.

        Returns:
            PIL.Image: The image with drawn bounding boxes.
        """
        image = self.load_image(image_path)
        draw = ImageDraw.Draw(image)

        # Use a larger font for drawing text, increase the font size 
        try:
            font = ImageFont.truetype("arial.ttf", 70) 
        except IOError:
            # Use a default font with larger size if arial.ttf is not available
            font = ImageFont.load_default(size=20)

        for prediction in predictions:
            box = prediction["box"]
            score = prediction["confidence"]
            # Draw the bounding box
            draw.rectangle(box, outline="red", width=2)

            # Optionally display the confidence score above the box
            if show_scores:
                score_text = f"{score:.2f}"
                # Position the text above the box
                text_position = (box[0], box[1] - 60)
                draw.text(text_position, score_text, fill="blue", font=font)

        return image