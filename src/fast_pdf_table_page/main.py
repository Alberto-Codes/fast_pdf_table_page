import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

# Imports for using docling's loader
from docling.models.layout_model import LayoutModel
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

class TableDetectionService:
    def __init__(self, pdf_path, output_folder, artifacts_path="artifacts/model_artifacts/layout", dpi=150, score_threshold=0.9):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.artifacts_path = Path(artifacts_path)
        self.dpi = dpi
        self.score_threshold = score_threshold
        os.makedirs(self.output_folder, exist_ok=True)

        # Check if the provided artifacts path exists
        if not self.artifacts_path.exists() or not (self.artifacts_path / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Layout model artifacts not found at expected location: {self.artifacts_path}"
                f" Ensure the directory exists and contains 'model.safetensors'."
            )
        print(f"Using layout artifacts from local path: {self.artifacts_path}")

        # Load the predictor from the specified local path
        # Using CPU device for now
        self.layout_predictor = LayoutPredictor(
            artifact_path=str(self.artifacts_path), device="cpu"
        )
        print("Layout predictor loaded.")

    def pdf_to_images(self):
        pages = convert_from_path(self.pdf_path, dpi=self.dpi)
        images = []
        for idx, page in enumerate(pages):
            # Convert PIL image to a NumPy array in BGR (OpenCV uses BGR)
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append((idx + 1, img))
        return images

    def detect_table(self, image):
        # Convert OpenCV image (BGR) to PIL Image (RGB) as LayoutPredictor expects PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Use the docling layout predictor
        predictions = self.layout_predictor.predict(image_pil)

        # Check if any prediction is a table with sufficient confidence
        # Docling uses string labels (e.g., 'Table') and confidence key
        table_label_str = "Table" # Case-sensitive match based on typical output
        for pred in predictions:
            # Check label first for efficiency
            if pred.get("label") == table_label_str:
                # Then check confidence
                if pred.get("confidence", 0) >= self.score_threshold:
                    # Found a table above threshold
                    return True

        # No tables detected above the threshold
        return False

    def save_page_as_pdf(self, page_number):
        doc = fitz.open(self.pdf_path)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_number-1, to_page=page_number-1)
        output_path = os.path.join(self.output_folder, f"page_{page_number}.pdf")
        new_doc.save(output_path)
        new_doc.close()

    def process_pdf(self):
        images = self.pdf_to_images()
        candidate_pages = []
        for page_number, image in images:
            if self.detect_table(image):
                candidate_pages.append(page_number)
                self.save_page_as_pdf(page_number)
        print("Pages with detected tables:", candidate_pages)
        return candidate_pages

# Example usage:
def main():
    pdf_path = "data/pdfs/2022_10k.pdf"  # Make sure this path is correct
    output_folder = "output/pdf_pages"
    # Specify the path to your downloaded artifacts if different from the default
    # artifacts_dir = "path/to/your/artifacts/layout_model" 
    service = TableDetectionService(pdf_path, output_folder, dpi=150, score_threshold=0.9)
    service.process_pdf()

if __name__ == "__main__":
    main()
