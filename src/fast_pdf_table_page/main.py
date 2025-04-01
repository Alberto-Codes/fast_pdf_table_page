import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import multiprocessing
import time # Optional: for timing

# Imports for using docling's loader
from docling.models.layout_model import LayoutModel
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# Worker function for multiprocessing
# Must be defined at the top level
def _detect_table_worker(args):
    """Worker function to detect tables on a single page image."""
    page_number, image_data, artifacts_path_str, score_threshold = args

    try:
        # Initialize predictor within the worker
        # This avoids pickling complex objects but adds init overhead per task
        predictor = LayoutPredictor(artifact_path=artifacts_path_str, device="cpu")

        # Convert OpenCV image (BGR) to PIL Image (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))

        # Predict
        predictions = predictor.predict(image_pil)

        # Check for tables
        table_label_str = "Table"
        for pred in predictions:
            if pred.get("label") == table_label_str and pred.get("confidence", 0) >= score_threshold:
                # print(f"Table found on page {page_number}") # Optional debug print
                return page_number # Return page number if table found

        # print(f"No table found on page {page_number}") # Optional debug print
        return None # Return None if no table found
    except Exception as e:
        print(f"Error processing page {page_number} in worker: {e}")
        return None # Ensure worker returns None on error

class TableDetectionService:
    def __init__(self, pdf_path, output_folder, artifacts_path="artifacts/model_artifacts/layout", dpi=300, score_threshold=0.5):
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

    def save_page_as_pdf(self, page_number):
        doc = fitz.open(self.pdf_path)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_number-1, to_page=page_number-1)
        output_path = os.path.join(self.output_folder, f"page_{page_number}.pdf")
        new_doc.save(output_path)
        new_doc.close()

    def process_pdf(self):
        print("Converting PDF to images...")
        start_time = time.time() # Optional timing
        images = self.pdf_to_images()
        conversion_time = time.time() # Optional timing
        print(f"PDF to image conversion took {conversion_time - start_time:.2f} seconds.")

        # Prepare arguments for the worker function
        # Pass artifact path as string
        tasks = [
            (page_number, image, str(self.artifacts_path), self.score_threshold)
            for page_number, image in images
        ]

        # Use multiprocessing pool
        # Use cpu_count() - 1 or a fixed number if preferred
        num_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
        print(f"Starting parallel table detection with {num_workers} workers...")
        detection_start_time = time.time() # Optional timing

        candidate_pages = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for potentially better memory usage and responsiveness
            # as results are processed as they complete
            results = pool.imap_unordered(_detect_table_worker, tasks)
            for page_result in results:
                if page_result is not None:
                    candidate_pages.append(page_result)

        detection_time = time.time() # Optional timing
        print(f"Parallel detection took {detection_time - detection_start_time:.2f} seconds.")

        # Sort pages for consistent output order
        candidate_pages.sort()

        print(f"Found {len(candidate_pages)} pages with potential tables: {candidate_pages}")

        # Save the identified pages sequentially
        if candidate_pages:
            print("Saving identified pages as PDFs...")
            save_start_time = time.time() # Optional timing
            for page_number in candidate_pages:
                self.save_page_as_pdf(page_number)
            save_time = time.time() # Optional timing
            print(f"Saving pages took {save_time - save_start_time:.2f} seconds.")
        else:
            print("No pages with tables found to save.")

        total_time = time.time() # Optional timing
        print(f"Total processing time: {total_time - start_time:.2f} seconds.")
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
