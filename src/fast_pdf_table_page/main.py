import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import multiprocessing
import time
from typing import Iterable, Optional, List, Tuple, Any # Added typing imports

# Attempt to import torch for device detection, fail gracefully if not installed
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False
    print("Warning: torch not found. GPU acceleration will be unavailable.")

# Imports from docling and related packages
from docling.models.layout_model import LayoutModel # Assuming this is okay per user
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# Global variable for the predictor within each worker process
_worker_predictor: Optional[LayoutPredictor] = None

def _init_worker(artifacts_path_str: str, device_str: str) -> None:
    """Initializes the LayoutPredictor in each worker process."""
    global _worker_predictor
    process_id = os.getpid()
    print(f"Initializing predictor in worker {process_id} on device '{device_str}'...")
    try:
        _worker_predictor = LayoutPredictor(
            artifact_path=artifacts_path_str, device=device_str
        )
        print(f"Predictor initialized successfully in worker {process_id}.")
    except Exception as e:
        print(f"Error initializing predictor in worker {process_id}: {e}")
        # Worker will fail tasks if predictor is None

def _detect_table_worker(args: Tuple[int, np.ndarray, float]) -> Optional[int]:
    """
    Worker function to detect tables on a single page image using the
    pre-initialized predictor.

    Args:
        args: A tuple containing (page_number, image_data, score_threshold).

    Returns:
        The page number if a table is detected above the threshold, otherwise None.
    """
    page_number, image_data, score_threshold = args
    global _worker_predictor

    if _worker_predictor is None:
        print(f"Error: Predictor not initialized in worker {os.getpid()}. Skipping page {page_number}.")
        return None

    try:
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))

        # Predict
        predictions = _worker_predictor.predict(image_pil)

        # Check for tables
        table_label_str = "Table"
        for pred in predictions:
            if (
                pred.get("label") == table_label_str
                and pred.get("confidence", 0) >= score_threshold
            ):
                return page_number  # Return page number if table found

        return None  # Return None if no table found
    except Exception as e:
        print(f"Error processing page {page_number} in worker {os.getpid()}: {e}")
        return None # Ensure worker returns None on error


class TableDetectionService:
    """
    Service to detect pages containing tables in PDF content (bytes) using a
    pre-trained layout detection model.
    """

    def __init__(
        self,
        output_folder: str, # Keep output folder for potential saving/debugging
        pdf_content: bytes, # Changed from pdf_path
        artifacts_path: str = "artifacts/model_artifacts/layout",
        dpi: int = 300,
        score_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initializes the TableDetectionService with PDF content as bytes.

        Args:
            output_folder: Path to a directory (used if saving blobs later).
            pdf_content: The content of the PDF file as bytes.
            artifacts_path: Path to the local directory containing the layout
                            model artifacts (e.g., model.safetensors).
            dpi: Dots per inch resolution for rendering PDF pages to images.
            score_threshold: Minimum confidence score for detecting a table.
            device: The device to run the model on ('cpu', 'cuda', etc.).
                    If None, automatically selects CUDA if available, else CPU.
        """
        # self.pdf_path = Path(pdf_path) # Removed
        self.pdf_content = pdf_content # Added
        self.output_folder = Path(output_folder)
        self.artifacts_path = Path(artifacts_path)
        self.dpi = dpi
        self.score_threshold = score_threshold
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Determine device
        if device:
            self.device = device
        elif _torch_available and torch.cuda.is_available():
             self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Validate artifacts path immediately
        model_file = self.artifacts_path / "model.safetensors"
        if not self.artifacts_path.is_dir() or not model_file.is_file():
            raise FileNotFoundError(
                f"Layout model artifacts directory not found or missing "
                f"'model.safetensors' at: {self.artifacts_path}"
            )
        print(f"Using layout artifacts from local path: {self.artifacts_path}")

        # Note: Predictor is now initialized in worker processes, not here.
        # print("Layout predictor will be loaded in worker processes.")


    def pdf_to_images(self) -> List[Tuple[int, np.ndarray]]:
        """
        Converts PDF content (bytes) to a list of images.

        Returns:
            A list of tuples, each containing (page_number, image_data_as_numpy_array).
        """
        print(f"Converting PDF content ({len(self.pdf_content)} bytes) to images at {self.dpi} DPI...")
        start_time = time.time()
        try:
            # Use convert_from_bytes instead of convert_from_path
            pages = convert_from_bytes(self.pdf_content, dpi=self.dpi)
        except Exception as e:
            print(f"Error during PDF content to image conversion: {e}")
            return [] # Return empty list on error

        images = []
        for idx, page in enumerate(pages):
            img = np.array(page)
            # Ensure image is in BGR format for consistency if needed later,
            # although worker converts to RGB for PIL.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append((idx + 1, img)) # Use 1-based page numbers

        end_time = time.time()
        print(f"Converted {len(images)} pages in {end_time - start_time:.2f} seconds.")
        return images

    # Removed detect_table method (now handled by _detect_table_worker)

    def extract_page_blob(self, page_number: int) -> Optional[bytes]:
        """
        Extracts a single page from the source PDF content and returns it as bytes.

        Args:
            page_number: The 1-based index of the page to extract.
        
        Returns:
            The content of the extracted page as a PDF blob (bytes), or None on error.
        """
        try:
            # Open PDF from bytes
            with fitz.open("pdf", self.pdf_content) as doc, fitz.open() as new_doc:
                 # fitz uses 0-based indexing
                new_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                # Return blob instead of saving
                return new_doc.tobytes()
        except Exception as e:
            print(f"Error extracting page {page_number} blob: {e}")
            return None

    def process_pdf(self) -> List[Tuple[int, bytes]]:
        """
        Processes the PDF content to find pages with tables and returns them as blobs.

        Uses multiprocessing for parallel page detection.

        Returns:
            A list of tuples, where each tuple contains:
            (page_number, page_content_blob)
            The list is sorted by page number.
        """
        overall_start_time = time.time()

        images = self.pdf_to_images()
        if not images:
            print("No images generated from PDF content. Aborting.")
            return []

        # Prepare arguments for the worker function
        # Note: Artifacts path and device are passed via initializer now
        tasks = [
            (page_number, image_data, self.score_threshold)
            for page_number, image_data in images
        ]

        # Use multiprocessing pool
        cpu_cores = os.cpu_count()
        num_workers = max(1, cpu_cores - 1) if cpu_cores is not None else 1
        print(f"Starting parallel table detection with {num_workers} workers...")
        detection_start_time = time.time()

        candidate_pages: List[int] = []
        try:
            with multiprocessing.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(str(self.artifacts_path), self.device), # Pass args for init
            ) as pool:
                results = pool.imap_unordered(_detect_table_worker, tasks)
                for page_result in results:
                    if page_result is not None:
                        candidate_pages.append(page_result)
        except Exception as e:
            print(f"Error occurred during multiprocessing: {e}")
            # Continue with any results gathered so far, but log the error.

        detection_time = time.time()
        print(f"Parallel detection finished in {detection_time - detection_start_time:.2f} seconds.")

        # Sort pages for consistent output order
        candidate_pages.sort()

        print(f"Found {len(candidate_pages)} pages with potential tables: {candidate_pages}")

        # Extract page blobs for the identified pages
        page_blobs: List[Tuple[int, bytes]] = []
        if candidate_pages:
            print("Extracting identified page blobs...")
            extract_start_time = time.time()
            for page_number in candidate_pages:
                page_blob = self.extract_page_blob(page_number)
                if page_blob:
                    page_blobs.append((page_number, page_blob))
            extract_time = time.time()
            print(f"Page blob extraction took {extract_time - extract_start_time:.2f} seconds.")
        else:
            print("No pages with tables found to extract.")

        overall_end_time = time.time()
        print(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds.")
        # Return list of (page_number, page_blob) tuples
        return page_blobs

# Example usage:
def main():
    pdf_path = Path("data/pdfs/2022_10k.pdf")  # Path to the source PDF
    output_folder = Path("output/pdf_pages")
    artifacts_dir = "artifacts/model_artifacts/layout" # Path to model artifacts
    dpi_setting = 150 # Example DPI
    threshold_setting = 0.95 # Example threshold

    # --- Load PDF into a blob --- 
    try:
        print(f"Loading PDF file: {pdf_path}")
        with open(pdf_path, "rb") as f:
            pdf_blob = f.read()
        print(f"Loaded {len(pdf_blob)} bytes.")
    except FileNotFoundError:
        print(f"Error: Input PDF not found at {pdf_path}")
        return
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return
    
    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Instantiate the service with the blob --- 
    service = TableDetectionService(
        output_folder=str(output_folder), # Pass output folder path as string
        pdf_content=pdf_blob,             # Pass the loaded blob
        artifacts_path=artifacts_dir,
        dpi=dpi_setting,
        score_threshold=threshold_setting,
        # device=None # Let it auto-detect or specify 'cpu', 'cuda'
    )

    # --- Process the PDF blob --- 
    # Returns a list of tuples: (page_number, page_blob)
    table_page_blobs = service.process_pdf()

    # --- Save the resulting page blobs to files (for development/testing) ---
    if table_page_blobs:
        print(f"\nSaving {len(table_page_blobs)} extracted page blobs to files in '{output_folder}'...")
        for page_number, page_blob in table_page_blobs:
            output_file_path = output_folder / f"page_{page_number}.pdf"
            try:
                with open(output_file_path, "wb") as f:
                    f.write(page_blob)
                # print(f"  Saved: {output_file_path.name}") # Optional print per file
            except Exception as e:
                print(f"  Error saving {output_file_path.name}: {e}")
        print("Finished saving page blobs.")
    else:
        print("\nNo table page blobs were extracted to save.")

if __name__ == "__main__":
    main()
