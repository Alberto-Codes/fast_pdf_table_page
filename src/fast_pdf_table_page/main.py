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

import structlog # Added structlog

# --- structlog configuration ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(min_level="info"),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger() # Create logger instance
# --- End structlog configuration ---

# Attempt to import torch for device detection, fail gracefully if not installed
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False
    log.warning("torch_not_found", reason="GPU acceleration will be unavailable.") # Changed to log.warning

# Imports from docling and related packages
from docling.models.layout_model import LayoutModel # Assuming this is okay per user
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# Global variable for the predictor within each worker process
_worker_predictor: Optional[LayoutPredictor] = None

def _init_worker(artifacts_path_str: str, device_str: str) -> None:
    """Initializes the LayoutPredictor in each worker process."""
    global _worker_predictor
    log.debug("worker.init.start", device=device_str, artifacts_path=artifacts_path_str)
    try:
        _worker_predictor = LayoutPredictor(
            artifact_path=artifacts_path_str, device=device_str
        )
        log.info("worker.init.success", device=device_str)
    except Exception as e:
        log.error("worker.init.fail", device=device_str, error=str(e), exc_info=True)

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
    log = structlog.get_logger().bind(page_number=page_number)

    if _worker_predictor is None:
        log.error("worker.predictor_missing", reason="Predictor not initialized")
        return None

    try:
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))

        # Predict and immediately convert generator to list
        start_pred = time.time()
        predictions_list = list(_worker_predictor.predict(image_pil))
        pred_time = time.time() - start_pred
        # Now we can get the length from the list
        log.debug("worker.predict.run", duration_sec=round(pred_time, 3), num_predictions=len(predictions_list))

        # Check for tables using the list
        table_label_str = "Table"
        for pred_idx, pred in enumerate(predictions_list):
            if (
                pred.get("label") == table_label_str
                and pred.get("confidence", 0) >= score_threshold
            ):
                log.debug("worker.table_found", confidence=pred.get("confidence"))
                return page_number  # Return page number if table found
        # log.debug("worker.no_table_found") # Optional: log if no table found
        return None
    except Exception as e:
        log.error("worker.process.fail", error=str(e), exc_info=True)
        return None


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
        dpi: int = 150,
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
            self.log.error("init.artifacts.fail", path=str(self.artifacts_path))
            raise FileNotFoundError(
                f"Layout model artifacts directory not found or missing "
                f"'model.safetensors' at: {self.artifacts_path}"
            )
        print(f"Using layout artifacts from local path: {self.artifacts_path}")

        # Note: Predictor is now initialized in worker processes, not here.
        # print("Layout predictor will be loaded in worker processes.")

        self.log = structlog.get_logger().bind(service_instance=id(self)) # Bind instance ID for context

    def pdf_to_images(self) -> List[Tuple[int, np.ndarray]]:
        """
        Converts PDF content (bytes) to a list of images.

        Returns:
            A list of tuples, each containing (page_number, image_data_as_numpy_array).
        """
        self.log.info("pdf_to_images.start", dpi=self.dpi, pdf_bytes=len(self.pdf_content))
        start_time = time.time()
        try:
            # Use convert_from_bytes instead of convert_from_path
            pages = convert_from_bytes(self.pdf_content, dpi=self.dpi)
        except Exception as e:
            self.log.error("pdf_to_images.fail", error=str(e), exc_info=True)
            return [] # Return empty list on error

        images = []
        for idx, page in enumerate(pages):
            img = np.array(page)
            # Ensure image is in BGR format for consistency if needed later,
            # although worker converts to RGB for PIL.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append((idx + 1, img)) # Use 1-based page numbers

        end_time = time.time()
        self.log.info("pdf_to_images.success", count=len(images), duration_sec=round(end_time - start_time, 2))
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
        log = self.log.bind(page_number=page_number)
        log.debug("extract_page.start")
        try:
            # Open PDF from bytes
            with fitz.open("pdf", self.pdf_content) as doc, fitz.open() as new_doc:
                 # fitz uses 0-based indexing
                new_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                # Return blob instead of saving
                blob = new_doc.tobytes()
                log.debug("extract_page.success", blob_size=len(blob))
                return blob
        except Exception as e:
            log.error("extract_page.fail", error=str(e), exc_info=True)
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
        log = self.log # Use instance logger
        log.info("process.start")
        overall_start_time = time.time()

        images = self.pdf_to_images()
        if not images:
            log.warning("process.no_images", reason="Aborting PDF processing.")
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
            log.error("process.pool.fail", error=str(e), exc_info=True)
            # Continue with any results gathered so far, but log the error.

        detection_time = time.time()
        log.info("process.pool.finish", duration_sec=round(detection_time - detection_start_time, 2))

        # Sort pages for consistent output order
        candidate_pages.sort()
        log.info("process.found_pages", count=len(candidate_pages), pages=candidate_pages)

        print(f"Found {len(candidate_pages)} pages with potential tables: {candidate_pages}")

        # Extract page blobs for the identified pages
        page_blobs: List[Tuple[int, bytes]] = []
        if candidate_pages:
            log.info("process.extract.start", count=len(candidate_pages))
            extract_start_time = time.time()
            for page_number in candidate_pages:
                page_blob = self.extract_page_blob(page_number)
                if page_blob:
                    page_blobs.append((page_number, page_blob))
            extract_time = time.time()
            log.info("process.extract.finish", duration_sec=round(extract_time - extract_start_time, 2))
        else:
            log.info("process.extract.skip", reason="No candidate pages found.")

        overall_end_time = time.time()
        log.info("process.finish", total_duration_sec=round(overall_end_time - overall_start_time, 2))
        # Return list of (page_number, page_blob) tuples
        return page_blobs

# Example usage:
def main():
    log = structlog.get_logger()
    pdf_path = Path("data/pdfs/CORP 10K 2024 - FINAL.pdf")  # Path to the source PDF
    output_folder = Path("output/pdf_pages")
    artifacts_dir = "artifacts/model_artifacts/layout" # Path to model artifacts
    dpi_setting = 100 # Example DPI
    threshold_setting = 0.95 # Example threshold

    # --- Load PDF into a blob --- 
    try:
        log.info("main.load_pdf.start", path=str(pdf_path))
        with open(pdf_path, "rb") as f:
            pdf_blob = f.read()
        log.info("main.load_pdf.success", path=str(pdf_path), pdf_bytes=len(pdf_blob))
    except FileNotFoundError:
        log.error("main.load_pdf.not_found", path=str(pdf_path))
        return
    except Exception as e:
        log.error("main.load_pdf.fail", path=str(pdf_path), error=str(e), exc_info=True)
        return
    
    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Instantiate the service with the blob --- 
    log.info("main.service.init_start")
    service = TableDetectionService(
        output_folder=str(output_folder), # Pass output folder path as string
        pdf_content=pdf_blob,             # Pass the loaded blob
        artifacts_path=artifacts_dir,
        dpi=dpi_setting,
        score_threshold=threshold_setting,
        # device=None # Let it auto-detect or specify 'cpu', 'cuda'
    )
    log.info("main.service.init_finish")

    # --- Process the PDF blob --- 
    log.info("main.service.process_start")
    table_page_blobs = service.process_pdf()
    log.info("main.service.process_finish", extracted_page_count=len(table_page_blobs))

    # --- Save the resulting page blobs to files (for development/testing) ---
    if table_page_blobs:
        log.info("main.save_blobs.start", count=len(table_page_blobs), output_dir=str(output_folder))
        for page_number, page_blob in table_page_blobs:
            output_file_path = output_folder / f"page_{page_number}.pdf"
            try:
                with open(output_file_path, "wb") as f:
                    f.write(page_blob)
                # print(f"  Saved: {output_file_path.name}") # Optional print per file
            except Exception as e:
                log.error("main.save_blob.fail", file=output_file_path.name, error=str(e), exc_info=True)
        log.info("main.save_blobs.finish")
    else:
        log.info("main.save_blobs.skip", reason="No blobs extracted.")

if __name__ == "__main__":
    main()
