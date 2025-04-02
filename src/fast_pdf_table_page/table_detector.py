"""Fast table detection in PDF pages using OpenCV."""
from pathlib import Path
from typing import List, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

import cv2
import fitz  # type: ignore
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

class FastTableDetector:
    """Rapidly detect tables in PDF pages using morphological operations and grid analysis."""
    
    def __init__(
        self,
        scale_factor: float = 0.3,  # Slightly larger for better detail
        min_table_area: int = 200,  # Even more lenient
        structure_threshold: float = 0.03,  # Minimum line density
    ):
        """Initialize the table detector with configurable parameters.
        
        Args:
            scale_factor: Factor to downscale images for faster processing
            min_table_area: Minimum area for a contour to be considered a table
            structure_threshold: Minimum line density threshold
        """
        self.scale_factor = scale_factor
        self.min_table_area = min_table_area
        self.structure_threshold = structure_threshold

    def _preprocess_page(self, page_pixmap: np.ndarray) -> np.ndarray:
        """Preprocess the page image for table detection.
        
        Args:
            page_pixmap: Input page image
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(page_pixmap.shape) == 3:
            gray = cv2.cvtColor(page_pixmap, cv2.COLOR_BGR2GRAY)
        else:
            gray = page_pixmap

        # Aggressive downscaling for speed
        height = int(gray.shape[0] * self.scale_factor)
        width = int(gray.shape[1] * self.scale_factor)
        scaled = cv2.resize(gray, (width, height))

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(scaled, 9, 75, 75)

        # Adaptive thresholding with a smaller block size
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            7,  # Smaller block size
            2
        )
        
        return binary

    def _detect_grid_structure(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect grid-like structures using morphological operations.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines, combined_lines)
        """
        height, width = binary_image.shape
        
        # Create structure elements for lines
        h_kernel_length = max(width // 40, 20)  # More sensitive to shorter lines
        v_kernel_length = max(height // 40, 20)

        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
        temp_h = cv2.erode(binary_image, h_kernel)
        horizontal_lines = cv2.dilate(temp_h, h_kernel)

        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_length))
        temp_v = cv2.erode(binary_image, v_kernel)
        vertical_lines = cv2.dilate(temp_v, v_kernel)

        # Combine lines
        combined = cv2.add(horizontal_lines, vertical_lines)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return horizontal_lines, vertical_lines, combined

    def _analyze_region(
        self, 
        region: np.ndarray, 
        h_lines: np.ndarray, 
        v_lines: np.ndarray
    ) -> bool:
        """Analyze a region for table-like properties.
        
        Args:
            region: Binary image region to analyze
            h_lines: Horizontal lines image
            v_lines: Vertical lines image
            
        Returns:
            True if region likely contains a table
        """
        height, width = region.shape
        total_pixels = height * width
        
        if total_pixels == 0:
            return False

        # Count line pixels
        h_pixels = cv2.countNonZero(h_lines)
        v_pixels = cv2.countNonZero(v_lines)
        
        # Calculate line densities
        h_density = h_pixels / total_pixels
        v_density = v_pixels / total_pixels
        
        # Check for minimum line density in both directions
        return (h_density >= self.structure_threshold and 
                v_density >= self.structure_threshold)

    def _has_table_structure(self, binary_image: np.ndarray) -> bool:
        """Detect if the binary image contains table-like structures.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            True if table-like structure detected, False otherwise
        """
        height, width = binary_image.shape
        min_dim = min(height, width)
        
        # Detect grid structure
        h_lines, v_lines, combined = self._detect_grid_structure(binary_image)
        
        # Find contours in the combined image
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:3]:  # Check only top 3 largest contours
            area = cv2.contourArea(contour)
            if area < self.min_table_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small relative to page
            if w < min_dim * 0.1 or h < min_dim * 0.1:
                continue
                
            # Analyze the region
            region = binary_image[y:y+h, x:x+w]
            h_region = h_lines[y:y+h, x:x+w]
            v_region = v_lines[y:y+h, x:x+w]
            
            if self._analyze_region(region, h_region, v_region):
                return True
        
        return False

    def _process_page(self, page_data: tuple) -> tuple[int, bool]:
        """Process a single page and return its number and whether it contains a table.
        
        Args:
            page_data: Tuple of (page_number, page_image)
            
        Returns:
            Tuple of (page_number, has_table)
        """
        page_num, img = page_data
        binary = self._preprocess_page(img)
        has_table = self._has_table_structure(binary)
        return page_num, has_table

    def find_pages_with_tables(self, pdf_input: str | Path | bytes, max_workers: int | None = None) -> List[int]:
        """Find page numbers containing potential tables in a PDF using multiprocessing.
        
        Args:
            pdf_input: Path to the PDF file (str or Path) or PDF content as bytes.
            max_workers: Maximum number of worker processes. If None, uses CPU count.
            
        Returns:
            List of page numbers (0-based) containing potential tables
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        table_pages = []
        input_type = "path" if isinstance(pdf_input, (str, Path)) else "blob"
        log_context = {"max_workers": max_workers, "input_type": input_type}

        doc = None
        try:
            if isinstance(pdf_input, (str, Path)):
                pdf_path = Path(pdf_input)
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                log_context["pdf_path"] = str(pdf_path)
                doc = fitz.open(pdf_path)
            elif isinstance(pdf_input, bytes):
                log_context["pdf_blob_size"] = len(pdf_input)
                doc = fitz.open(stream=pdf_input, filetype="pdf")
            else:
                raise TypeError("pdf_input must be a file path (str or Path) or bytes")

            total_pages = len(doc)
            log_context["total_pages"] = total_pages
            logger.info("starting_table_detection", **log_context)

            # Prepare page data for parallel processing
            page_data = []
            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                page_data.append((page_num, img))

            # Process pages in parallel
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Pass self to map function if needed, or make _process_page static
                # For simplicity, let's assume _process_page uses instance attributes
                results = list(executor.map(self._process_page, page_data))
            
            # Collect results
            table_pages = [page_num for page_num, has_table in results if has_table]
            
            duration = time.time() - start_time
            logger.info("completed_table_detection",
                       tables_found=len(table_pages),
                       duration_seconds=duration,
                       pages_per_second=(total_pages / duration) if duration > 0 else float('inf'),
                       **log_context) # Add original context back
            
            return table_pages
            
        except Exception as e:
            logger.error("table_detection_failed", error=str(e), **log_context)
            raise
        finally:
            if doc:
                doc.close() 