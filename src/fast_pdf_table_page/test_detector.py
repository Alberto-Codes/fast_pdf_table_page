"""Test script for the FastTableDetector."""
from pathlib import Path
import structlog
from table_detector import FastTableDetector

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

def process_pdf(pdf_path: Path) -> None:
    """Process a single PDF file and report results."""
    try:
        detector = FastTableDetector(
            scale_factor=0.3,
            min_table_area=200,
            structure_threshold=0.03
        )
        
        logger.info("processing_pdf", pdf_path=str(pdf_path))
        table_pages = detector.find_pages_with_tables(pdf_path)
        
        logger.info("pdf_processing_complete",
                   pdf_path=str(pdf_path),
                   total_pages_with_tables=len(table_pages),
                   table_pages=[p+1 for p in table_pages])  # Convert to 1-based page numbers
                   
    except Exception as e:
        logger.error("pdf_processing_failed",
                    pdf_path=str(pdf_path),
                    error=str(e))

def main():
    """Main function to process all PDFs in the data directory."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "pdfs"
    
    if not data_dir.exists():
        logger.error("data_directory_not_found", path=str(data_dir))
        return
        
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("no_pdf_files_found", directory=str(data_dir))
        return
        
    logger.info("found_pdf_files", count=len(pdf_files))
    
    for pdf_path in pdf_files:
        process_pdf(pdf_path)

if __name__ == "__main__":
    main() 