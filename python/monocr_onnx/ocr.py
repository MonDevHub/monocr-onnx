from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from .predictor import MonOCR
from .utils import calculate_accuracy

def read_image(image_path, model_path=None, charset_path=None):
    """
    Recognize text from an image file.
    
    Args:
        image_path (str or Path): Path to the image file.
        model_path (str, optional): Path to the ONNX model. If None, uses auto-downloaded model.
        charset_path (str, optional): Path to the charset file. If None, uses bundled charset.
        
    Returns:
        str: Recognized text.
    """
    ocr = MonOCR(model_path, charset_path)
    return ocr.predict(image_path)

from concurrent.futures import ThreadPoolExecutor

def read_images(image_paths, model_path=None, charset_path=None, workers=4):
    """
    Recognize text from multiple image files in parallel.
    
    Args:
        image_paths (list[str or Path]): List of paths to image files.
        model_path (str, optional): Path to model.
        charset_path (str, optional): Path to charset.
        workers (int, optional): Number of parallel workers. Defaults to 4.
        
    Returns:
        list[str]: List of recognized text for each image.
    """
    ocr = MonOCR(model_path, charset_path)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ocr.predict, image_paths))
        
    return results

def read_pdf(pdf_path, model_path=None, charset_path=None):
    """
    Recognize text from a PDF file.
    
    Args:
        pdf_path (str or Path): Path to the PDF file.
        model_path (str, optional): Path to model.
        charset_path (str, optional): Path to charset.
        
    Returns:
        list[str]: list of recognized text per page.
    """
    try:
        page_count = pdfinfo_from_path(str(pdf_path))["Pages"]
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF. Ensure poppler-utils is installed. Error: {e}")

    ocr = MonOCR(model_path, charset_path)
    # One page at a time: at 300 dpi an A4 page is about 25 MB of RGB, so
    # rasterising the whole document up front held every page in memory at
    # once, and a few hundred pages exhausted it before the first was read.
    results = []
    for page in range(1, page_count + 1):
        try:
            images = convert_from_path(str(pdf_path), dpi=300, first_page=page, last_page=page)
        except Exception as e:
            raise RuntimeError(f"Failed to convert page {page} of the PDF. Error: {e}")
        results.extend(ocr.predict(img) for img in images)

    return results

def read_pdfs(pdf_paths, model_path=None, charset_path=None, workers=4):
    """
    Recognize text from multiple PDF files.
    
    Args:
        pdf_paths (list[str or Path]): List of paths to PDF files.
        model_path (str, optional): Path to model.
        charset_path (str, optional): Path to charset.
        workers (int, optional): Workers for parallelism.
        
    Returns:
        list[list[str]]: List of lists of recognized text per page for each PDF.
    """
    def process_single_pdf(path):
        return read_pdf(path, model_path, charset_path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_single_pdf, pdf_paths))
        
    return results

def read_image_with_accuracy(image_path, ground_truth, model_path=None, charset_path=None):
    """
    Recognize text and calculate accuracy against ground truth.
    
    Returns:
        tuple: (recognized_text, accuracy_percentage)
    """
    text = read_image(image_path, model_path, charset_path)
    accuracy = calculate_accuracy(text, ground_truth)
    return text, accuracy
