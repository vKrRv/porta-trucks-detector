'''
Saudi License Plate Reader - OCR Module
Author: Hassan Alzourei

This module provides OCR functionality for reading Saudi license plates.
Can be imported and used in other Python files.
'''

# Install dependencies (uncomment if not already installed)
# !pip install langchain langchain-community paddlepaddle paddleocr opencv-python-headless
# !pip install easyocr opencv-python-headless

# Import necessary libraries
import cv2
import easyocr
import numpy as np
import json
import re
import os

# Map each English letter/digit to its corresponding Arabic letter/digit for Saudi license plates
SAUDI_MAPPING = {
    'A': 'ا', 'B': 'ب', 'J': 'ح', 'D': 'د', 'R': 'ر', 'S': 'س',
    'X': 'ص', 'T': 'ط', 'E': 'ع', 'G': 'ق', 'K': 'ك', 'L': 'ل',
    'Z': 'م', 'N': 'ن', 'H': 'ه', 'U': 'و', 'V': 'ى'
}

DIGIT_MAPPING = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
}

# Global reader instance (initialized when module is imported)
reader = None


def initialize_reader(use_gpu=True):
    """
    Initialize the EasyOCR reader.

    Args:
        use_gpu (bool): Whether to use GPU acceleration (default: True)

    Returns:
        easyocr.Reader: Initialized reader instance
    """
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=use_gpu)
    return reader


def enhance_image(img):
    """
    Augment the input image for better OCR results.

    Args:
        img (numpy.ndarray): Input image (BGR format)

    Returns:
        numpy.ndarray: Enhanced grayscale image
    """
    scale = 2
    img_large = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def read_license_plate(image_path):
    """
    Main function to read Saudi license plate from an image.

    This function processes a license plate image and extracts both English and Arabic
    representations of the plate numbers and letters.

    Args:
        image_path (str): Path to the license plate image file

    Returns:
        tuple: (response_dict, original_image, numbers_crop, letters_crop)
            - response_dict: JSON-formatted dictionary with plate information
            - original_image: Original input image (numpy array)
            - numbers_crop: Processed numbers region (numpy array)
            - letters_crop: Processed letters region (numpy array)

    Example:
        >>> result, img, nums, lets = process_plate_clean('plate.jpg')
        >>> print(result['data']['english']['full'])
        '1234 ABC'
    """
    # Initialize reader if not already done
    initialize_reader()

    if not os.path.exists(image_path):
        return {"error": "File not found"}, [None, None, None]

    original = cv2.imread(image_path)
    if original is None:
        return {"error": "Image corrupted"}, [None, None, None]

    h, w, _ = original.shape

    # 1. CUT HORIZONTAL: Keep Bottom 55%
    bottom_half = original[int(h * 0.45):, :]
    bh, bw, _ = bottom_half.shape
    mid_x = int(bw / 2) - int(bw * 0.025)
    gap = int(bw * 0.03)

    # 2. CUT VERTICAL (Left=Numbers, Right=Letters)
    start_nums = int(bw * 0.02)
    end_nums = mid_x - int(gap/2)
    crop_numbers = bottom_half[:, start_nums:end_nums]

    start_lets = mid_x + int(gap/2)
    end_lets = bw - int(bw * 0.02)
    crop_letters = bottom_half[:, start_lets:end_lets]

    # 3. Augment the images for processing
    clean_L_Numbers = enhance_image(crop_numbers)
    clean_R_Letters = enhance_image(crop_letters)

    # 4. Read the partial plates
    res_nums = reader.readtext(clean_L_Numbers, allowlist='0123456789')
    res_lets = reader.readtext(clean_R_Letters, allowlist='ABJDRSXTEGKLZNHUV')

    # 5. Prepare the output
    raw_nums = "".join([x[1] for x in res_nums])
    raw_lets = "".join([x[1] for x in res_lets]).upper()

    final_numbers = re.sub(r'[^0-9]', '', raw_nums)
    final_letters = re.sub(r'[^A-Z]', '', raw_lets)

    # Rule of 3
    if len(final_letters) > 3:
        final_letters = final_letters[-3:]

    # 6. ARABIC TRANSLATION & FORMATTING

    # --- A. NUMBERS (REVERSE ORDER) ---
    arb_nums_list = [DIGIT_MAPPING.get(c, c) for c in final_numbers]
    # arb_nums_list.reverse()
    arb_nums_str = " ".join(arb_nums_list)

    # --- B. LETTERS (REVERSE ORDER) ---
    arb_lets_list = [SAUDI_MAPPING.get(c, c) for c in final_letters]
    # arb_lets_list.reverse()
    arb_lets_str = " ".join(arb_lets_list)

    # 7. JSON RESPONSE (Clean - No Debug)
    response = {
        "status": "success",
        "data": {
            "english": {
                "letters": final_letters,
                "numbers": final_numbers,
                "full": f"{final_numbers} {final_letters}"
            },
            "arabic": {
                "letters": arb_lets_str,
                "numbers": arb_nums_str,
                "full": f"{arb_nums_str}   {arb_lets_str}"
            }
        }
    }

    return response, [original, clean_L_Numbers, clean_R_Letters]

"""
Example usage of the license_plate_reader module
This demonstrates how to import and use the OCR functions in your own code.
"""
'''
# Import the main function from the license plate reader module
from license_plate_reader import process_plate_clean, initialize_reader

import json
'''

def get_license_plate(image_path):
    """
    Example of using the license plate reader in your own application.
    """

    result, [original_img, numbers_crop, letters_crop] = read_license_plate(image_path)

    # Check if processing was successful
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Pretty print the full JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return json.dumps(result, indent=2, ensure_ascii=False)


# ... (rest of your code above remains the same)

# 7. JSON RESPONSE
    response = {
        "status": "success",
        "data": {
            "english": {
                "letters": final_letters,
                "numbers": final_numbers,
                "full": f"{final_numbers} {final_letters}"
            },
            "arabic": {
                "letters": arb_lets_str,
                "numbers": arb_nums_str,
                "full": f"{arb_nums_str}   {arb_lets_str}"
            }
        }
    }

    return response, [original, clean_L_Numbers, clean_R_Letters]

# ==========================================
# TEST CODE (Only runs if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    # Test path (Change this to a real image on your desktop for testing)
    test_image_path = "/home/ali/Desktop/test_plate.jpeg" 
    
    if os.path.exists(test_image_path):
        print(f"Testing on: {test_image_path}")
        result, _ = read_license_plate(test_image_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Test image not found. Please update 'test_image_path' to test.")