import os
from typing import Dict, List, Tuple
import PyPDF2
import tabula
import pandas as pd
import pdfplumber
import google.generativeai as genai
from pdf2image import convert_from_path
from config import GEMINI_API_KEY

class PDFExtractor:
    def __init__(self, pdf_path: str):
        """Initialize the PDF extractor with a PDF file path."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.pdf_path = pdf_path
        
        # Configure Gemini for flowchart analysis
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found in environment. Image analysis will be skipped.")
            self.vision_model = None
            return
            
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.vision_model = genai.GenerativeModel('gemini-pro-vision')
            print("Successfully configured Gemini API")
        except Exception as e:
            print(f"Error configuring Gemini API: {str(e)}")
            self.vision_model = None

    def has_images(self, page) -> bool:
        """
        Efficiently check if a page contains images using multiple detection methods.
        Returns True if the page has any images, False otherwise.
        """
        try:
            found_images = False
            
            # Method 1: Check page.images
            images = page.images
            if images and len(images) > 0:
                print(f"Found {len(images)} images via page.images on page {page.page_number}")
                for i, img in enumerate(images):
                    print(f"  Image {i+1}: {img.get('width', 'N/A')}x{img.get('height', 'N/A')}")
                found_images = True

            # Method 2: Check for image-like objects in page resources
            if hasattr(page, '_objects'):
                for obj in page._objects.values():
                    if isinstance(obj, dict) and '/Subtype' in obj and obj['/Subtype'] == '/Image':
                        print(f"Found image object in resources on page {page.page_number}")
                        found_images = True
                    elif str(obj).lower().find('/image') >= 0:
                        print(f"Found image reference in objects on page {page.page_number}")
                        found_images = True
                        
            # Method 3: Look for XObject images
            if hasattr(page, 'resources') and page.resources:
                if '/XObject' in page.resources:
                    xobjects = page.resources['/XObject']
                    if isinstance(xobjects, dict):
                        for key, obj in xobjects.items():
                            if '/Subtype' in obj and obj['/Subtype'] == '/Image':
                                print(f"Found XObject image on page {page.page_number}")
                                found_images = True

            if not found_images:
                print(f"No images found on page {page.page_number}")
            return found_images
            
        except Exception as e:
            print(f"Error checking for images on page {page.page_number}: {str(e)}")
            return False

    def analyze_page_with_gemini(self, page_image) -> str:
        """
        Analyze a page image with Gemini to detect and extract flowchart content.
        """
        if not self.vision_model:
            print("Skipping Gemini analysis - API not configured")
            return ""
            
        try:
            # Convert PIL image to bytes for Gemini
            import io
            img_byte_arr = io.BytesIO()
            page_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_byte_arr.close()

            print("Sending image to Gemini for image detection...")
            # First check if this page contains a flowchart with a more detailed prompt
            detect_prompt = """Analyze this image and determine if it contains any images relevant to the text. 
            
            Answer with ONLY 'yes' or 'no'."""
            
            
            print("defhere2")
            try:
                response = self.vision_model.generate_content([detect_prompt, img_bytes])
                if not response:
                    # print("Error: Received empty response from Gemini")
                    return ""
                if not hasattr(response, 'text'):
                    # print(f"Error: Response missing text attribute. Response type: {type(response)}")
                    return ""
                result = response.text.strip().lower()
            except Exception as e:
                print(f"Error during Gemini API call: {type(e).__name__}: {str(e)}")
                return ""
            print("defhere")
            # print(f"Gemini flowchart detection response: {result}")
            
            if result != 'yes':
                # print("No clinical flowchart or algorithm detected")
                return ""

            # print("Clinical flowchart/algorithm detected, analyzing content...")
            # If flowchart detected, analyze it with a more specific prompt
            analyze_prompt = """Analyze this clinical image and extract its medical knowledge in a structured format.
            Focus on:
            If it is a flowchart, then:
            1. Title or purpose of the image
            2. Initial assessment or starting point
            3. Key decision points and their specific criteria (include exact numbers/thresholds)
            4. Treatment recommendations at each step
            5. Follow-up or monitoring instructions
            6. Any special notes or exceptions
            
            else:
                describe the image in detail to the best of your ability.
            Maintain all medical terminology, units, and thresholds exactly as shown in the image.
            Format the response as clear, structured text that preserves the logical flow of the algorithm."""

            response = self.vision_model.generate_content([analyze_prompt, img_bytes])
            return "\n\nClinical Algorithm Analysis:\n" + response.text + "\n"
            
        except Exception as e:
            print(f"Warning: Error analyzing page with Gemini: {str(e)}")
            return ""

    def extract_text(self) -> Dict[int, str]:
        """
        Extract text from all pages of the PDF using pdfplumber.
        Also detects pages with images and analyzes them with Gemini if they contain flowcharts.
        Returns a dictionary with page numbers as keys and extracted text as values.
        """
        text_content = {}
        pages_with_images = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                print(f"\nProcessing PDF with {len(pdf.pages)} pages")
                
                # First pass: Extract text and detect images
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        print(f"\nChecking page {page_num}...")
                        
                        # Extract text
                        text = page.extract_text(x_tolerance=1, y_tolerance=3)
                        text_content[page_num] = text.strip() if text else ""
                        print(f"  Extracted {len(text_content[page_num])} characters of text")
                        
                        # Check for images
                        if self.has_images(page):
                            pages_with_images.append(page_num)
                            
                    except Exception as e:
                        print(f"Error processing page {page_num}: {str(e)}")
                        text_content[page_num] = ""
                
                print(f"\nFound {len(pages_with_images)} pages with images: {pages_with_images}")
                
                # Second pass: Process pages with images using Gemini
                if pages_with_images:
                    print("\nConverting pages with images for analysis...")
                    images = convert_from_path(self.pdf_path, first_page=min(pages_with_images), last_page=max(pages_with_images))
                    print(f"Converted {len(images)} pages to images")
                    
                    for page_num, image in zip(pages_with_images, images):
                        print(f"\nAnalyzing page {page_num} with Gemini...")
                        flowchart_text = self.analyze_page_with_gemini(image)
                        if flowchart_text:
                            text_content[page_num] += flowchart_text
                            print(f"Added image content to page {page_num}")
                        else:
                            print(f"No image detected on page {page_num}")
                
                print(f"\nSuccessfully processed {len(text_content)} pages")
                        
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise
        
        return text_content

    def extract_tables(self) -> List[Tuple[int, List[pd.DataFrame]]]:
        """
        Extract tables from all pages of the PDF.
        Returns a list of tuples containing page number and list of tables found on that page.
        """
        try:
            # Extract all tables from the PDF
            tables = tabula.read_pdf(
                self.pdf_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=True
            )
            
            # Group tables by page
            page_tables = []
            current_page = 1
            current_tables = []
            
            for table in tables:
                if not table.empty:
                    current_tables.append(table)
                page_tables.append((current_page, current_tables))
                current_page += 1
                current_tables = []
            
            return page_tables
            
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

def process_pdfs(pdf_paths: List[str]) -> Dict[str, Dict]:
    """
    Process multiple PDF files and extract text, tables, and flowcharts.
    Returns a dictionary with PDF filenames as keys and their content as values.
    """
    results = {}
    
    for pdf_path in pdf_paths:
        try:
            print(f"\nProcessing: {pdf_path}")
            extractor = PDFExtractor(pdf_path)
            
            # Extract text (including flowcharts)
            text_content = extractor.extract_text()
            
            # Extract tables
            table_content = extractor.extract_tables()
            
            results[os.path.basename(pdf_path)] = {
                'text': text_content,
                'tables': table_content
            }
            
            print(f"Successfully processed: {pdf_path}")
            print(f"Pages with content: {sorted(text_content.keys())}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            results[os.path.basename(pdf_path)] = {
                'error': str(e)
            }
    return results

if __name__ == "__main__":
    # Example usage
    pdf_files = [
        "Clinical Practice Guidelines _ Hypertension in children and adolescents.pdf",
        "hypertension_adults.pdf"
    ]
    
    results = process_pdfs(pdf_files)
    
    # Print results
    for pdf_name, content in results.items():
        print(f"\nResults for {pdf_name}:")
        if 'error' in content:
            print(f"Error: {content['error']}")
            continue
            
        print("\nText content:")
        for page_num, text in content['text'].items():
            print(f"\nPage {page_num} ({len(text)} characters)") 