import os
from typing import Dict, List, Tuple
import PyPDF2
import tabula
import pandas as pd

class PDFExtractor:
    def __init__(self, pdf_path: str):
        """Initialize the PDF extractor with a PDF file path."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.pdf_path = pdf_path

    def extract_text(self) -> Dict[int, str]:
        """
        Extract text from all pages of the PDF.
        Returns a dictionary with page numbers as keys and extracted text as values.
        """
        text_content = {}
        
        with open(self.pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content[page_num + 1] = page.extract_text()
        
        return text_content

    def extract_tables(self) -> List[Tuple[int, List[pd.DataFrame]]]:
        """
        Extract tables from all pages of the PDF.
        Returns a list of tuples containing page number and list of tables found on that page.
        """
        # Extract all tables from the PDF
        tables = tabula.read_pdf(
            self.pdf_path,
            pages='all',
            multiple_tables=True
        )
        
        # Group tables by page
        page_tables = []
        current_page = 1
        current_tables = []
        
        for table in tables:
            if not table.empty:
                current_tables.append(table)
            # Note: This assumes tables are returned in page order
            page_tables.append((current_page, current_tables))
            current_page += 1
            current_tables = []
        
        return page_tables

def process_pdfs(pdf_paths: List[str]) -> Dict[str, Dict]:
    """
    Process multiple PDF files and extract both text and tables.
    Returns a dictionary with PDF filenames as keys and their content as values.
    """
    results = {}
    
    for pdf_path in pdf_paths:
        try:
            extractor = PDFExtractor(pdf_path)
            
            # Extract both text and tables
            text_content = extractor.extract_text()
            table_content = extractor.extract_tables()
            
            results[os.path.basename(pdf_path)] = {
                'text': text_content,
                'tables': table_content
            }
            
            print(f"Successfully processed: {pdf_path}")
            
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
            print(f"\nPage {page_num}:")
            print(text[:200] + "..." if len(text) > 200 else text)
            
        print("\nTables found:")
        for page_num, tables in content['tables']:
            print(f"\nPage {page_num}:")
            for i, table in enumerate(tables, 1):
                print(f"Table {i}:")
                print(table) 