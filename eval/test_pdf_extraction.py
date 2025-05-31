import sys
import os
from pathlib import Path
sys.path.append("..")

from pdf_extractor import PDFExtractor

def test_pdf_extraction():
    """Test PDF extraction specifically for the first few pages."""
    pdf_path = "../Clinical Practice Guidelines _ Hypertension in children and adolescents.pdf"
    
    print(f"\nTesting extraction for: {pdf_path}")
    print("="*80)
    
    try:
        extractor = PDFExtractor(pdf_path)
        text_content = extractor.extract_text()
        
        # Print detailed info for first few pages
        for page_num in sorted(text_content.keys())[:3]:  # First 3 pages
            print(f"\nPAGE {page_num}")
            print("-"*80)
            text = text_content[page_num]
            print(f"Length: {len(text)} characters")
            print(f"First 500 characters:\n{text[:500]}")
            print("\nLast 500 characters:\n{text[-500:] if len(text) > 500 else text}")
            print("-"*80)
            
        # Save raw output for inspection
        output_dir = Path("eval/pdf_extraction_debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        debug_file = output_dir / "raw_extraction.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            for page_num in sorted(text_content.keys()):
                f.write(f"\n{'='*80}\n")
                f.write(f"PAGE {page_num}\n")
                f.write(f"{'='*80}\n")
                f.write(text_content[page_num])
        
        print(f"\nFull extraction saved to: {debug_file}")
        
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_extraction() 