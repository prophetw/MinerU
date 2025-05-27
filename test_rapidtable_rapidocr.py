#!/usr/bin/env python3
"""
Test file for RapidTable with RapidOCR integration
"""
import os
import sys
from pathlib import Path
from PIL import Image
from lxml import etree

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from magic_pdf.model.sub_modules.model_init import rapidocr_model_init
from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel


def test_rapidtable_with_rapidocr():
    """Test RapidTable with RapidOCR engine"""
    print("Testing RapidTable with RapidOCR...")
    
    # Initialize RapidOCR engine
    try:
        ocr_engine = rapidocr_model_init()
        print("✓ RapidOCR engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RapidOCR: {e}")
        return False
    
    # Initialize RapidTable model
    try:
        table_model = RapidTableModel(ocr_engine, 'slanet_plus')
        print("✓ RapidTable model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RapidTable: {e}")
        return False
    
    # Test with a sample image (use existing test image if available)
    test_image_path = project_root / "tests" / "unittest" / "test_table" / "assets" / "table.jpg"
    if not test_image_path.exists():
        # Try another common location
        test_image_path = project_root / "demo" / "pdfs" / "table.pdf"
        if not test_image_path.exists():
            print(f"✗ Test image not found at {test_image_path}")
            print("Creating a simple test image...")
            # Create a simple test image with text that looks like a table
            import numpy as np
            import cv2
            
            # Create a white image
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            # Add some table-like text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Column1  Column2  Column3', (50, 100), font, 0.8, (0, 0, 0), 2)
            cv2.putText(img, 'Row1Val1  Row1Val2  Row1Val3', (50, 150), font, 0.6, (0, 0, 0), 1)
            cv2.putText(img, 'Row2Val1  Row2Val2  Row2Val3', (50, 200), font, 0.6, (0, 0, 0), 1)
            cv2.putText(img, 'Row3Val1  Row3Val2  Row3Val3', (50, 250), font, 0.6, (0, 0, 0), 1)
            
            # Draw some table borders
            cv2.rectangle(img, (40, 80), (560, 120), (0, 0, 0), 2)  # Header
            cv2.rectangle(img, (40, 120), (560, 270), (0, 0, 0), 1)  # Body
            cv2.line(img, (200, 80), (200, 270), (0, 0, 0), 1)  # Vertical line 1
            cv2.line(img, (400, 80), (400, 270), (0, 0, 0), 1)  # Vertical line 2
            
            # Convert to PIL Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_image = Image.fromarray(img_rgb)
    else:
        try:
            test_image = Image.open(test_image_path)
            print(f"✓ Loaded test image from {test_image_path}")
        except Exception as e:
            print(f"✗ Failed to load test image: {e}")
            return False
    
    # Test table prediction
    try:
        print("Running table prediction...")
        html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(test_image)
        
        if html_code is not None:
            print(f"✓ Table prediction successful (elapsed: {elapse:.3f}s)")
            print(f"✓ Generated HTML: {len(html_code)} characters")
            print(f"✓ Cell bboxes: {len(table_cell_bboxes) if table_cell_bboxes else 0}")
            print(f"✓ Logic points: {len(logic_points) if logic_points else 0}")
            
            # Validate HTML structure
            try:
                parser = etree.HTMLParser()
                tree = etree.fromstring(html_code, parser)
                
                # Check for basic table elements
                table_elem = tree.find('.//table')
                tr_elems = tree.findall('.//tr')
                td_elems = tree.findall('.//td')
                
                if table_elem is not None:
                    print("✓ HTML contains <table> element")
                else:
                    print("✗ HTML missing <table> element")
                    
                print(f"✓ HTML contains {len(tr_elems)} <tr> elements")
                print(f"✓ HTML contains {len(td_elems)} <td> elements")
                
                print("\nGenerated HTML:")
                print("-" * 50)
                print(html_code)
                print("-" * 50)
                
                return True
                
            except Exception as e:
                print(f"✗ HTML validation failed: {e}")
                print(f"Generated HTML: {html_code}")
                return False
                
        else:
            print("✗ Table prediction returned None")
            return False
            
    except Exception as e:
        print(f"✗ Table prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_interface_detection():
    """Test that RapidOCR interface is correctly detected"""
    print("\nTesting OCR interface detection...")
    
    try:
        # Test RapidOCR
        rapid_ocr = rapidocr_model_init()
        has_ocr_method = hasattr(rapid_ocr, 'ocr')
        print(f"RapidOCR has 'ocr' method: {has_ocr_method}")
        print(f"RapidOCR methods: {[m for m in dir(rapid_ocr) if not m.startswith('_')]}")
        
        # Test with a simple image
        import numpy as np
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        if not has_ocr_method:
            # Should use RapidOCR interface
            det_result = rapid_ocr.auto_text_det(test_img)
            print(f"✓ RapidOCR detection result type: {type(det_result)}")
            
            full_result = rapid_ocr(test_img)
            print(f"✓ RapidOCR full result type: {type(full_result)}")
            if full_result[0]:
                print(f"✓ RapidOCR result format: {type(full_result[0][0]) if full_result[0] else 'Empty'}")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RapidTable + RapidOCR Integration Test")
    print("=" * 60)
    
    # Test OCR interface detection
    interface_ok = test_ocr_interface_detection()
    
    # Test RapidTable with RapidOCR
    table_ok = test_rapidtable_with_rapidocr()
    
    print("\n" + "=" * 60)
    if interface_ok and table_ok:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
