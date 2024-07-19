import pytest
import time
from src.document_table_detection import DocumentTableDetector

class TestTableDetection:
    @pytest.fixture(scope="class")
    def table_detector(self):
        """Fixture to initialize the TableDetector only once per test session."""
        return DocumentTableDetector()

    
    def test_invoice_table_detection(self, table_detector):
        """
        Test to ensure that table detection on an invoice image works correctly
        and that detected tables are labeled appropriately. Also, display the image with detected tables.
        """
        result = table_detector.predict("unit_tests/images/invoice.jpg")
        assert len(result) > 0, "No tables detected in the invoice image."
        assert any(detection["label"] == "table" for detection in result), "Detected objects are not labeled as 'table'."

        # Draw and show the result image with scores
        img = table_detector.draw_boxes(result, "unit_tests/images/invoice.jpg", show_scores=True)
        img.show()
        time.sleep(1)

    
    def test_no_tables_document(self, table_detector):
        """
        Test to ensure that no tables are detected in an image that does not contain tables.
        """
        result = table_detector.predict("unit_tests/images/no_table.jpg")
        assert len(result) == 0, "False positives detected in an image without tables."

    
    def test_bank_document_table_detection(self, table_detector):
        """
        Test to ensure that tables are detected correctly in a bank document.
        Also, display the image with detected tables.
        """
        result = table_detector.predict("unit_tests/images/bank_document.png")
        assert len(result) > 0, "No tables detected in the bank document."
        assert any(detection["label"] == "table" for detection in result), "Detected objects are not labeled as 'table'."

        # Draw and show the result image with scores
        img = table_detector.draw_boxes(result, "unit_tests/images/bank_document.png", show_scores=True)
        img.show()
        time.sleep(1)

    
    def test_detection_in_rotated_images(self, table_detector):
        """Test that tables are detected correctly even in rotated images."""
        result = table_detector.predict("unit_tests/images/rotated_table.png")
        assert len(result) > 0, "No tables detected in the rotated image."
        assert any(detection["label"] == "table" for detection in result), "No detection labeled as 'table'."

        # Optionally show the image with detected tables and scores for manual verification
        img = table_detector.draw_boxes(result, "unit_tests/images/rotated_table.png", show_scores=True)
        img.show()
        time.sleep(1)

    
    def test_multiple_tables_detection(self, table_detector):
        """
        Test that the table detector identifies multiple tables in a single image and
        verifies that at least two tables are correctly labeled as 'table'.
        """
        result = table_detector.predict("unit_tests/images/multiple_tables.png")
        table_detections = [detection for detection in result if detection["label"] == "table"]
        assert len(table_detections) > 1, "The detector should identify more than one table in the image."

        # Draw and show the image with scores
        img = table_detector.draw_boxes(result, "unit_tests/images/multiple_tables.png", show_scores=True)
        img.show()
        time.sleep(1)

    
    def test_invalid_file_format(self, table_detector):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            table_detector.predict("unit_tests/images/unsupported_format.gif")
