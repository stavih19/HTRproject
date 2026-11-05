import json
import tempfile
import unittest
from pathlib import Path

from recognition.run_recognition import write_recognized_rows, write_row_crop_results


class RecognitionOutputTests(unittest.TestCase):
    def test_writes_unicode_hebrew_rows_and_metadata(self):
        alto = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout><Page><PrintSpace><TextBlock>
    <TextLine ID="row_01"><String CONTENT="שָׁלוֹם" WC="0.8" HPOS="10"/></TextLine>
    <TextLine ID="row_02"><String CONTENT="עולם" WC="0.6"/></TextLine>
    <TextLine ID="row_03"><String CONTENT="   " WC="0.2"/></TextLine>
  </TextBlock></PrintSpace></Page></Layout>
</alto>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            alto_path = directory / "ocr.xml"
            alto_path.write_text(alto, encoding="utf-8")

            write_recognized_rows(alto_path, directory, {"model_id": "test"})

            text = (directory / "recognized_rows.txt").read_text(encoding="utf-8")
            self.assertEqual(text, "שָׁלוֹם\nעולם\n\n")
            data = json.loads(
                (directory / "recognized_rows.json").read_text(encoding="utf-8")
            )
            self.assertEqual(data["row_count"], 3)
            self.assertEqual(data["rows"][0]["line_id"], "row_01")
            self.assertEqual(data["rows"][0]["text"], "שָׁלוֹם")
            self.assertEqual(data["rows"][0]["confidence"], 0.8)
            self.assertEqual(data["rows"][0]["words"][0]["bbox"]["x"], 10)
            self.assertEqual(data["rows"][2]["text"], "")
            self.assertEqual(
                (directory / "transcription.txt").read_text(encoding="utf-8"),
                text,
            )

    def test_row_crop_output_pairs_crop_name_with_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            crop_1 = directory / "row_000.png"
            crop_2 = directory / "row_001.png"
            text_1 = directory / "row_000.txt"
            text_2 = directory / "row_001.txt"
            crop_1.touch()
            crop_2.touch()
            text_1.write_text("שלום", encoding="utf-8")
            text_2.write_text("עולם", encoding="utf-8")

            write_row_crop_results(
                [text_1, text_2],
                [crop_1, crop_2],
                ["line_1", "line_2"],
                directory,
                {"model_id": "test"},
            )

            self.assertEqual(
                (directory / "recognized_rows.txt").read_text(encoding="utf-8"),
                "row_000.png\nשלום\nrow_001.png\nעולם\n",
            )
            self.assertEqual(
                (directory / "transcription.txt").read_text(encoding="utf-8"),
                "שלום\nעולם\n",
            )
            report = (directory / "recognition_report.html").read_text(
                encoding="utf-8"
            )
            self.assertIn('<html lang="he" dir="rtl">', report)
            self.assertIn("row_000.png", report)
            self.assertIn("שלום", report)
            self.assertIn('class="prediction" dir="rtl" lang="he"', report)


if __name__ == "__main__":
    unittest.main()
