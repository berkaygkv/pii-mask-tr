"""PDF text extraction via Docling.

Born-digital PDFs keep their text layer; scanned pages are OCR'd by
EasyOCR with Turkish + English. Output is plain text matching the
surface form the BERTurk model trained on.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE,
        ),
        ocr_options=EasyOcrOptions(
            lang=["tr", "en"],
            confidence_threshold=0.5,
        ),
        accelerator_options=AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO,
        ),
        generate_page_images=False,
        generate_picture_images=False,
        document_timeout=180.0,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def parse_pdf_bytes(converter: DocumentConverter, data: bytes, filename: str) -> str:
    stream = DocumentStream(name=filename, stream=BytesIO(data))
    result = converter.convert(stream)
    return result.document.export_to_text()


def parse_pdf_path(converter: DocumentConverter, path: Path) -> str:
    return parse_pdf_bytes(converter, path.read_bytes(), path.name)
