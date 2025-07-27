"""
Document Segmenter Module
This module provides functionality to segment documents into smaller parts for processing.
It supports various file types and uses a tokenizer to manage segment sizes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from abc import ABC, abstractmethod
import os
import re
import csv
import io
import json
from pathlib import Path
import numpy as np
import PyPDF2
import tiktoken
import docx
from bs4 import BeautifulSoup


@dataclass
class SegmenterConfig:
    """Configuration for the segmenter"""

    segment_size: int = 400  # Target tokens per segment
    segment_overlap: int = 50  # Overlap between segments
    min_segment_size: int = 20  # Minimum segment size (reduced for tiny files)
    max_segment_size: int = 600  # Maximum segment size

    # CSV-specific settings
    csv_max_rows_per_segment: int = 10  # Max CSV rows per segment
    csv_include_headers: bool = True  # Include headers in each segment

    # Supported file types - but we'll try to process ANY file
    supported_extensions: List[str] = field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".csv",
            ".pdf",
            ".json",
            ".docx",
            ".xml",
            ".html",
            ".py",
            ".js",
            ".ts",
            ".bib",
        ]
    )

    # File-agnostic settings
    max_file_size_mb: int = 50  # Skip files larger than this
    encoding_fallbacks: List[str] = field(
        default_factory=lambda: ["utf-8", "latin-1", "cp1252", "ascii"]
    )


@dataclass
class Segment:
    id: str
    text: str
    source_file: str = ""
    doc_type: str = ""
    segment_index: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure metadata is properly initialized"""
        if not isinstance(self.metadata, dict):
            self.metadata = {}


# --- FileType Segmenter Base and Subclasses ---
class BaseFileSegmenter(ABC):
    def __init__(self, config: SegmenterConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def read_content(self, file_path: Path) -> str:
        pass

    @classmethod
    @abstractmethod
    def file_type(cls) -> str:
        pass

    def segment(self, file_path: Path, source_file: str = "") -> List[Segment]:
        text = self.read_content(file_path)
        return DocumentSegmenter.split_into_segments_static(
            text,
            source_file=source_file,
            doc_type=self.file_type(),
            config=self.config,
            tokenizer=self.tokenizer,
        )


class PDFSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "pdf"

    def read_content(self, file_path: Path) -> str:
        text_parts = []
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception:
                    text_parts.append(f"[Page {page_num + 1}] - Text extraction failed")
        return "\n\n".join(text_parts) if text_parts else "No text extracted from PDF"


class DocxSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "docx"

    def read_content(self, file_path: Path) -> str:
        doc = docx.Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs) if paragraphs else "No text found in document"


class CSVSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "csv"

    def read_content(self, file_path: Path) -> str:
        text_parts = []
        for encoding in self.config.encoding_fallbacks:
            try:
                with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    reader = csv.DictReader(csvfile, delimiter=delimiter)
                    if reader.fieldnames:
                        text_parts.append(
                            f"CSV Columns: {', '.join(reader.fieldnames)}"
                        )
                        text_parts.append("")
                    for row_num, row in enumerate(reader):
                        if row_num >= 1000:
                            text_parts.append(f"... (truncated after 1000 rows)")
                            break
                        row_text = [
                            f"{k}: {v}" for k, v in row.items() if v and str(v).strip()
                        ]
                        if row_text:
                            text_parts.append(
                                f"Row {row_num + 1}: " + " | ".join(row_text)
                            )
                    return "\n".join(text_parts)
            except Exception:
                continue
        return "Failed to parse CSV file"


class JSONSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "json"

    def read_content(self, file_path: Path) -> str:
        for encoding in self.config.encoding_fallbacks:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    data = json.load(f)
                return self._json_to_text(data)
            except Exception:
                continue
        return "Failed to parse JSON file"

    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        lines = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, prefix + "  "))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, prefix + "  "))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
        else:
            lines.append(f"{prefix}{data}")
        return "\n".join(lines)


class HTMLSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "html"

    def read_content(self, file_path: Path) -> str:
        for encoding in self.config.encoding_fallbacks:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                soup = BeautifulSoup(content, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                return "\n".join(chunk for chunk in chunks if chunk)
            except Exception:
                continue
        return "Failed to extract HTML content"


class BibTeXSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "bib"

    def read_content(self, file_path: Path) -> str:
        entries = []
        entry = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("@"):
                    if entry:
                        entries.append("".join(entry))
                        entry = []
                entry.append(line)
            if entry:
                entries.append("".join(entry))
        return "\n\n".join(entries)


class TextSegmenter(BaseFileSegmenter):
    @classmethod
    def file_type(cls) -> str:
        return "text"

    def read_content(self, file_path: Path) -> str:
        for encoding in self.config.encoding_fallbacks:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                    break
            except Exception:
                continue
        if text is None:
            with open(file_path, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
        # Ensure at least one segment for non-empty file
        if text and text.strip():
            return text
        return ""


class DocumentSegmenter:
    def __init__(self, config=None, tokenizer=None):
        self.config = config or SegmenterConfig()
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self._num_segments = 0
        self._num_embeddings = 0
        self._total_segment_tokens = 0

    def process_directory(
        self, dir_path: Union[str, Path], embed_fn=None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Walk through a directory, process and index all supported files.
        Returns a dict mapping file paths to their segments and embeddings.
        """
        import logging

        logger = logging.getLogger("document_segmenter")
        dir_path = Path(dir_path)
        results = {}
        self._num_segments = 0
        self._num_embeddings = 0
        self._total_segment_tokens = 0
        for root, _, files in os.walk(dir_path):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in self.config.supported_extensions:
                    logger.info(f"Processing file: {fpath}")
                    segments = self.process_file(fpath)
                    if embed_fn:
                        embeddings = self.prepare_embeddings(segments, embed_fn)
                    else:
                        embeddings = None
                    logger.info(
                        f"File: {fpath} | Segments: {len(segments)} | Embeddings: {embeddings.shape if embeddings is not None else 'None'}"
                    )
                    results[str(fpath)] = {
                        "segments": segments,
                        "embeddings": embeddings,
                    }
                    self._num_segments += len(segments)
                    if embeddings is not None:
                        self._num_embeddings += len(embeddings)
                    self._total_segment_tokens += sum(
                        len(self.tokenizer.encode(seg.text)) for seg in segments
                    )
        return results

    @property
    def compute_stats(self):
        return {
            "num_segments": self._num_segments,
            "num_embeddings": self._num_embeddings,
            "total_segment_tokens": self._total_segment_tokens,
        }

    @staticmethod
    def split_into_segments_static(
        text: str, source_file: str, doc_type: str, config: SegmenterConfig, tokenizer
    ) -> List[Segment]:
        if not text or not text.strip():
            return []
        sentences = DocumentSegmenter._split_into_sentences_static(text)
        segments = []
        current_segment = ""
        current_tokens = 0
        segment_index = 0
        char_start = 0
        for sentence in sentences:
            sentence_tokens = (
                len(tokenizer.encode(sentence)) if tokenizer else len(sentence) // 4
            )
            if (
                current_tokens + sentence_tokens > config.segment_size
                and current_tokens >= config.min_segment_size
            ):
                if current_segment.strip():
                    segment = Segment(
                        id=f"{source_file}_{segment_index}",
                        text=current_segment.strip(),
                        source_file=source_file,
                        doc_type=doc_type,
                        segment_index=segment_index,
                        char_start=char_start,
                        char_end=char_start + len(current_segment),
                        metadata={
                            "token_count": current_tokens,
                            "sentence_count": len(
                                [s for s in current_segment.split(".") if s.strip()]
                            ),
                        },
                    )
                    segments.append(segment)
                    if config.segment_overlap > 0:
                        overlap_text = DocumentSegmenter._get_overlap_text_static(
                            current_segment, config.segment_overlap, tokenizer
                        )
                        char_start = (
                            char_start + len(current_segment) - len(overlap_text)
                        )
                        current_segment = overlap_text + " " + sentence
                        current_tokens = (
                            len(tokenizer.encode(current_segment))
                            if tokenizer
                            else len(current_segment) // 4
                        )
                    else:
                        char_start = char_start + len(current_segment)
                        current_segment = sentence
                        current_tokens = sentence_tokens
                    segment_index += 1
            else:
                current_segment += " " + sentence if current_segment else sentence
                current_tokens += sentence_tokens
        # Patch: If file is smaller than min_segment_size, pad and create a segment
        if current_segment.strip():
            seg_text = current_segment.strip()
            if len(seg_text) < config.min_segment_size:
                seg_text = seg_text.ljust(config.min_segment_size, " ")
            segment = Segment(
                id=f"{source_file}_{segment_index}",
                text=seg_text,
                source_file=source_file,
                doc_type=doc_type,
                segment_index=segment_index,
                char_start=char_start,
                char_end=char_start + len(seg_text),
                metadata={
                    "token_count": (
                        len(tokenizer.encode(seg_text))
                        if tokenizer
                        else len(seg_text) // 4
                    ),
                    "sentence_count": len(
                        [s for s in seg_text.split(".") if s.strip()]
                    ),
                },
            )
            segments.append(segment)
        return segments

    @staticmethod
    def _split_into_sentences_static(text: str) -> List[str]:
        sentence_endings = r"[.!?]+[\s]*"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    @staticmethod
    def _get_overlap_text_static(text: str, overlap_tokens: int, tokenizer) -> str:
        if not text or overlap_tokens <= 0:
            return ""
        overlap_chars = overlap_tokens * 4
        if tokenizer:
            tokens = tokenizer.encode(text)
            if len(tokens) <= overlap_tokens:
                return text
            decoded = tokenizer.decode(tokens[-overlap_tokens:])
            return decoded
        if len(text) <= overlap_chars:
            return text
        overlap_start = len(text) - overlap_chars
        remaining_text = text[overlap_start:]
        sentence_break = re.search(r"[.!?]\s+", remaining_text)
        if sentence_break:
            return remaining_text[sentence_break.end() :]
        words = remaining_text.split()
        return " ".join(words[1:]) if len(words) > 1 else remaining_text

    def detect_file_type(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        ext_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".csv": "csv",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".htm": "html",
            ".md": "markdown",
            ".txt": "text",
            ".py": "code",
            ".js": "code",
            ".ts": "code",
            ".java": "code",
            ".cpp": "code",
            ".c": "code",
            ".bib": "bib",
        }
        return ext_map.get(extension, "text")

    def process_file(self, file_path: Union[str, Path]) -> List[Segment]:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return [
                Segment(
                    id=f"error_{file_path.name}",
                    text=f"File not found: {file_path}",
                    source_file=str(file_path),
                    doc_type="error",
                )
            ]
        file_type = self.detect_file_type(file_path)
        segmenter_cls = self.segmenter_map.get(file_type, TextSegmenter)
        segmenter = segmenter_cls(self.config, self.tokenizer)
        try:
            return segmenter.segment(file_path, source_file=str(file_path))
        except Exception as e:
            return [
                Segment(
                    id=f"error_{file_path.name}",
                    text=f"Processing failed: {str(e)}",
                    source_file=str(file_path),
                    doc_type="error",
                    segment_index=0,
                    char_start=0,
                    char_end=0,
                    metadata={"error": str(e)},
                )
            ]

    def __init__(self, config: Optional[SegmenterConfig] = None):
        # Use a smaller segment size for more granular segments
        if config is None:
            config = SegmenterConfig(
                segment_size=100, min_segment_size=40, segment_overlap=20
            )
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.segmenter_map = {
            "pdf": PDFSegmenter,
            "docx": DocxSegmenter,
            "csv": CSVSegmenter,
            "json": JSONSegmenter,
            "html": HTMLSegmenter,
            "bib": BibTeXSegmenter,
            "text": TextSegmenter,
            "markdown": TextSegmenter,
            "code": TextSegmenter,
            "xml": TextSegmenter,
        }

    def prepare_embeddings(self, segments: List[Segment], embed_fn) -> np.ndarray:
        """
        Given a list of segments and an embedding function, return a numpy array of embeddings.
        embed_fn: Callable[[str], np.ndarray] or Callable[[List[str]], np.ndarray]
        """
        texts = [seg.text for seg in segments]
        # Try batch embedding first, fallback to single
        try:
            embeddings = embed_fn(texts)
        except Exception:
            embeddings = np.array([embed_fn(t) for t in texts])
        return np.array(embeddings)

    def segment_and_embed(
        self, file_path: str, embed_fn
    ) -> Tuple[List[Segment], np.ndarray]:
        """
        Segment a file and prepare embeddings for each segment using embed_fn.
        Returns (segments, embeddings)
        """
        segments = self.process_file(file_path)
        embeddings = self.prepare_embeddings(segments, embed_fn)
        return segments, embeddings
