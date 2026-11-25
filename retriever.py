from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "do", "does",
    "how", "what", "when", "where", "who",
    "to", "of", "in", "on", "for", "and", "get",
}

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
DEFAULT_RESULT_COUNT = 4
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float
    chunk_index: int


def _read_pdf(path: Path) -> str:
    try:
        with path.open("rb") as file:
            is_pdf = file.read(5) == b"%PDF-"
    except OSError as error:
        raise RuntimeError(f"Could not read {path.name}: {error}") from error

    if not is_pdf:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                f"{path.name} is not a valid PDF or UTF-8 text file: {error}"
            ) from error
        return text

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise RuntimeError(f"{path.name} is encrypted and cannot be indexed.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (OSError, PdfReadError, ValueError) as error:
        raise RuntimeError(f"Could not extract text from {path.name}: {error}") from error

    if not text.strip():
        raise RuntimeError(
            f"No extractable text found in {path.name}; scanned PDFs require OCR."
        )
    return text


def load_documents() -> dict[str, str]:
    """Load UTF-8 text files and text-extractable PDFs from the knowledge directory."""
    if not KNOWLEDGE_DIR.is_dir():
        return {}

    files = sorted(
        (*KNOWLEDGE_DIR.glob("*.txt"), *KNOWLEDGE_DIR.glob("*.pdf")),
        key=lambda path: path.name.casefold(),
    )
    documents = {}
    for path in files:
        if path.suffix.casefold() == ".pdf":
            documents[path.name] = _read_pdf(path)
        else:
            try:
                documents[path.name] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise RuntimeError(f"Could not read {path.name}: {error}") from error
    return documents


def read_knowledge_document(filename: str) -> str:
    """Read a supported file from the knowledge directory only."""
    filename = filename.strip()
    if not filename:
        raise ValueError("A knowledge filename is required.")

    supplied_path = Path(filename)
    knowledge_directory = KNOWLEDGE_DIR.resolve()
    try:
        document_path = (
            supplied_path
            if supplied_path.is_absolute()
            else knowledge_directory / supplied_path
        ).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Could not find knowledge file {filename!r}: {error}") from error

    try:
        document_path.relative_to(knowledge_directory)
    except ValueError as error:
        raise ValueError("The knowledge file path must stay inside knowledge/.") from error

    if not document_path.is_file():
        raise ValueError(f"{filename!r} is not a file.")
    if document_path.suffix.casefold() == ".pdf":
        return _read_pdf(document_path)
    if document_path.suffix.casefold() != ".txt":
        raise ValueError("Only .txt and .pdf knowledge files are supported.")

    try:
        return document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Could not read knowledge file {filename!r}: {error}") from error


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in STOP_WORDS
    ]


def _chunk_document(text: str) -> list[tuple[str, tuple[str, ...], int]]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk_text = " ".join(words[start : start + CHUNK_SIZE])
        tokens = tuple(_tokenize(chunk_text))
        if tokens:
            chunks.append((chunk_text, tokens, chunk_index))
        if start + CHUNK_SIZE >= len(words):
            break
    return chunks


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[RetrievedChunk, ...]:
    chunks = []
    for source, text in load_documents().items():
        for chunk_text, tokens, chunk_index in _chunk_document(text):
            chunks.append(
                RetrievedChunk(
                    source=source,
                    text=chunk_text,
                    score=0.0,
                    chunk_index=chunk_index,
                )
            )
    return tuple(chunks)


def retrieve_documents(
    query: str, result_count: int = DEFAULT_RESULT_COUNT
) -> list[RetrievedChunk]:
    """Rank knowledge chunks against a query using BM25 keyword relevance."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty search query is required.")
    if result_count < 1:
        raise ValueError("result_count must be at least 1.")

    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    chunks = _load_chunks()
    if not chunks:
        return []

    filename_matches = {}
    for chunk in chunks:
        filename_terms = set(_tokenize(Path(chunk.source).stem))
        match_count = len(query_terms & filename_terms)
        if match_count:
            filename_matches[chunk.source] = max(
                filename_matches.get(chunk.source, 0), match_count
            )

    if filename_matches:
        matching_chunks = []
        for source in sorted(
            filename_matches,
            key=lambda name: (-filename_matches[name], name.casefold()),
        ):
            source_chunks = sorted(
                (chunk for chunk in chunks if chunk.source == source),
                key=lambda chunk: chunk.chunk_index,
            )
            matching_chunks.extend(
                RetrievedChunk(
                    source=chunk.source,
                    text=chunk.text,
                    score=float(filename_matches[source]),
                    chunk_index=chunk.chunk_index,
                )
                for chunk in source_chunks
            )
        return matching_chunks[:result_count]

    tokenized_chunks = [_tokenize(chunk.text) for chunk in chunks]
    document_frequencies = Counter(
        term for tokens in tokenized_chunks for term in set(tokens)
    )
    average_length = sum(map(len, tokenized_chunks)) / len(tokenized_chunks)

    k1 = 1.5
    b = 0.75
    scored_chunks = []
    for chunk, tokens in zip(chunks, tokenized_chunks):
        term_frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequencies[term]
            if not frequency:
                continue
            doc_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (len(chunks) - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            length_normalization = k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / (frequency + length_normalization)
            )
        if score > 0:
            scored_chunks.append(
                RetrievedChunk(
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                    chunk_index=chunk.chunk_index,
                )
            )

    ranked_chunks = sorted(
        scored_chunks,
        key=lambda chunk: (-chunk.score, chunk.source.casefold(), chunk.chunk_index),
    )
    results = []
    for chunk in ranked_chunks:
        overlaps_existing = any(
            result.source == chunk.source
            and abs(result.chunk_index - chunk.chunk_index) == 1
            for result in results
        )
        if not overlaps_existing:
            results.append(chunk)
            if len(results) == result_count:
                break
    return results


def retrieve_context(query: str) -> str:
    """Return ranked knowledge excerpts with source filenames, or an empty string."""
    results = retrieve_documents(query)
    return "\n\n".join(
        f"[Source: {result.source}]\n{result.text}" for result in results
    )


def search_knowledge(query: str) -> str:
    """Return ranked knowledge excerpts with source filenames for the assistant."""
    context = retrieve_context(query)
    if not context:
        return "No relevant passages were found in the local knowledge files."
    return context
