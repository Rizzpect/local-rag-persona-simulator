"""Text processing utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())


def sanitize_for_filename(text: str) -> str:
    """
    Sanitize text for use as a filename.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", text)
    sanitized = re.sub(r"_{2,}", "_", sanitized)
    return sanitized.strip("_")


def highlight_keywords(text: str, keywords: list[str]) -> str:
    """
    Highlight keywords in text.

    Args:
        text: Input text
        keywords: Keywords to highlight

    Returns:
        Text with highlighted keywords
    """
    result = text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(f"**{keyword}**", result)
    return result


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text.

    Args:
        text: Input text

    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.

    Args:
        text: Input text

    Returns:
        Text without URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.sub(url_pattern, "", text)


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks
