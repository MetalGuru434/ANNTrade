def clean_text(text: str) -> str:
    """Lowercase text and remove non-alphabetic characters."""
    text = text.lower()
    # Keep only alphabetic characters and spaces
    cleaned = ''.join(ch for ch in text if ch.isalpha() or ch.isspace())
    return cleaned.strip()
