import os
import urllib.request

def load_text(source):
    """
    Loads text data from a local file path or a URL.
    Args:
        source (str): Path to a local file or a URL (http/https).
    Returns:
        str: The loaded text data.
    """
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as file:
            text_data = file.read()
    elif source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            text_data = response.read().decode('utf-8')
    else:
        raise ValueError("Source must be a valid file path or URL.")
    return text_data