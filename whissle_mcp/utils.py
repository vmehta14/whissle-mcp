"""Utility functions for the Whissle MCP server."""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime


def make_error(message: str) -> None:
    """Raise an error with a message."""
    raise ValueError(message)


def make_output_path(output_directory: Optional[str], base_path: Optional[str]) -> Path:
    """Create the output path for files."""
    if output_directory:
        return Path(output_directory).expanduser()
    elif base_path:
        return Path(base_path).expanduser()
    else:
        return Path.home() / "Desktop"


def make_output_file(
    prefix: str,
    base_name: str,
    output_path: Path,
    extension: str,
    full_id: bool = False,
) -> Path:
    """Create an output file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if full_id:
        file_name = f"{prefix}_{base_name}_{timestamp}.{extension}"
    else:
        file_name = f"{prefix}_{Path(base_name).stem}_{timestamp}.{extension}"
    return output_path / file_name


def handle_input_file(file_path: str) -> Path:
    """Handle input file path."""
    path = Path(file_path).expanduser()
    if not path.exists():
        make_error(f"File not found: {file_path}")
    return path 