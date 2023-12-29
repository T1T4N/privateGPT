"""Emlx parser.

Contains simple parser for mbox files.

"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

logger = logging.getLogger(__name__)


class EmlxReader(BaseReader):
    """Emlx parser.

    Extract messages from emlx files.
    Returns string including date, subject, sender, receiver and
    content for each message.

    """

    DEFAULT_MESSAGE_FORMAT: str = (
        "Date: {_date}\n"
        "From: {_from}\n"
        "To: {_to}\n"
        "Subject: {_subject}\n"
        "Content: {_content}"
    )

    def __init__(
        self,
        *args: Any,
        max_count: int = 0,
        message_format: str = DEFAULT_MESSAGE_FORMAT,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError(
                "`beautifulsoup4` package not found: `pip install beautifulsoup4`"
            )

        super().__init__(*args, **kwargs)
        self.max_count = max_count
        self.message_format = message_format

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file into string."""
        # Import required libraries
        import emlx
        from bs4 import BeautifulSoup

        results: List[str] = []
        # Load file using mailbox
        msg = emlx.read(file.absolute().as_posix())

        try:
            content = ""
            if msg.html is None:
                content = msg.text
            else:
                # Parse message HTML content and remove unneeded whitespace
                soup = BeautifulSoup(msg.html, "html.parser", from_encoding="utf-8")
                content = " ".join(soup.get_text().split())

            # Format message to include date, sender, receiver and subject
            msg_string = self.message_format.format(
                _date=msg.headers["Date"],
                _from=msg.headers["From"],
                _to=msg.headers["To"],
                _subject=msg.headers["Subject"],
                _content=content,
            )
            # Add message string to results
            results.append(msg_string)
        except Exception as e:
            logger.warning(f"Failed to parse message:\n{file}\n with exception {e}")

        return [Document(text=result, metadata=extra_info or {}) for result in results]

