import logging
from pathlib import Path

from llama_index import Document
from llama_index.readers import JSONReader, StringIterableReader
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

logger = logging.getLogger(__name__)

# Patching the default file reader to support other file types
FILE_READER_CLS = DEFAULT_FILE_READER_CLS.copy()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)

UNSUPPORTED_FILE_FORMATS = {
    ".epub",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".png"
}


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        if extension in UNSUPPORTED_FILE_FORMATS:
            return []

        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            try:
                return string_reader.load_data([file_data.read_text()])
            except:
                logging.exception(f"String reader threw an exception when processing: {file_name}")
                return []

        logger.debug("Specific reader found for extension=%s", extension)
        try:
            return reader_cls().load_data(file_data)
        except:
            # ex. FileNotDecryptedError("File has not been decrypted")
            logging.exception(f"Reader class threw an exception when processing: {file_name}")
            return []

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
