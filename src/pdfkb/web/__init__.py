"""Web interface for PDF Knowledgebase using FastAPI."""

from .middleware import setup_exception_handlers, setup_middleware
from .server import PDFKnowledgebaseWebServer
from .websocket_handlers import WebSocketEventHandler

__all__ = [
    "PDFKnowledgebaseWebServer",
    "setup_middleware",
    "setup_exception_handlers",
    "WebSocketEventHandler",
]
