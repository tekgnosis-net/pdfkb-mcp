"""WebSocket connection and event management."""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from ..models.web_models import WebsocketEventType, WebsocketMessage

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket connection."""

    def __init__(self, websocket: WebSocket, client_id: str):
        """Initialize WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
        """
        self.websocket = websocket
        self.client_id = client_id
        self.is_active = True

    async def send_message(self, message: WebsocketMessage) -> bool:
        """Send message to this connection.

        Args:
            message: Message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            if not self.is_active:
                return False

            # Convert message to JSON
            message_data = message.model_dump()
            await self.websocket.send_text(json.dumps(message_data, default=str))
            return True

        except Exception as e:
            logger.error(f"Failed to send message to client {self.client_id}: {e}")
            self.is_active = False
            return False

    async def close(self) -> None:
        """Close the WebSocket connection."""
        try:
            self.is_active = False
            await self.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket for client {self.client_id}: {e}")


class WebSocketManager:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.connections: Dict[str, WebSocketConnection] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance

        Returns:
            Generated client ID
        """
        client_id = str(uuid.uuid4())

        try:
            await websocket.accept()
            connection = WebSocketConnection(websocket, client_id)

            async with self._lock:
                self.connections[client_id] = connection

            logger.info(f"WebSocket client connected: {client_id}")

            # Send welcome message
            welcome_message = WebsocketMessage(
                event_type=WebsocketEventType.SYSTEM_STATUS_CHANGED,
                data={"status": "connected", "client_id": client_id},
                message="Connected to PDF KB server",
            )
            await connection.send_message(welcome_message)

            return client_id

        except Exception as e:
            logger.error(f"Failed to connect WebSocket client: {e}")
            raise

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a WebSocket client.

        Args:
            client_id: Client ID to disconnect
        """
        async with self._lock:
            if client_id in self.connections:
                connection = self.connections[client_id]
                await connection.close()
                del self.connections[client_id]
                logger.info(f"WebSocket client disconnected: {client_id}")

    async def broadcast(
        self, event_type: WebsocketEventType, data: Dict[str, Any], message: Optional[str] = None
    ) -> int:
        """Broadcast message to all connected clients.

        Args:
            event_type: Type of event
            data: Event data
            message: Optional human-readable message

        Returns:
            Number of clients that received the message
        """
        if not self.connections:
            return 0

        websocket_message = WebsocketMessage(
            event_type=event_type,
            data=data,
            message=message,
        )

        sent_count = 0
        failed_connections = []

        async with self._lock:
            for client_id, connection in self.connections.items():
                if await connection.send_message(websocket_message):
                    sent_count += 1
                else:
                    failed_connections.append(client_id)

            # Clean up failed connections
            for client_id in failed_connections:
                if client_id in self.connections:
                    del self.connections[client_id]
                    logger.warning(f"Removed failed WebSocket connection: {client_id}")

        logger.debug(f"Broadcasted {event_type} to {sent_count} clients")
        return sent_count

    async def send_to_client(
        self,
        client_id: str,
        event_type: WebsocketEventType,
        data: Dict[str, Any],
        message: Optional[str] = None,
    ) -> bool:
        """Send message to a specific client.

        Args:
            client_id: Target client ID
            event_type: Type of event
            data: Event data
            message: Optional human-readable message

        Returns:
            True if message was sent successfully, False otherwise
        """
        async with self._lock:
            if client_id not in self.connections:
                logger.warning(f"Client not found: {client_id}")
                return False

            connection = self.connections[client_id]
            websocket_message = WebsocketMessage(
                event_type=event_type,
                data=data,
                message=message,
                client_id=client_id,
            )

            if await connection.send_message(websocket_message):
                return True
            else:
                # Clean up failed connection
                del self.connections[client_id]
                return False

    async def handle_client_message(self, client_id: str, message_data: Dict[str, Any]) -> None:
        """Handle message received from client.

        Args:
            client_id: Client ID that sent the message
            message_data: Message data from client
        """
        try:
            # For now, we mainly handle ping/pong and subscription requests
            message_type = message_data.get("type")

            if message_type == "ping":
                await self.send_to_client(
                    client_id,
                    WebsocketEventType.SYSTEM_STATUS_CHANGED,
                    {"type": "pong"},
                    "Pong response",
                )
            elif message_type == "subscribe":
                # Handle subscription to specific event types
                event_types = message_data.get("event_types", [])
                logger.info(f"Client {client_id} subscribed to events: {event_types}")
                # For now, all clients receive all events
                # In the future, we could implement selective event filtering

            else:
                logger.debug(f"Unhandled message type from client {client_id}: {message_type}")

        except Exception as e:
            logger.error(f"Error handling client message from {client_id}: {e}")

    async def get_connection_count(self) -> int:
        """Get the number of active connections.

        Returns:
            Number of active WebSocket connections
        """
        async with self._lock:
            return len(self.connections)

    async def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all active connections.

        Returns:
            List of connection information dictionaries
        """
        async with self._lock:
            return [
                {
                    "client_id": client_id,
                    "is_active": connection.is_active,
                }
                for client_id, connection in self.connections.items()
            ]

    async def cleanup_inactive_connections(self) -> int:
        """Clean up inactive connections.

        Returns:
            Number of connections cleaned up
        """
        cleanup_count = 0

        async with self._lock:
            inactive_clients = [
                client_id for client_id, connection in self.connections.items() if not connection.is_active
            ]

            for client_id in inactive_clients:
                del self.connections[client_id]
                cleanup_count += 1
                logger.debug(f"Cleaned up inactive connection: {client_id}")

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} inactive WebSocket connections")

        return cleanup_count

    # Event broadcasting convenience methods

    async def broadcast_document_added(self, document_data: Dict[str, Any]) -> int:
        """Broadcast document added event.

        Args:
            document_data: Document information

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.DOCUMENT_ADDED,
            document_data,
            f"Document added: {document_data.get('title', document_data.get('filename', 'Unknown'))}",
        )

    async def broadcast_document_removed(self, document_id: str, document_path: str) -> int:
        """Broadcast document removed event.

        Args:
            document_id: ID of removed document
            document_path: Path of removed document

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.DOCUMENT_REMOVED,
            {"document_id": document_id, "document_path": document_path},
            f"Document removed: {document_path}",
        )

    async def broadcast_processing_started(self, filename: str, document_id: Optional[str] = None) -> int:
        """Broadcast document processing started event.

        Args:
            filename: Name of file being processed
            document_id: Optional document ID

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.PROCESSING_STARTED,
            {"filename": filename, "document_id": document_id},
            f"Processing started: {filename}",
        )

    async def broadcast_processing_completed(self, document_data: Dict[str, Any]) -> int:
        """Broadcast document processing completed event.

        Args:
            document_data: Processed document information

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.PROCESSING_COMPLETED,
            document_data,
            f"Processing completed: {document_data.get('filename', 'Unknown')}",
        )

    async def broadcast_processing_failed(self, filename: str, error: str) -> int:
        """Broadcast document processing failed event.

        Args:
            filename: Name of file that failed to process
            error: Error message

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.PROCESSING_FAILED,
            {"filename": filename, "error": error},
            f"Processing failed: {filename} - {error}",
        )

    async def broadcast_search_performed(self, query: str, result_count: int) -> int:
        """Broadcast search performed event.

        Args:
            query: Search query
            result_count: Number of results found

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.SEARCH_PERFORMED,
            {"query": query, "result_count": result_count},
            f"Search performed: '{query}' ({result_count} results)",
        )

    async def broadcast_job_status_changed(self, job_id: str, status: str, progress: Optional[float] = None) -> int:
        """Broadcast job status changed event.

        Args:
            job_id: Job identifier
            status: New job status
            progress: Optional progress value (0.0 to 1.0)

        Returns:
            Number of clients notified
        """
        data = {"job_id": job_id, "status": status}
        if progress is not None:
            data["progress"] = progress

        return await self.broadcast(
            WebsocketEventType.JOB_STATUS_CHANGED, data, f"Job {job_id} status changed to {status}"
        )

    async def broadcast_job_progress_updated(self, job_id: str, progress: float, message: Optional[str] = None) -> int:
        """Broadcast job progress update event.

        Args:
            job_id: Job identifier
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.JOB_PROGRESS_UPDATED,
            {"job_id": job_id, "progress": progress, "message": message},
            f"Job {job_id} progress: {int(progress * 100)}%",
        )

    async def broadcast_job_cancelled(self, job_id: str, reason: Optional[str] = None) -> int:
        """Broadcast job cancelled event.

        Args:
            job_id: Job identifier
            reason: Optional cancellation reason

        Returns:
            Number of clients notified
        """
        return await self.broadcast(
            WebsocketEventType.JOB_CANCELLED,
            {"job_id": job_id, "reason": reason},
            f"Job {job_id} cancelled" + (f": {reason}" if reason else ""),
        )
