"""WebSocket endpoint handlers and advanced connection management."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .models.web_models import WebsocketEventType
from .services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class WebSocketEventHandler:
    """Advanced WebSocket event handling with message routing and filtering."""

    def __init__(self, websocket_manager: WebSocketManager):
        """Initialize WebSocket event handler.

        Args:
            websocket_manager: WebSocket connection manager
        """
        self.websocket_manager = websocket_manager
        self.client_subscriptions: Dict[str, List[str]] = {}  # client_id -> event_types
        self.client_filters: Dict[str, Dict[str, Any]] = {}  # client_id -> filters

    async def handle_websocket_connection(self, websocket: WebSocket) -> None:
        """Handle a complete WebSocket connection lifecycle.

        Args:
            websocket: FastAPI WebSocket instance
        """
        client_id = None
        try:
            # Accept connection
            client_id = await self.websocket_manager.connect(websocket)
            logger.info(f"WebSocket client connected with advanced handling: {client_id}")

            # Initialize client state
            self.client_subscriptions[client_id] = []
            self.client_filters[client_id] = {}

            # Main message handling loop
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Route message to appropriate handler
                    await self._route_client_message(client_id, message)

                except WebSocketDisconnect:
                    logger.info(f"WebSocket client disconnected: {client_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from client {client_id}: {e}")
                    await self._send_error_message(client_id, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                    await self._send_error_message(client_id, f"Message handling error: {str(e)}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            if client_id:
                # Clean up client state
                self.client_subscriptions.pop(client_id, None)
                self.client_filters.pop(client_id, None)
                await self.websocket_manager.disconnect(client_id)

    async def _route_client_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Route client message to appropriate handler.

        Args:
            client_id: Client ID
            message: Message data from client
        """
        message_type = message.get("type")

        handlers = {
            "ping": self._handle_ping,
            "subscribe": self._handle_subscription,
            "unsubscribe": self._handle_unsubscription,
            "set_filter": self._handle_set_filter,
            "clear_filter": self._handle_clear_filter,
            "get_status": self._handle_get_status,
            "broadcast_test": self._handle_broadcast_test,
        }

        handler = handlers.get(message_type, self._handle_unknown_message)
        await handler(client_id, message)

    async def _handle_ping(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle ping message from client.

        Args:
            client_id: Client ID
            message: Ping message
        """
        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {"type": "pong", "timestamp": message.get("timestamp")},
            "Pong response",
        )

    async def _handle_subscription(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle event subscription request.

        Args:
            client_id: Client ID
            message: Subscription message
        """
        event_types = message.get("event_types", [])

        if not isinstance(event_types, list):
            await self._send_error_message(client_id, "event_types must be a list")
            return

        # Validate event types
        valid_event_types = [e.value for e in WebsocketEventType]
        invalid_types = [et for et in event_types if et not in valid_event_types]

        if invalid_types:
            await self._send_error_message(
                client_id, f"Invalid event types: {invalid_types}. Valid types: {valid_event_types}"
            )
            return

        # Update subscriptions
        self.client_subscriptions[client_id] = event_types

        logger.info(f"Client {client_id} subscribed to events: {event_types}")

        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {
                "type": "subscription_confirmed",
                "subscribed_events": event_types,
                "subscription_count": len(event_types),
            },
            f"Subscribed to {len(event_types)} event types",
        )

    async def _handle_unsubscription(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle event unsubscription request.

        Args:
            client_id: Client ID
            message: Unsubscription message
        """
        event_types = message.get("event_types", [])

        if not isinstance(event_types, list):
            await self._send_error_message(client_id, "event_types must be a list")
            return

        # Remove from subscriptions
        current_subs = self.client_subscriptions.get(client_id, [])
        updated_subs = [et for et in current_subs if et not in event_types]
        self.client_subscriptions[client_id] = updated_subs

        logger.info(f"Client {client_id} unsubscribed from events: {event_types}")

        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {
                "type": "unsubscription_confirmed",
                "unsubscribed_events": event_types,
                "remaining_subscriptions": updated_subs,
            },
            f"Unsubscribed from {len(event_types)} event types",
        )

    async def _handle_set_filter(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle filter setting request.

        Args:
            client_id: Client ID
            message: Filter setting message
        """
        filters = message.get("filters", {})

        if not isinstance(filters, dict):
            await self._send_error_message(client_id, "filters must be a dictionary")
            return

        self.client_filters[client_id] = filters

        logger.info(f"Client {client_id} set filters: {filters}")

        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {
                "type": "filter_set",
                "filters": filters,
            },
            "Filters updated successfully",
        )

    async def _handle_clear_filter(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle filter clearing request.

        Args:
            client_id: Client ID
            message: Filter clearing message
        """
        self.client_filters[client_id] = {}

        logger.info(f"Client {client_id} cleared filters")

        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {"type": "filter_cleared"},
            "Filters cleared successfully",
        )

    async def _handle_get_status(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle status request from client.

        Args:
            client_id: Client ID
            message: Status request message
        """
        subscriptions = self.client_subscriptions.get(client_id, [])
        filters = self.client_filters.get(client_id, {})
        connection_count = await self.websocket_manager.get_connection_count()

        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.SYSTEM_STATUS_CHANGED,
            {
                "type": "client_status",
                "client_id": client_id,
                "subscriptions": subscriptions,
                "filters": filters,
                "total_connections": connection_count,
            },
            "Client status information",
        )

    async def _handle_broadcast_test(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle broadcast test request (for development/testing).

        Args:
            client_id: Client ID
            message: Broadcast test message
        """
        test_data = message.get("test_data", {"sender": client_id})

        # Only allow in development mode
        # In production, this handler should be disabled
        if hasattr(self, "_allow_test_broadcasts") and self._allow_test_broadcasts:
            count = await self.websocket_manager.broadcast(
                WebsocketEventType.SYSTEM_STATUS_CHANGED,
                {
                    "type": "broadcast_test",
                    "sender": client_id,
                    "data": test_data,
                },
                f"Test broadcast from {client_id}",
            )

            logger.info(f"Test broadcast from {client_id} sent to {count} clients")
        else:
            await self._send_error_message(client_id, "Broadcast testing not enabled")

    async def _handle_unknown_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle unknown message type.

        Args:
            client_id: Client ID
            message: Unknown message
        """
        message_type = message.get("type", "unknown")
        logger.warning(f"Unknown message type from client {client_id}: {message_type}")

        await self._send_error_message(client_id, f"Unknown message type: {message_type}")

    async def _send_error_message(self, client_id: str, error_message: str) -> None:
        """Send error message to client.

        Args:
            client_id: Client ID
            error_message: Error message to send
        """
        await self.websocket_manager.send_to_client(
            client_id,
            WebsocketEventType.ERROR_OCCURRED,
            {
                "error": error_message,
                "timestamp": asyncio.get_event_loop().time(),
            },
            f"Error: {error_message}",
        )

    def should_send_to_client(self, client_id: str, event_type: WebsocketEventType, event_data: Dict[str, Any]) -> bool:
        """Check if event should be sent to specific client based on subscriptions and filters.

        Args:
            client_id: Client ID
            event_type: Event type
            event_data: Event data

        Returns:
            True if event should be sent to client
        """
        # Check subscriptions
        client_subs = self.client_subscriptions.get(client_id, [])
        if client_subs and event_type.value not in client_subs:
            return False

        # Check filters
        client_filters = self.client_filters.get(client_id, {})
        if client_filters:
            for filter_key, filter_value in client_filters.items():
                event_value = event_data.get(filter_key)

                # Simple filter matching - could be extended with more complex logic
                if isinstance(filter_value, list):
                    if event_value not in filter_value:
                        return False
                elif event_value != filter_value:
                    return False

        return True

    async def broadcast_filtered(
        self,
        event_type: WebsocketEventType,
        data: Dict[str, Any],
        message: Optional[str] = None,
    ) -> int:
        """Broadcast message to clients with subscription and filter checking.

        Args:
            event_type: Event type
            data: Event data
            message: Optional message

        Returns:
            Number of clients that received the message
        """
        if not self.websocket_manager.connections:
            return 0

        sent_count = 0

        for client_id in list(self.websocket_manager.connections.keys()):
            if self.should_send_to_client(client_id, event_type, data):
                success = await self.websocket_manager.send_to_client(client_id, event_type, data, message)
                if success:
                    sent_count += 1

        logger.debug(f"Filtered broadcast of {event_type} sent to {sent_count} clients")
        return sent_count

    async def get_client_info(self) -> Dict[str, Any]:
        """Get detailed information about all connected clients.

        Returns:
            Dictionary with client information
        """
        client_info = {}

        for client_id in self.client_subscriptions.keys():
            client_info[client_id] = {
                "subscriptions": self.client_subscriptions.get(client_id, []),
                "filters": self.client_filters.get(client_id, {}),
                "subscription_count": len(self.client_subscriptions.get(client_id, [])),
                "filter_count": len(self.client_filters.get(client_id, {})),
            }

        return {
            "total_clients": len(client_info),
            "clients": client_info,
            "total_subscriptions": sum(len(subs) for subs in self.client_subscriptions.values()),
            "total_filters": sum(len(filters) for filters in self.client_filters.values()),
        }

    def enable_test_broadcasts(self, enabled: bool = True) -> None:
        """Enable or disable test broadcast functionality.

        Args:
            enabled: Whether to enable test broadcasts
        """
        self._allow_test_broadcasts = enabled
        logger.info(f"Test broadcasts {'enabled' if enabled else 'disabled'}")
