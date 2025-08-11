"""Test that datetime fields use timezone-aware UTC timestamps."""

from datetime import timezone

from pdfkb.web.models.web_models import HealthCheckResponse, WebsocketEventType, WebsocketMessage


class TestDatetimeTimezone:
    """Test cases for timezone-aware datetime handling."""

    def test_health_check_timezone_aware(self):
        """Test that HealthCheckResponse timestamp is timezone-aware."""
        response = HealthCheckResponse()

        # Should be timezone-aware
        assert response.timestamp.tzinfo is not None
        assert response.timestamp.tzinfo == timezone.utc

        # Should be close to current UTC time
        from datetime import datetime

        now_utc = datetime.now(timezone.utc)
        time_diff = abs((now_utc - response.timestamp).total_seconds())
        assert time_diff < 1.0, "Timestamp should be very close to current UTC time"

    def test_websocket_message_timezone_aware(self):
        """Test that WebsocketMessage timestamp is timezone-aware."""
        message = WebsocketMessage(event_type=WebsocketEventType.DOCUMENT_ADDED)

        # Should be timezone-aware
        assert message.timestamp.tzinfo is not None
        assert message.timestamp.tzinfo == timezone.utc

        # Should be close to current UTC time
        from datetime import datetime

        now_utc = datetime.now(timezone.utc)
        time_diff = abs((now_utc - message.timestamp).total_seconds())
        assert time_diff < 1.0, "Timestamp should be very close to current UTC time"

    def test_utc_now_function(self):
        """Test that utc_now helper function works correctly."""
        from pdfkb.web.models.web_models import utc_now

        result = utc_now()

        # Should be timezone-aware
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

        # Should be close to current UTC time
        from datetime import datetime

        now_utc = datetime.now(timezone.utc)
        time_diff = abs((now_utc - result).total_seconds())
        assert time_diff < 1.0, "Result should be very close to current UTC time"
