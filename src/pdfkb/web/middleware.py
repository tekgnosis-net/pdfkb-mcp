"""FastAPI middleware for CORS, error handling, and other cross-cutting concerns."""

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import ServerConfig
from .models.web_models import ErrorResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request with logging.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        start_time = time.time()

        # Log request
        logger.info(f"HTTP {request.method} {request.url.path} - Started")

        try:
            response = await call_next(request)

            # Calculate request duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"HTTP {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Duration: {duration:.3f}s"
            )

            # Add duration header
            response.headers["X-Process-Time"] = str(duration)

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"HTTP {request.method} {request.url.path} - " f"Error: {str(e)} - " f"Duration: {duration:.3f}s"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions and returning standardized error responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request with error handling.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with error handling
        """
        try:
            return await call_next(request)

        except HTTPException:
            # Re-raise HTTP exceptions (they're already handled by FastAPI)
            raise

        except RequestValidationError as e:
            # Handle validation errors
            logger.warning(f"Validation error on {request.method} {request.url.path}: {e}")

            error_details = []
            for error in e.errors():
                error_details.append(
                    {
                        "field": " -> ".join(str(loc) for loc in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                )

            error_response = ErrorResponse(
                error="Validation failed", error_code="VALIDATION_ERROR", details={"validation_errors": error_details}
            )

            return JSONResponse(status_code=422, content=error_response.model_dump())

        except Exception as e:
            # Handle unexpected exceptions
            logger.error(f"Unexpected error on {request.method} {request.url.path}: {e}", exc_info=True)

            error_response = ErrorResponse(
                error="Internal server error", error_code="INTERNAL_ERROR", details={"message": str(e)}
            )

            return JSONResponse(status_code=500, content=error_response.model_dump())


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(self, app, add_security_headers: bool = True):
        """Initialize security headers middleware.

        Args:
            app: FastAPI application
            add_security_headers: Whether to add security headers
        """
        super().__init__(app)
        self.add_security_headers = add_security_headers

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request with security headers.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with security headers
        """
        response = await call_next(request)

        if self.add_security_headers:
            # Add security headers
            response.headers.update(
                {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                    "Content-Security-Policy": (
                        "default-src 'self'; script-src 'self' 'unsafe-inline'; " "style-src 'self' 'unsafe-inline'"
                    ),
                }
            )

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, max_requests_per_minute: int = 100):
        """Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            max_requests_per_minute: Maximum requests per minute per client
        """
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request with rate limiting.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response or rate limit error
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        current_minute = int(current_time // 60)

        # Clean old entries
        self._cleanup_old_entries(current_minute)

        # Check rate limit
        if client_ip in self.request_counts:
            client_data = self.request_counts[client_ip]
            if client_data["minute"] == current_minute:
                if client_data["count"] >= self.max_requests:
                    logger.warning(f"Rate limit exceeded for client {client_ip}")

                    error_response = ErrorResponse(
                        error="Rate limit exceeded",
                        error_code="RATE_LIMIT_EXCEEDED",
                        details={"max_requests_per_minute": self.max_requests, "retry_after": 60 - (current_time % 60)},
                    )

                    return JSONResponse(
                        status_code=429, content=error_response.model_dump(), headers={"Retry-After": "60"}
                    )

                client_data["count"] += 1
            else:
                # New minute
                self.request_counts[client_ip] = {"minute": current_minute, "count": 1}
        else:
            # New client
            self.request_counts[client_ip] = {"minute": current_minute, "count": 1}

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.max_requests - self.request_counts[client_ip]["count"])
        response.headers.update(
            {
                "X-RateLimit-Limit": str(self.max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str((current_minute + 1) * 60),
            }
        )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check for forwarded headers first (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"

    def _cleanup_old_entries(self, current_minute: int) -> None:
        """Clean up old rate limit entries.

        Args:
            current_minute: Current minute timestamp
        """
        # Remove entries older than 2 minutes
        cutoff_minute = current_minute - 2
        clients_to_remove = [
            client_ip for client_ip, data in self.request_counts.items() if data["minute"] < cutoff_minute
        ]

        for client_ip in clients_to_remove:
            del self.request_counts[client_ip]


def setup_middleware(app: FastAPI, config: ServerConfig) -> None:
    """Set up all middleware for the FastAPI application.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    # CORS middleware (must be added first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.web_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-RateLimit-*"],
    )

    # Custom middleware (order matters - last added is executed first)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware, add_security_headers=True)

    # Rate limiting (optional - can be disabled for development)
    if hasattr(config, "enable_rate_limiting") and config.enable_rate_limiting:
        app.add_middleware(
            RateLimitingMiddleware, max_requests_per_minute=getattr(config, "max_requests_per_minute", 100)
        )

    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("All middleware configured successfully")


def setup_exception_handlers(app: FastAPI) -> None:
    """Set up global exception handlers for the FastAPI application.

    Args:
        app: FastAPI application
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions with standardized error format.

        Args:
            request: HTTP request
            exc: HTTP exception

        Returns:
            Standardized error response
        """
        logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")

        error_response = ErrorResponse(
            error=exc.detail, error_code=f"HTTP_{exc.status_code}", details={"status_code": exc.status_code}
        )

        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump(), headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation exceptions with detailed error information.

        Args:
            request: HTTP request
            exc: Validation exception

        Returns:
            Detailed validation error response
        """
        logger.warning(f"Validation error on {request.method} {request.url.path}: {exc}")

        error_details = []
        for error in exc.errors():
            error_details.append(
                {
                    "field": " -> ".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

        error_response = ErrorResponse(
            error="Request validation failed",
            error_code="VALIDATION_ERROR",
            details={
                "validation_errors": error_details,
                "error_count": len(error_details),
            },
        )

        return JSONResponse(status_code=422, content=error_response.model_dump())

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions with logging.

        Args:
            request: HTTP request
            exc: Unexpected exception

        Returns:
            Generic error response
        """
        logger.error(f"Unexpected error on {request.method} {request.url.path}: {exc}", exc_info=True)

        error_response = ErrorResponse(
            error="An unexpected error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            details={"message": "Please try again or contact support if the problem persists"},
        )

        return JSONResponse(status_code=500, content=error_response.model_dump())

    logger.info("Exception handlers configured successfully")
