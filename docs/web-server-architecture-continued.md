# Web Server Architecture - Continuation

## 9. Error Handling (Continued)

### 9.1 Global Error Handler (Continued)

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for consistent error responses."""

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Map internal exceptions to HTTP status codes
    status_code_map = {
        ValidationError: 400,
        DocumentNotFoundError: 404,
        PDFProcessingError: 422,
        VectorStoreError: 503,
        EmbeddingError: 503,
        FileSystemError: 500,
        ConfigurationError: 500
    }

    status_code = status_code_map.get(type(exc), 500)

    # Create error response
    error_response = ErrorResponse(
        success=False,
        message=str(exc) if app.debug else "An error occurred",
        error_code=exc.__class__.__name__
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            success=False,
            message="Validation error",
            error_code="VALIDATION_ERROR",
            error_details={
                "errors": exc.errors(),
                "body": exc.body
            }
        ).dict()
    )
```

### 9.2 Service-Level Error Handling

```python
class WebDocumentService:
    """Web service layer with comprehensive error handling."""

    async def upload_document(self, file: UploadFile, metadata: Optional[Dict] = None) -> DocumentResponse:
        """Upload document with error handling."""
        try:
            # Validate file
            await self._validate_upload_file(file)

            # Save temporary file
            temp_path = await self._save_temp_file(file)

            try:
                # Process document
                result = await self.pdf_processor.process_pdf(temp_path, metadata)

                if not result.success:
                    raise PDFProcessingError(result.error or "Processing failed")

                # Add to vector store
                await self.vector_store.add_document(result.document)

                return DocumentResponse(
                    success=True,
                    document=result.document.to_web_dict(),
                    processing_stats={
                        "chunks_created": result.chunks_created,
                        "embeddings_generated": result.embeddings_generated,
                        "processing_time": result.processing_time
                    }
                )

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"Document upload failed: {e}", exc_info=True)

            # Re-raise known exceptions
            if isinstance(e, (ValidationError, PDFProcessingError, VectorStoreError)):
                raise

            # Wrap unknown exceptions
            raise PDFProcessingError(f"Upload failed: {str(e)}")
```

### 9.3 WebSocket Error Handling

```python
class WebSocketManager:
    async def handle_websocket_error(self, websocket: WebSocket, client_id: str, error: Exception):
        """Handle WebSocket errors gracefully."""

        error_message = {
            "type": "error",
            "data": {
                "error_type": error.__class__.__name__,
                "message": str(error),
                "client_id": client_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

        try:
            await websocket.send_json(error_message)
        except Exception:
            # Connection may be closed, remove from active connections
            self.connections.pop(client_id, None)

        logger.error(f"WebSocket error for client {client_id}: {error}")
```

### 9.4 Frontend Error Handling

```typescript
// services/api.js
class APIService {
  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      timeout: 30000
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add request ID for tracing
        config.headers['X-Request-ID'] = generateRequestId();
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response.data,
      (error) => {
        const errorResponse = this.handleAPIError(error);
        return Promise.reject(errorResponse);
      }
    );
  }

  handleAPIError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;

      return {
        type: 'api_error',
        status,
        message: data.message || 'An error occurred',
        code: data.error_code,
        details: data.error_details
      };
    } else if (error.request) {
      // Network error
      return {
        type: 'network_error',
        message: 'Unable to connect to server',
        details: { originalError: error.message }
      };
    } else {
      // Request setup error
      return {
        type: 'client_error',
        message: error.message
      };
    }
  }
}

// composables/useErrorHandler.js
export function useErrorHandler() {
  const showError = (error) => {
    const errorTypes = {
      'validation_error': 'Please check your input and try again',
      'network_error': 'Please check your connection and try again',
      'server_error': 'Server error - please try again later',
      'processing_error': 'Unable to process document - please check file format'
    };

    const userMessage = errorTypes[error.type] || error.message || 'An unexpected error occurred';

    // Show user-friendly notification
    toast.error(userMessage);

    // Log detailed error for debugging
    console.error('Application error:', error);
  };

  return { showError };
}
```

## 10. Implementation Plan

### 10.1 Phase 1: Core Web Server Integration

**Duration: 1-2 weeks**

**Tasks:**
1. **Extend PDFKnowledgebaseServer class** ([`src/pdfkb/main.py`](../src/pdfkb/main.py:31))
   - Add FastAPI app initialization
   - Implement concurrent MCP and web server running
   - Add web server configuration loading

2. **Create Web Service Layer**
   - [`src/pdfkb/web/services/document_service.py`](../src/pdfkb/web/services/document_service.py)
   - [`src/pdfkb/web/services/search_service.py`](../src/pdfkb/web/services/search_service.py)
   - [`src/pdfkb/web/services/status_service.py`](../src/pdfkb/web/services/status_service.py)

3. **Implement Core Data Models**
   - [`src/pdfkb/web/models/requests.py`](../src/pdfkb/web/models/requests.py)
   - [`src/pdfkb/web/models/responses.py`](../src/pdfkb/web/models/responses.py)
   - Extend existing models with web methods

4. **Update Configuration**
   - Add web server settings to [`ServerConfig`](../src/pdfkb/config.py:17)
   - Environment variable handling
   - Validation and defaults

**Deliverables:**
- Functional web server running alongside MCP server
- Basic API endpoints working with existing services
- Configuration integration complete

### 10.2 Phase 2: Document Management API

**Duration: 1 week**

**Tasks:**
1. **Document Upload Endpoints**
   - File upload with validation
   - Progress tracking integration
   - Metadata handling

2. **Document Management Endpoints**
   - List documents with pagination
   - Document details with preview
   - Document removal
   - Add by path functionality

3. **Error Handling**
   - Global exception handlers
   - Service-level error handling
   - Input validation

4. **Testing**
   - API endpoint testing
   - Integration tests with existing services
   - Error scenario testing

**Deliverables:**
- Complete document management API
- Comprehensive error handling
- API documentation
- Test coverage

### 10.3 Phase 3: Search and WebSocket Implementation

**Duration: 1 week**

**Tasks:**
1. **Search API Endpoints**
   - Basic search implementation
   - Advanced search with filters
   - Search result highlighting
   - Similar documents endpoint

2. **WebSocket Implementation**
   - Connection management
   - Event broadcasting system
   - Real-time processing updates
   - Client subscription handling

3. **Status and Statistics Endpoints**
   - System status monitoring
   - Processing statistics
   - Configuration information

4. **Integration with File Monitor**
   - Real-time file system events
   - Processing progress updates
   - Error notifications

**Deliverables:**
- Full search API functionality
- Real-time WebSocket system
- System monitoring endpoints
- Integration with existing monitoring

### 10.4 Phase 4: Frontend Development

**Duration: 2-3 weeks**

**Tasks:**
1. **Project Setup**
   - Vue 3/React 18 project initialization
   - Build tools and development environment
   - CSS framework integration (Tailwind CSS)

2. **Core Components**
   - Layout components (header, sidebar, main)
   - Document list and card components
   - Search interface components
   - Common UI components (modals, notifications, etc.)

3. **Document Management UI**
   - Document list with pagination
   - Upload interface with progress tracking
   - Document details and preview
   - Add by path functionality

4. **Search Interface**
   - Basic search bar
   - Advanced search form
   - Search results display
   - Result highlighting

5. **Real-time Features**
   - WebSocket integration
   - Real-time processing updates
   - Notification system
   - Processing queue display

6. **System Status Dashboard**
   - Status overview
   - Processing statistics
   - Configuration display

**Deliverables:**
- Complete frontend application
- Responsive design implementation
- Real-time update integration
- User-friendly interface

### 10.5 Phase 5: Testing and Optimization

**Duration: 1 week**

**Tasks:**
1. **Comprehensive Testing**
   - API integration testing
   - Frontend component testing
   - End-to-end testing
   - Performance testing

2. **Security Review**
   - Input validation testing
   - File upload security testing
   - WebSocket security validation
   - Error handling security review

3. **Performance Optimization**
   - API response optimization
   - Frontend build optimization
   - WebSocket performance tuning
   - Database query optimization

4. **Documentation**
   - API documentation generation
   - User guide creation
   - Deployment documentation
   - Configuration guide

**Deliverables:**
- Fully tested web interface
- Security-reviewed implementation
- Performance-optimized system
- Complete documentation

### 10.6 Deployment Considerations

**Development Deployment:**
```bash
# Start with web server enabled
WEB_SERVER_ENABLED=true WEB_SERVER_PORT=8080 python -m pdfkb.main
```

**Production Deployment Options:**

1. **Single Process (Recommended for small deployments):**
   ```bash
   # Both MCP and Web server in one process
   WEB_SERVER_ENABLED=true WEB_SERVER_PORT=8080 python -m pdfkb.main
   ```

2. **Reverse Proxy Setup:**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location /api/ {
           proxy_pass http://localhost:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /ws/ {
           proxy_pass http://localhost:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }

       location / {
           root /path/to/frontend/dist;
           try_files $uri $uri/ /index.html;
       }
   }
   ```

3. **Docker Deployment:**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app
   COPY . .
   RUN pip install -e .

   EXPOSE 8080

   ENV WEB_SERVER_ENABLED=true
   ENV WEB_SERVER_PORT=8080

   CMD ["python", "-m", "pdfkb.main"]
   ```

### 10.7 Monitoring and Observability

**Logging Integration:**
```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        "HTTP request",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        user_agent=request.headers.get("user-agent"),
        request_id=request.headers.get("x-request-id")
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response
```

**Metrics Collection:**
```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
DOCUMENT_PROCESSING_DURATION = Histogram('document_processing_duration_seconds', 'Document processing duration')
WEBSOCKET_CONNECTIONS = Counter('websocket_connections_total', 'Total WebSocket connections')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 11. Conclusion

This comprehensive technical specification provides a detailed blueprint for implementing a modern web interface for the PDF KB MCP server. The design leverages all existing components while providing a clean, scalable, and maintainable web layer.

### Key Features Delivered:

1. **Integrated Architecture**: Web server runs alongside MCP functionality in the same process
2. **Comprehensive API**: RESTful endpoints for all document management and search operations
3. **Real-time Updates**: WebSocket integration for live processing status and notifications
4. **Modern Frontend**: Responsive, accessible web interface with real-time capabilities
5. **Robust Error Handling**: Comprehensive error management at all layers
6. **Security Considerations**: Input validation, rate limiting, and security best practices
7. **Performance Optimization**: Caching, pagination, and efficient data handling
8. **Scalability**: Designed to handle growing document collections and user bases

### Integration Benefits:

- **Shared Services**: Leverages existing [`PDFProcessor`](../src/pdfkb/pdf_processor.py:27), [`VectorStore`](../src/pdfkb/vector_store.py:13), and other core services
- **Configuration Unity**: Single configuration system for both MCP and web interfaces
- **Cache Efficiency**: Utilizes existing [`IntelligentCacheManager`](../src/pdfkb/intelligent_cache.py:13) for optimal performance
- **Monitoring Integration**: Works with existing [`FileMonitor`](../src/pdfkb/file_monitor.py:135) for real-time file system updates

The implementation follows a phased approach, ensuring each component is thoroughly tested and integrated before moving to the next phase. The result will be a production-ready web interface that complements the existing MCP functionality while providing users with a modern, intuitive way to interact with their PDF knowledge base.

This design satisfies all the original requirements while providing a solid foundation for future enhancements and scaling.
