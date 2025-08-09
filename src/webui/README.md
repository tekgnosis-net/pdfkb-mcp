# PDF Knowledgebase - Modern Web Interface

A modern, responsive single-page application for the PDF Knowledgebase system with real-time updates and comprehensive document management capabilities.

## 🎯 Features

### Core Functionality
- **Document Management**: Upload, view, and manage PDF documents
- **Semantic Search**: Advanced search with relevance scoring and highlighting
- **Real-time Updates**: Live WebSocket connections for instant updates
- **Document Preview**: Full document content viewing and export
- **Chunk Navigation**: Browse document chunks with metadata
- **File Upload**: Drag-and-drop upload with progress tracking

### User Interface
- **Modern Design**: Clean, professional interface with dark/light themes
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile
- **Accessibility**: ARIA labels, keyboard navigation, and screen reader support
- **Progressive Enhancement**: Graceful degradation for older browsers
- **Loading States**: Comprehensive loading indicators and skeleton screens

### Technical Features
- **WebSocket Integration**: Real-time document processing updates
- **Client-side Routing**: Single-page application with browser history
- **Component Architecture**: Modular, maintainable JavaScript components
- **Error Handling**: User-friendly error messages and recovery
- **Performance Optimized**: Lazy loading, debounced search, and caching

## 📁 File Structure

```
src/webui/
├── index.html              # Main HTML structure
├── styles.css              # Modern CSS framework with themes
├── app.js                  # Main application logic and routing
├── components/             # Modular JavaScript components
│   ├── DocumentList.js     # Document listing and management
│   ├── DocumentDetail.js   # Document detail view with chunks
│   ├── SearchInterface.js  # Search functionality
│   ├── FileUpload.js       # File upload and path management
│   └── StatusBar.js        # Real-time status and metrics
├── README.md              # This documentation
└── test_frontend.py       # Basic functionality test
```

## 🚀 Getting Started

### Prerequisites
- PDF Knowledgebase backend server running
- Modern web browser (Chrome 88+, Firefox 85+, Safari 14+, Edge 88+)

### Running the Web Interface

1. **Start the Backend Server**:
   ```bash
   # From the project root directory
   python -m pdfkb.web_server
   ```

2. **Access the Web Interface**:
   - Open your browser and navigate to: `http://localhost:8000`
   - The interface should load automatically with the connection status indicator

3. **API Documentation**:
   - REST API docs: `http://localhost:8000/docs`
   - ReDoc documentation: `http://localhost:8000/redoc`

### Configuration
The web interface automatically connects to the backend API and WebSocket endpoints. Configuration is handled server-side through environment variables.

## 🎨 User Interface Guide

### Navigation
- **Documents**: Main document library with upload and management
- **Search**: Semantic search across all document content
- **Upload**: File upload via drag-and-drop or path input

### Document Management
1. **Upload Documents**:
   - Drag and drop PDF files onto the upload zone
   - Or click "browse to select files" for file picker
   - Add documents by file system path

2. **View Documents**:
   - Click any document card to view details
   - Use the filter box to search by filename or path
   - Pagination controls for large document collections

3. **Document Details**:
   - View comprehensive document metadata
   - Browse all document chunks with navigation
   - Preview full document content
   - Export document text

### Search Features
1. **Semantic Search**:
   - Enter natural language queries
   - Adjust result count and minimum relevance score
   - Search terms are highlighted in results

2. **Advanced Options**:
   - Filter by document metadata
   - Set minimum similarity thresholds
   - Export search results to CSV

### Real-time Updates
- Connection status indicator in header
- Live updates for document processing
- Real-time search and upload notifications
- System metrics and status

## 🛠️ Technical Implementation

### Architecture
- **Frontend**: Vanilla JavaScript ES6+ with modular components
- **Styling**: CSS Grid/Flexbox with CSS Custom Properties
- **Communication**: REST API + WebSocket for real-time updates
- **State Management**: Centralized app state with component coordination

### Components Overview

#### Main Application (`app.js`)
- Client-side routing and navigation
- WebSocket connection management
- API client with error handling
- Theme management and persistence
- Toast notification system

#### DocumentList (`components/DocumentList.js`)
- Document grid/list rendering
- Filtering and pagination
- Document actions (view, preview, remove)
- Real-time document updates

#### SearchInterface (`components/SearchInterface.js`)
- Search form with advanced options
- Results rendering with highlighting
- Search suggestions and history
- Export functionality

#### DocumentDetail (`components/DocumentDetail.js`)
- Comprehensive document information
- Chunk navigation and expansion
- Content preview and export
- Search term highlighting

#### FileUpload (`components/FileUpload.js`)
- Drag-and-drop file handling
- Upload progress tracking
- Batch upload support
- Path-based document addition

#### StatusBar (`components/StatusBar.js`)
- Connection status monitoring
- System metrics display
- Real-time activity indicators
- Diagnostic information

### Responsive Design
- **Desktop**: Full-featured interface with multi-column layouts
- **Tablet**: Adapted layouts with touch-friendly controls
- **Mobile**: Single-column layout with collapsible navigation

### Browser Support
- **Modern Browsers**: Full feature support
- **Legacy Support**: Progressive enhancement for older browsers
- **Accessibility**: WCAG 2.1 AA compliance

## 🔧 Customization

### Theming
The interface supports light and dark themes with system preference detection:
- Themes stored in localStorage
- CSS custom properties for easy customization
- Automatic theme switching based on system preferences

### Configuration
Key customizable aspects:
- Color schemes and branding
- Component layouts and spacing
- API endpoints and timeouts
- Upload file size limits

## 🐛 Troubleshooting

### Common Issues

1. **Connection Failed**:
   - Verify backend server is running
   - Check firewall and network settings
   - Confirm correct port and host configuration

2. **Upload Failures**:
   - Ensure PDF file format
   - Check file size limits
   - Verify server disk space

3. **Search Not Working**:
   - Confirm documents are processed (indexed status)
   - Check embedding service configuration
   - Verify vector store initialization

4. **WebSocket Issues**:
   - Browser developer tools → Network tab
   - Look for WebSocket connection errors
   - Check server logs for WebSocket messages

### Performance Optimization
- Use browser caching for static assets
- Enable gzip compression on server
- Optimize document chunk sizes
- Consider CDN for static files in production

## 📱 Mobile Support

The interface is fully responsive and optimized for mobile devices:
- Touch-friendly controls and spacing
- Adaptive layouts for different screen sizes
- Mobile-optimized file upload
- Swipe gestures for navigation

## 🔒 Security

Security considerations implemented:
- CSRF protection via FastAPI
- Input validation and sanitization
- File type restrictions (PDF only)
- WebSocket connection authentication
- Error message sanitization

## 📈 Performance

Performance optimizations included:
- Lazy loading for large document lists
- Debounced search input (300ms delay)
- Virtual scrolling for large result sets
- Optimistic UI updates
- Efficient WebSocket message handling

## 🤝 Contributing

To extend or modify the web interface:

1. **Adding Components**:
   - Create new JS files in `components/`
   - Register with main app in `app.js`
   - Add corresponding CSS in `styles.css`

2. **Styling**:
   - Use CSS custom properties for consistency
   - Follow existing naming conventions
   - Test across different browsers and devices

3. **API Integration**:
   - Extend the API client in `app.js`
   - Add error handling and loading states
   - Update WebSocket message handlers

## 📄 License

This web interface is part of the PDF Knowledgebase project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review browser developer console for errors
3. Check server logs for backend issues
4. Ensure all dependencies are installed and up-to-date

---

**Built with modern web technologies for a seamless document management experience.**
