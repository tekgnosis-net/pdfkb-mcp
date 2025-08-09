/**
 * PDF Knowledgebase Frontend Application
 * Main application logic with routing, WebSocket, and API integration
 */

class PDFKnowledgebaseApp {
    constructor() {
        // Application state
        this.currentView = 'documents';
        this.currentPage = 1;
        this.documentsPerPage = 20;
        this.currentDocument = null;
        this.searchQuery = '';
        this.searchResults = [];
        this.documents = [];
        this.systemStats = { documents: 0, chunks: 0 };

        // WebSocket connection
        this.websocket = null;
        this.wsReconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.wsReconnectDelay = 1000; // Start with 1 second

        // Track client-initiated actions to prevent duplicate notifications
        this.clientInitiatedActions = new Map(); // Action type -> timestamp
        this.actionIgnoreWindow = 3000; // 3 seconds to ignore duplicate WebSocket events

        // API base URL
        this.apiBaseUrl = '/api';
        this.wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

        // Theme management
        this.theme = localStorage.getItem('theme') ||
                    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

        // Components (will be initialized by component files)
        this.components = {};

        // Initialize application
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing PDF Knowledgebase App...');

        // Set up theme
        this.applyTheme();

        // Set up event listeners
        this.setupEventListeners();

        // Initialize WebSocket connection
        this.initWebSocket();

        // Load initial data
        await this.loadSystemStatus();
        await this.loadDocuments();

        // Set up routing
        this.setupRouting();

        console.log('PDF Knowledgebase App initialized successfully');
    }

    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                if (view) this.navigateTo(view);
            });
        });

        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Breadcrumb back button
        const breadcrumbBack = document.getElementById('breadcrumb-back');
        if (breadcrumbBack) {
            breadcrumbBack.addEventListener('click', () => this.navigateBack());
        }

        // Modal controls
        this.setupModalControls();

        // Search input (documents view)
        const documentsSearch = document.getElementById('documents-search');
        if (documentsSearch) {
            documentsSearch.addEventListener('input', this.debounce((e) => {
                this.filterDocuments(e.target.value);
            }, 300));
        }

        // Refresh documents
        const refreshBtn = document.getElementById('refresh-documents');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDocuments(true));
        }

        // Upload first document button
        const uploadFirstDoc = document.getElementById('upload-first-doc');
        if (uploadFirstDoc) {
            uploadFirstDoc.addEventListener('click', () => this.navigateTo('upload'));
        }

        // Handle window events
        window.addEventListener('online', () => this.handleNetworkChange(true));
        window.addEventListener('offline', () => this.handleNetworkChange(false));
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    /**
     * Set up modal controls
     */
    setupModalControls() {
        const modal = document.getElementById('confirm-modal');
        const modalClose = document.getElementById('modal-close');
        const modalCancel = document.getElementById('confirm-cancel');
        const modalBackdrop = modal?.querySelector('.modal-backdrop');

        const closeModal = () => {
            if (modal) {
                modal.classList.remove('active');
                modal.classList.add('hidden');
            }
        };

        if (modalClose) modalClose.addEventListener('click', closeModal);
        if (modalCancel) modalCancel.addEventListener('click', closeModal);
        if (modalBackdrop) modalBackdrop.addEventListener('click', closeModal);

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
                closeModal();
            }
        });
    }

    /**
     * Set up client-side routing
     */
    setupRouting() {
        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            const view = e.state?.view || 'documents';
            this.navigateTo(view, false); // Don't push to history
        });

        // Set initial route
        const hash = window.location.hash.slice(1);
        if (hash) {
            this.navigateTo(hash, false);
        }
    }

    /**
     * Navigate to a specific view
     */
    navigateTo(view, pushHistory = true) {
        console.log(`Navigating to: ${view}`);

        // Update navigation state
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.view === view);
        });

        // Hide all views
        document.querySelectorAll('.view').forEach(viewEl => {
            viewEl.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(`${view}-view`);
        if (targetView) {
            targetView.classList.add('active');
        }

        // Update breadcrumb
        this.updateBreadcrumb(view);

        // Update browser history
        if (pushHistory) {
            const url = view === 'documents' ? '/' : `/#${view}`;
            window.history.pushState({ view }, '', url);
        }

        // Update current view
        this.currentView = view;

        // View-specific initialization
        switch (view) {
            case 'documents':
                if (this.documents.length === 0) {
                    this.loadDocuments();
                }
                break;
            case 'search':
                // Focus search input
                setTimeout(() => {
                    const searchInput = document.getElementById('search-input');
                    if (searchInput) searchInput.focus();
                }, 100);
                break;
        }
    }

    /**
     * Navigate back from detail views
     */
    navigateBack() {
        if (this.currentView === 'document-detail') {
            this.navigateTo('documents');
        } else {
            window.history.back();
        }
    }

    /**
     * Update breadcrumb navigation
     */
    updateBreadcrumb(view) {
        const breadcrumb = document.getElementById('breadcrumb');
        const breadcrumbText = document.getElementById('breadcrumb-text');

        if (view === 'document-detail' && this.currentDocument) {
            breadcrumb.classList.remove('hidden');
            breadcrumbText.textContent = `Documents / ${this.currentDocument.filename}`;
        } else {
            breadcrumb.classList.add('hidden');
        }
    }

    /**
     * Initialize WebSocket connection
     */
    initWebSocket() {
        console.log('Connecting to WebSocket...');
        this.updateConnectionStatus('connecting');

        try {
            this.websocket = new WebSocket(this.wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected successfully');
                this.updateConnectionStatus('connected');
                this.wsReconnectAttempts = 0;
                this.wsReconnectDelay = 1000;
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.websocket.onclose = (event) => {
                console.log('WebSocket connection closed:', event.code, event.reason);
                this.updateConnectionStatus('disconnected');
                this.scheduleReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus('disconnected');
            this.scheduleReconnect();
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(message) {
        console.log('WebSocket message received:', message);

        const { event_type, data, message: msg } = message;

        switch (event_type) {
            case 'document_added':
                this.handleDocumentAdded(data);
                this.showToast('success', 'Document Added', `${data.filename} has been processed successfully.`);
                break;

            case 'document_removed':
                this.handleDocumentRemoved(data);
                // Check if this removal was client-initiated (to prevent duplicate notifications)
                if (!this.isClientInitiatedAction('document_removed', data.document_id)) {
                    this.showToast('info', 'Document Removed', 'Document has been removed from the knowledgebase.');
                }
                break;

            case 'processing_started':
                this.showToast('info', 'Processing Started', `Processing ${data.filename}...`);
                break;

            case 'processing_completed':
                this.handleProcessingCompleted(data);
                this.showToast('success', 'Document Added',
                    `${data.filename} has been processed successfully with ${data.chunks_created} chunks.`);
                break;

            case 'processing_failed':
                this.handleProcessingFailed(data);
                this.showToast('error', 'Processing Failed', `Failed to process ${data.filename}: ${data.error || 'Unknown error'}`);
                break;

            case 'search_performed':
                // Optional: Update search analytics
                break;

            case 'system_status_changed':
                this.handleSystemStatusChanged(data);
                break;

            case 'error_occurred':
                this.showToast('error', 'Error', data.error || 'An error occurred');
                break;

            default:
                console.log('Unknown WebSocket event type:', event_type);
        }
    }

    /**
     * Schedule WebSocket reconnection
     */
    scheduleReconnect() {
        if (this.wsReconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max WebSocket reconnection attempts reached');
            this.showToast('error', 'Connection Lost', 'Unable to reconnect. Please refresh the page.');
            return;
        }

        this.wsReconnectAttempts++;
        const delay = Math.min(this.wsReconnectDelay * Math.pow(2, this.wsReconnectAttempts - 1), 30000);

        console.log(`Scheduling WebSocket reconnection in ${delay}ms (attempt ${this.wsReconnectAttempts})`);

        setTimeout(() => {
            if (this.websocket?.readyState === WebSocket.CLOSED) {
                this.initWebSocket();
            }
        }, delay);
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status) {
        const statusIndicator = document.getElementById('connection-status');
        const statusText = statusIndicator?.querySelector('.status-text');
        const indicator = statusIndicator?.querySelector('.status-indicator');

        if (indicator) {
            indicator.setAttribute('data-status', status);
        }

        if (statusText) {
            const statusMessages = {
                connecting: 'Connecting...',
                connected: 'Connected',
                disconnected: 'Disconnected'
            };
            statusText.textContent = statusMessages[status] || status;
        }
    }

    /**
     * Handle network connectivity changes
     */
    handleNetworkChange(isOnline) {
        if (isOnline) {
            console.log('Network connection restored');
            if (this.websocket?.readyState === WebSocket.CLOSED) {
                this.initWebSocket();
            }
        } else {
            console.log('Network connection lost');
            this.updateConnectionStatus('disconnected');
        }
    }

    /**
     * API Client Methods
     */

    /**
     * Make API request
     */
    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    /**
     * Load system status
     */
    async loadSystemStatus() {
        try {
            const status = await this.apiRequest('/status');
            this.systemStats = {
                documents: status.documents_count || 0,
                chunks: status.chunks_count || 0
            };
            this.updateSystemStats();
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }

    /**
     * Update system stats display
     */
    updateSystemStats() {
        const documentsCount = document.getElementById('documents-count');
        const chunksCount = document.getElementById('chunks-count');
        const systemStats = document.getElementById('system-stats');

        if (documentsCount) documentsCount.textContent = this.systemStats.documents;
        if (chunksCount) chunksCount.textContent = this.systemStats.chunks;

        if (systemStats && (this.systemStats.documents > 0 || this.systemStats.chunks > 0)) {
            systemStats.classList.remove('hidden');
        }
    }

    /**
     * Load documents list
     */
    async loadDocuments(force = false) {
        if (this.documents.length > 0 && !force) return;

        this.showLoadingState('documents');

        try {
            const response = await this.apiRequest(
                `/documents?page=${this.currentPage}&page_size=${this.documentsPerPage}`
            );

            this.documents = response.documents || [];
            this.updateDocumentsList();
            this.updatePagination(response);

        } catch (error) {
            console.error('Failed to load documents:', error);
            this.showToast('error', 'Load Error', 'Failed to load documents. Please try again.');
            this.showEmptyState('documents');
        }
    }

    /**
     * Search documents
     */
    async searchDocuments(query, options = {}) {
        const searchOptions = {
            limit: 10,
            min_score: 0.0,
            include_chunks: true,
            ...options
        };

        this.showLoadingState('search-results');

        try {
            const response = await this.apiRequest('/search', {
                method: 'POST',
                body: JSON.stringify({
                    query,
                    ...searchOptions
                })
            });

            this.searchResults = response.results || [];
            this.searchQuery = query;

            if (this.components.searchInterface) {
                this.components.searchInterface.updateResults(this.searchResults);
            }

        } catch (error) {
            console.error('Search failed:', error);
            this.showToast('error', 'Search Error', 'Failed to search documents. Please try again.');
            this.showEmptyState('search-results');
        }
    }

    /**
     * Get document details
     */
    async getDocumentDetail(documentId, includeChunks = true) {
        try {
            const response = await this.apiRequest(
                `/documents/${documentId}?include_chunks=${includeChunks}`
            );
            return response;
        } catch (error) {
            console.error('Failed to get document detail:', error);
            throw error;
        }
    }

    /**
     * Remove document
     */
    async removeDocument(documentId) {
        try {
            // Mark this as a client-initiated action
            this.markClientInitiatedAction('document_removed', documentId);

            await this.apiRequest(`/documents/${documentId}`, {
                method: 'DELETE'
            });

            // Remove from local documents array
            const removedDoc = this.documents.find(doc => doc.id === documentId);
            this.documents = this.documents.filter(doc => doc.id !== documentId);
            this.updateDocumentsList();

            // Show success notification (WebSocket duplicate will be ignored)
            const docName = removedDoc ? removedDoc.filename : 'Document';
            this.showToast('success', 'Document Removed', `${docName} has been removed successfully.`);

            return true;
        } catch (error) {
            console.error('Failed to remove document:', error);
            // Remove the action marker on error
            this.clearClientInitiatedAction('document_removed', documentId);
            throw error;
        }
    }

    /**
     * Upload document
     */
    async uploadDocument(file, metadata = {}) {
        const formData = new FormData();
        formData.append('file', file);

        if (Object.keys(metadata).length > 0) {
            formData.append('metadata', JSON.stringify(metadata));
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/documents/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Document upload failed:', error);
            throw error;
        }
    }

    /**
     * Add document by path
     */
    async addDocumentByPath(path, metadata = {}) {
        try {
            const response = await this.apiRequest('/documents/add-path', {
                method: 'POST',
                body: JSON.stringify({ path, metadata })
            });
            return response;
        } catch (error) {
            console.error('Failed to add document by path:', error);
            throw error;
        }
    }

    /**
     * Event Handlers for WebSocket Messages
     */

    handleDocumentAdded(data) {
        // Reload documents if we're on the documents view
        if (this.currentView === 'documents') {
            this.loadDocuments(true);
        }

        // Update system stats
        this.systemStats.documents++;
        this.updateSystemStats();
    }

    handleDocumentRemoved(data) {
        // Remove from local documents array
        this.documents = this.documents.filter(doc => doc.document_id !== data.document_id);
        this.updateDocumentsList();

        // Update system stats
        this.systemStats.documents--;
        this.updateSystemStats();
    }

    handleProcessingCompleted(data) {
        // Reload documents if we're on the documents view (document was added)
        if (this.currentView === 'documents') {
            this.loadDocuments(true);
        }

        // Update system stats
        this.systemStats.documents++;
        this.updateSystemStats();

        // Update document status in local array if it exists
        const doc = this.documents.find(d => d.id === data.document_id);
        if (doc) {
            doc.has_embeddings = true;
            doc.chunk_count = data.chunks_created;
            this.updateDocumentsList();
        }
    }

    handleProcessingFailed(data) {
        // Could update document status to show error state
        console.error('Document processing failed:', data);
    }

    handleSystemStatusChanged(data) {
        if (data.status === 'shutting_down') {
            this.showToast('warning', 'Server Maintenance', 'Server is shutting down for maintenance.');
        }
    }

    /**
     * UI Helper Methods
     */

    /**
     * Show loading state for a container
     */
    showLoadingState(containerId) {
        // Normalize container id to support both "xyz" and "xyz-container"
        const baseId = containerId.endsWith('-container') ? containerId.replace(/-container$/, '') : containerId;

        const loading = document.getElementById(`${baseId}-loading`);
        // Prefer "-list" content when present (e.g., documents-list), else fall back to baseId (e.g., search-results)
        const content = document.getElementById(`${baseId}-list`) || document.getElementById(baseId);
        const empty = document.getElementById(`${baseId}-empty`);

        if (loading) loading.classList.remove('hidden');
        if (content) content.classList.add('hidden');
        if (empty) empty.classList.add('hidden');
    }

    /**
     * Show empty state for a container
     */
    showEmptyState(containerId) {
        const baseId = containerId.endsWith('-container') ? containerId.replace(/-container$/, '') : containerId;

        const loading = document.getElementById(`${baseId}-loading`);
        const content = document.getElementById(`${baseId}-list`) || document.getElementById(baseId);
        const empty = document.getElementById(`${baseId}-empty`);

        if (loading) loading.classList.add('hidden');
        if (content) content.classList.add('hidden');
        if (empty) empty.classList.remove('hidden');
    }

    /**
     * Hide loading and show content
     */
    showContent(containerId) {
        const baseId = containerId.endsWith('-container') ? containerId.replace(/-container$/, '') : containerId;

        const loading = document.getElementById(`${baseId}-loading`);
        const content = document.getElementById(`${baseId}-list`) || document.getElementById(baseId);
        const empty = document.getElementById(`${baseId}-empty`);

        if (loading) loading.classList.add('hidden');
        if (content) content.classList.remove('hidden');
        if (empty) empty.classList.add('hidden');
    }

    /**
     * Update documents list display
     */
    updateDocumentsList() {
        const listContainer = document.getElementById('documents-list');
        const emptyState = document.getElementById('documents-empty');

        if (!listContainer) return;

        if (this.documents.length === 0) {
            this.showEmptyState('documents');
            return;
        }

        this.showContent('documents');

        // Let DocumentList component handle the rendering
        if (this.components.documentList) {
            this.components.documentList.render(this.documents);
        }
    }

    /**
     * Update pagination controls
     */
    updatePagination(response) {
        const pagination = document.getElementById('documents-pagination');
        const paginationText = document.getElementById('pagination-text');
        const prevBtn = document.getElementById('pagination-prev');
        const nextBtn = document.getElementById('pagination-next');

        if (!pagination) return;

        if (response.total_pages <= 1) {
            pagination.classList.add('hidden');
            return;
        }

        pagination.classList.remove('hidden');

        // Update pagination text
        const start = (response.page - 1) * response.page_size + 1;
        const end = Math.min(start + response.page_size - 1, response.total_count);

        if (paginationText) {
            paginationText.textContent = `Showing ${start}-${end} of ${response.total_count}`;
        }

        // Update buttons
        if (prevBtn) {
            prevBtn.disabled = !response.has_previous;
        }
        if (nextBtn) {
            nextBtn.disabled = !response.has_next;
        }

        // Generate page numbers
        this.generatePageNumbers(response);
    }

    /**
     * Generate pagination page numbers
     */
    generatePageNumbers(response) {
        const pagesContainer = document.getElementById('pagination-pages');
        if (!pagesContainer) return;

        pagesContainer.innerHTML = '';

        const currentPage = response.page;
        const totalPages = response.total_pages;
        const maxVisiblePages = 7;

        let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }

        for (let i = startPage; i <= endPage; i++) {
            const pageBtn = document.createElement('button');
            pageBtn.className = `pagination-page ${i === currentPage ? 'active' : ''}`;
            pageBtn.textContent = i;
            pageBtn.addEventListener('click', () => this.goToPage(i));
            pagesContainer.appendChild(pageBtn);
        }
    }

    /**
     * Go to specific page
     */
    async goToPage(page) {
        this.currentPage = page;
        await this.loadDocuments(true);
    }

    /**
     * Filter documents by search term
     */
    filterDocuments(searchTerm) {
        // This would typically filter the displayed documents
        // For now, we'll just store the term
        console.log('Filtering documents:', searchTerm);
    }

    /**
     * Theme Management
     */

    /**
     * Apply current theme
     */
    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);

        const lightIcon = document.querySelector('.theme-icon-light');
        const darkIcon = document.querySelector('.theme-icon-dark');

        if (lightIcon && darkIcon) {
            if (this.theme === 'dark') {
                lightIcon.classList.add('hidden');
                darkIcon.classList.remove('hidden');
            } else {
                lightIcon.classList.remove('hidden');
                darkIcon.classList.add('hidden');
            }
        }
    }

    /**
     * Toggle theme between light and dark
     */
    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', this.theme);
        this.applyTheme();
    }

    /**
     * Toast Notification System
     */

    /**
     * Show toast notification
     */
    showToast(type, title, message, duration = 5000) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                ${this.getToastIcon(type)}
            </svg>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <p class="toast-message">${message}</p>
            </div>
            <button class="toast-close">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;

        // Add close functionality
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });

        toastContainer.appendChild(toast);

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, duration);
        }
    }

    /**
     * Get SVG icon for toast type
     */
    getToastIcon(type) {
        const icons = {
            success: '<polyline points="20,6 9,17 4,12"/>',
            error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
            warning: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
            info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
        };
        return icons[type] || icons.info;
    }

    /**
     * Confirmation Modal
     */

    /**
     * Show confirmation modal
     */
    showConfirmModal(title, message, onConfirm, onCancel = null) {
        const modal = document.getElementById('confirm-modal');
        const titleEl = document.getElementById('confirm-title');
        const messageEl = document.getElementById('confirm-message');
        const confirmBtn = document.getElementById('confirm-ok');

        if (!modal || !titleEl || !messageEl || !confirmBtn) return;

        titleEl.textContent = title;
        messageEl.textContent = message;

        // Remove existing event listeners
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);

        // Add new event listener
        newConfirmBtn.addEventListener('click', () => {
            modal.classList.remove('active');
            modal.classList.add('hidden');
            if (onConfirm) onConfirm();
        });

        // Show modal
        modal.classList.remove('hidden');
        modal.classList.add('active');
    }

    /**
     * Utility Methods
     */

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    /**
     * Format date
     */
    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Mark an action as client-initiated to prevent duplicate WebSocket notifications
     */
    markClientInitiatedAction(actionType, identifier = null) {
        const key = identifier ? `${actionType}:${identifier}` : actionType;
        this.clientInitiatedActions.set(key, Date.now());

        // Auto-cleanup after ignore window expires
        setTimeout(() => {
            this.clientInitiatedActions.delete(key);
        }, this.actionIgnoreWindow);
    }

    /**
     * Check if an action was recently initiated by this client
     */
    isClientInitiatedAction(actionType, identifier = null) {
        const key = identifier ? `${actionType}:${identifier}` : actionType;
        const timestamp = this.clientInitiatedActions.get(key);

        if (!timestamp) return false;

        const elapsed = Date.now() - timestamp;
        const isRecent = elapsed < this.actionIgnoreWindow;

        // Clean up expired actions
        if (!isRecent) {
            this.clientInitiatedActions.delete(key);
        }

        return isRecent;
    }

    /**
     * Clear a client-initiated action marker
     */
    clearClientInitiatedAction(actionType, identifier = null) {
        const key = identifier ? `${actionType}:${identifier}` : actionType;
        this.clientInitiatedActions.delete(key);
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.websocket) {
            this.websocket.close();
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.pdfkbApp = new PDFKnowledgebaseApp();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PDFKnowledgebaseApp;
}
