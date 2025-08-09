/**
 * DocumentList Component
 * Handles document listing, filtering, and management
 */

class DocumentList {
    constructor(app) {
        this.app = app;
        this.container = document.getElementById('documents-list');
        this.searchInput = document.getElementById('documents-search');
        this.refreshBtn = document.getElementById('refresh-documents');
        this.paginationContainer = document.getElementById('documents-pagination');

        this.filteredDocuments = [];
        this.currentFilter = '';

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        this.setupEventListeners();

        // Register with app
        this.app.components.documentList = this;

        console.log('DocumentList component initialized');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Search/filter input
        if (this.searchInput) {
            this.searchInput.addEventListener('input', this.app.debounce((e) => {
                this.filterDocuments(e.target.value);
            }, 300));
        }

        // Pagination controls
        const prevBtn = document.getElementById('pagination-prev');
        const nextBtn = document.getElementById('pagination-next');

        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.goToPreviousPage());
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.goToNextPage());
        }
    }

    /**
     * Render documents list
     */
    render(documents) {
        if (!this.container) return;

        this.filteredDocuments = this.currentFilter
            ? this.filterDocumentsByTerm(documents, this.currentFilter)
            : documents;

        if (this.filteredDocuments.length === 0) {
            this.renderEmptyState();
            return;
        }

        this.container.innerHTML = '';

        this.filteredDocuments.forEach(document => {
            const documentCard = this.createDocumentCard(document);
            this.container.appendChild(documentCard);
        });
    }

    /**
     * Create a document card element
     */
    createDocumentCard(doc) {
        const card = document.createElement('div');
        card.className = 'document-card';
        card.setAttribute('data-document-id', doc.id);

        // Status indicator
        const status = this.getDocumentStatus(doc);
        const statusClass = status.toLowerCase().replace(/\s+/g, '-');

        card.innerHTML = `
            <div class="document-header">
                <div>
                    <h3 class="document-title">${this.escapeHtml(doc.title || doc.filename)}</h3>
                    <p class="document-path">${this.escapeHtml(doc.path)}</p>
                </div>
                <div class="document-status ${statusClass}">
                    ${this.getStatusIcon(status)}
                    <span>${status}</span>
                </div>
            </div>

            <div class="document-meta">
                <div class="document-meta-item">
                    <div class="document-meta-label">File Size</div>
                    <div class="document-meta-value">${this.app.formatFileSize(doc.file_size)}</div>
                </div>
                <div class="document-meta-item">
                    <div class="document-meta-label">Pages</div>
                    <div class="document-meta-value">${doc.page_count}</div>
                </div>
                <div class="document-meta-item">
                    <div class="document-meta-label">Chunks</div>
                    <div class="document-meta-value">${doc.chunk_count}</div>
                </div>
                <div class="document-meta-item">
                    <div class="document-meta-label">Added</div>
                    <div class="document-meta-value">${this.app.formatDate(doc.added_at)}</div>
                </div>
            </div>

            <div class="document-actions">
                <button class="document-action" data-action="view" data-document-id="${doc.id}">
                    View Details
                </button>
                <button class="document-action" data-action="preview" data-document-id="${doc.id}">
                    Preview
                </button>
                <button class="document-action danger" data-action="remove" data-document-id="${doc.id}">
                    Remove
                </button>
            </div>
        `;

        // Add click handler for the card (navigate to detail view)
        card.addEventListener('click', (e) => {
            // Don't trigger if clicking on action buttons
            if (e.target.matches('.document-action') || e.target.closest('.document-action')) {
                return;
            }
            this.viewDocumentDetail(doc);
        });

        // Add action button handlers
        this.setupCardActions(card);

        return card;
    }

    /**
     * Set up action button handlers for a document card
     */
    setupCardActions(card) {
        const actions = card.querySelectorAll('.document-action');

        actions.forEach(button => {
            button.addEventListener('click', async (e) => {
                e.stopPropagation(); // Prevent card click

                const action = button.dataset.action;
                const documentId = button.dataset.documentId;
                const document = this.app.documents.find(d => d.id === documentId);

                if (!document) return;

                switch (action) {
                    case 'view':
                        this.viewDocumentDetail(document);
                        break;

                    case 'preview':
                        await this.previewDocument(document);
                        break;

                    case 'remove':
                        this.confirmRemoveDocument(document);
                        break;
                }
            });
        });
    }

    /**
     * View document detail
     */
    async viewDocumentDetail(doc) {
        try {
            this.app.currentDocument = doc;

            // Load full document details if not already loaded
            const detailResponse = await this.app.getDocumentDetail(doc.id, true);

            if (this.app.components.documentDetail) {
                this.app.components.documentDetail.render(detailResponse);
            }

            this.app.navigateTo('document-detail');

        } catch (error) {
            console.error('Failed to load document detail:', error);
            this.app.showToast('error', 'Error', 'Failed to load document details');
        }
    }

    /**
     * Preview document content
     */
    async previewDocument(doc) {
        try {
            const previewResponse = await this.app.apiRequest(`/documents/${doc.id}/preview`);

            // Create a simple preview modal or panel
            this.showPreviewModal(doc, previewResponse.content);

        } catch (error) {
            console.error('Failed to load document preview:', error);
            this.app.showToast('error', 'Error', 'Failed to load document preview');
        }
    }

    /**
     * Show preview modal
     */
    showPreviewModal(doc, content) {
        // Create modal HTML
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content" style="max-width: 800px; max-height: 80vh;">
                <div class="modal-header">
                    <h3>Preview: ${this.escapeHtml(doc.filename)}</h3>
                    <button class="modal-close preview-modal-close">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body" style="overflow-y: auto; max-height: 60vh;">
                    <pre style="white-space: pre-wrap; font-family: inherit; font-size: 0.9rem; line-height: 1.6; color: var(--color-text-secondary);">${this.escapeHtml(content)}</pre>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary preview-modal-close">Close</button>
                    <button class="btn btn-primary" onclick="window.pdfkbApp.components.documentList.viewDocumentDetail(window.pdfkbApp.documents.find(d => d.id === '${doc.id}'))">
                        View Full Details
                    </button>
                </div>
            </div>
        `;

        // Add close handlers
        const closeButtons = modal.querySelectorAll('.preview-modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');

        const closeModal = () => {
            modal.remove();
        };

        closeButtons.forEach(btn => btn.addEventListener('click', closeModal));
        backdrop.addEventListener('click', closeModal);

        // Add to document
        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }

    /**
     * Confirm document removal
     */
    confirmRemoveDocument(doc) {
        const title = 'Remove Document';
        const message = `Are you sure you want to remove "${doc.filename}" from the knowledgebase? This action cannot be undone.`;

        this.app.showConfirmModal(title, message, async () => {
            await this.removeDocument(doc);
        });
    }

    /**
     * Remove document
     */
    async removeDocument(doc) {
        try {
            await this.app.removeDocument(doc.id);
            // Note: Don't show toast here - the app.removeDocument() handles the client-initiated action marking
            // and prevents duplicate notifications from WebSocket events

        } catch (error) {
            console.error('Failed to remove document:', error);
            this.app.showToast('error', 'Remove Failed', 'Failed to remove document. Please try again.');
        }
    }

    /**
     * Filter documents by search term
     */
    filterDocuments(searchTerm) {
        this.currentFilter = searchTerm.toLowerCase().trim();
        this.render(this.app.documents);
    }

    /**
     * Filter documents array by search term
     */
    filterDocumentsByTerm(documents, term) {
        if (!term) return documents;

        return documents.filter(doc => {
            const searchFields = [
                doc.title,
                doc.filename,
                doc.path,
            ].filter(Boolean);

            return searchFields.some(field =>
                field.toLowerCase().includes(term)
            );
        });
    }

    /**
     * Get document processing status
     */
    getDocumentStatus(document) {
        // Use the processing_status field from the API if available
        if (document.processing_status) {
            switch (document.processing_status.toLowerCase()) {
                case 'completed':
                    return 'Indexed';
                case 'processing':
                    return 'Processing';
                case 'failed':
                    return 'Error';
                default:
                    // Fall through to legacy logic
                    break;
            }
        }

        // Legacy logic for backward compatibility
        if (!document.has_embeddings) {
            return 'Processing';
        }
        if (document.chunk_count === 0) {
            return 'Error';
        }
        return 'Indexed';
    }

    /**
     * Get status icon SVG
     */
    getStatusIcon(status) {
        const icons = {
            'Indexed': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20,6 9,17 4,12"/></svg>',
            'Processing': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9c1.08 0 2.13.19 3.1.54"/></svg>',
            'Error': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        };
        return icons[status] || icons['Error'];
    }

    /**
     * Render empty state
     */
    renderEmptyState() {
        if (!this.container) return;

        const emptyMessage = this.currentFilter
            ? `No documents match "${this.currentFilter}"`
            : 'No documents found';

        this.container.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10,9 9,9 8,9"/>
                </svg>
                <h3>${emptyMessage}</h3>
                <p>${this.currentFilter ? 'Try adjusting your search terms' : 'Upload your first PDF document to get started'}</p>
                ${!this.currentFilter ? '<button class="btn btn-primary" onclick="window.pdfkbApp.navigateTo(\'upload\')">Upload Document</button>' : ''}
            </div>
        `;
    }

    /**
     * Pagination methods
     */
    goToPreviousPage() {
        if (this.app.currentPage > 1) {
            this.app.goToPage(this.app.currentPage - 1);
        }
    }

    goToNextPage() {
        this.app.goToPage(this.app.currentPage + 1);
    }

    /**
     * Refresh documents list
     */
    async refresh() {
        await this.app.loadDocuments(true);
    }

    /**
     * Update display when documents change
     */
    onDocumentsUpdated(documents) {
        this.render(documents);
    }

    /**
     * Handle real-time document updates
     */
    onDocumentAdded(documentData) {
        // Refresh the list to show new document
        this.refresh();
    }

    onDocumentRemoved(documentId) {
        // Remove from current display
        const card = this.container?.querySelector(`[data-document-id="${documentId}"]`);
        if (card) {
            card.remove();
        }

        // Update empty state if no documents left
        if (this.app.documents.length === 0) {
            this.renderEmptyState();
        }
    }

    onDocumentUpdated(documentData) {
        // Find and update the specific document card
        const card = this.container?.querySelector(`[data-document-id="${documentData.document_id}"]`);
        if (card) {
            const document = this.app.documents.find(d => d.id === documentData.document_id);
            if (document) {
                // Update document data
                Object.assign(document, documentData);

                // Re-render the card
                const newCard = this.createDocumentCard(document);
                card.replaceWith(newCard);
            }
        }
    }

    /**
     * Utility methods
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when app is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for app to be initialized
    const initComponent = () => {
        if (window.pdfkbApp) {
            new DocumentList(window.pdfkbApp);
        } else {
            setTimeout(initComponent, 100);
        }
    };
    initComponent();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DocumentList;
}
