/**
 * DocumentDetail Component
 * Handles detailed document viewing with chunks, metadata, and content
 */

class DocumentDetail {
    constructor(app) {
        this.app = app;
        this.container = document.getElementById('document-detail-container');
        this.currentDocument = null;
        this.currentOptions = {};
        this.expandedChunks = new Set();

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        // Register with app
        this.app.components.documentDetail = this;

        console.log('DocumentDetail component initialized');
    }

    /**
     * Render document detail view
     */
    render(documentDetail, options = {}) {
        if (!this.container) return;

        this.currentDocument = documentDetail.document;
        this.currentOptions = options;

        const document = documentDetail.document;
        const chunks = documentDetail.chunks || [];
        const metadata = documentDetail.metadata || {};

        this.container.innerHTML = `
            <div class="document-detail">
                <!-- Document Header -->
                <div class="document-detail-header">
                    <div class="document-detail-title-section">
                        <h1 class="document-detail-title">${this.escapeHtml(document.title || document.filename)}</h1>
                        <p class="document-detail-path">${this.escapeHtml(document.path)}</p>
                        <div class="document-detail-status ${this.getStatusClass(document)}">
                            ${this.getStatusIcon(document)}
                            <span>${this.getDocumentStatus(document)}</span>
                        </div>
                    </div>

                    <div class="document-detail-actions">
                        <button class="btn btn-secondary" id="preview-document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                                <circle cx="12" cy="12" r="3"/>
                            </svg>
                            Preview Content
                        </button>
                        <button class="btn btn-secondary" id="export-document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="7,10 12,15 17,10"/>
                                <line x1="12" y1="15" x2="12" y2="3"/>
                            </svg>
                            Export
                        </button>
                        <button class="btn btn-danger" id="remove-document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3,6 5,6 21,6"/>
                                <path d="m19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"/>
                            </svg>
                            Remove
                        </button>
                    </div>
                </div>

                <!-- Document Stats -->
                <div class="document-stats">
                    <div class="document-stat">
                        <div class="stat-value">${this.app.formatFileSize(document.file_size)}</div>
                        <div class="stat-label">File Size</div>
                    </div>
                    <div class="document-stat">
                        <div class="stat-value">${document.page_count}</div>
                        <div class="stat-label">Pages</div>
                    </div>
                    <div class="document-stat">
                        <div class="stat-value">${document.chunk_count}</div>
                        <div class="stat-label">Chunks</div>
                    </div>
                    <div class="document-stat">
                        <div class="stat-value">${this.app.formatDate(document.added_at)}</div>
                        <div class="stat-label">Added</div>
                    </div>
                    ${document.updated_at ? `
                    <div class="document-stat">
                        <div class="stat-value">${this.app.formatDate(document.updated_at)}</div>
                        <div class="stat-label">Updated</div>
                    </div>
                    ` : ''}
                </div>

                <!-- Document Metadata (if available) -->
                ${this.renderMetadata(metadata)}

                <!-- Document Chunks -->
                <div class="document-chunks-section">
                    <div class="section-header">
                        <h2>Document Chunks (${chunks.length})</h2>
                        <div class="chunk-controls">
                            <button class="btn btn-secondary btn-sm" id="expand-all-chunks">Expand All</button>
                            <button class="btn btn-secondary btn-sm" id="collapse-all-chunks">Collapse All</button>
                            ${options.searchQuery ? `
                            <div class="search-highlight-info">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="11" cy="11" r="8"/>
                                    <path d="m21 21-4.35-4.35"/>
                                </svg>
                                Highlighting: "${this.escapeHtml(options.searchQuery)}"
                            </div>
                            ` : ''}
                        </div>
                    </div>

                    <div class="document-chunks">
                        ${this.renderChunks(chunks, options)}
                    </div>
                </div>
            </div>
        `;

        this.setupEventListeners();

        // Scroll to highlighted chunk if specified
        if (options.highlightChunk) {
            setTimeout(() => {
                this.scrollToChunk(options.highlightChunk);
            }, 100);
        }
    }

    /**
     * Render document metadata
     */
    renderMetadata(metadata) {
        const metadataEntries = Object.entries(metadata);
        if (metadataEntries.length === 0) {
            return '';
        }

        return `
            <div class="document-metadata-section">
                <h2>Metadata</h2>
                <div class="document-metadata">
                    ${metadataEntries.map(([key, value]) => `
                        <div class="metadata-item">
                            <div class="metadata-key">${this.escapeHtml(key)}</div>
                            <div class="metadata-value">${this.escapeHtml(String(value))}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Render document chunks
     */
    renderChunks(chunks, options = {}) {
        if (chunks.length === 0) {
            return `
                <div class="empty-state">
                    <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14,2 14,8 20,8"/>
                    </svg>
                    <h3>No chunks available</h3>
                    <p>This document hasn't been processed into chunks yet.</p>
                </div>
            `;
        }

        return chunks.map((chunk, index) => {
            const isHighlighted = options.highlightChunk === chunk.id;
            const isExpanded = this.expandedChunks.has(chunk.id) || isHighlighted;

            let chunkText = chunk.text;
            if (options.searchQuery) {
                chunkText = this.highlightSearchTerms(chunk.text, options.searchQuery);
            } else {
                chunkText = this.escapeHtml(chunk.text);
            }

            return `
                <div class="document-chunk ${isHighlighted ? 'highlighted' : ''}" data-chunk-id="${chunk.id}">
                    <div class="chunk-header" data-chunk-id="${chunk.id}">
                        <div class="chunk-info">
                            <span class="chunk-number">Chunk ${chunk.chunk_index + 1}</span>
                            ${chunk.page_number ? `<span class="chunk-page">Page ${chunk.page_number}</span>` : ''}
                            <span class="chunk-length">${chunk.text.length} chars</span>
                        </div>
                        <button class="chunk-toggle" data-chunk-id="${chunk.id}">
                            <svg class="chunk-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="${isExpanded ? '6,9 12,15 18,9' : '9,18 15,12 9,6'}"/>
                            </svg>
                        </button>
                    </div>

                    <div class="chunk-content ${isExpanded ? 'expanded' : 'collapsed'}">
                        <div class="chunk-text">${chunkText}</div>
                        ${this.renderChunkMetadata(chunk.metadata)}
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Render chunk metadata
     */
    renderChunkMetadata(metadata) {
        const metadataEntries = Object.entries(metadata || {});
        if (metadataEntries.length === 0) {
            return '';
        }

        return `
            <div class="chunk-metadata">
                <h4>Chunk Metadata</h4>
                <div class="chunk-metadata-items">
                    ${metadataEntries.map(([key, value]) => `
                        <div class="chunk-metadata-item">
                            <span class="chunk-metadata-key">${this.escapeHtml(key)}:</span>
                            <span class="chunk-metadata-value">${this.escapeHtml(String(value))}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Document actions
        const previewBtn = document.getElementById('preview-document');
        const exportBtn = document.getElementById('export-document');
        const removeBtn = document.getElementById('remove-document');

        if (previewBtn) {
            previewBtn.addEventListener('click', () => this.previewDocument());
        }

        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportDocument());
        }

        if (removeBtn) {
            removeBtn.addEventListener('click', () => this.confirmRemoveDocument());
        }

        // Chunk controls
        const expandAllBtn = document.getElementById('expand-all-chunks');
        const collapseAllBtn = document.getElementById('collapse-all-chunks');

        if (expandAllBtn) {
            expandAllBtn.addEventListener('click', () => this.expandAllChunks());
        }

        if (collapseAllBtn) {
            collapseAllBtn.addEventListener('click', () => this.collapseAllChunks());
        }

        // Chunk toggles
        const chunkToggles = this.container.querySelectorAll('.chunk-toggle');
        chunkToggles.forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                const chunkId = toggle.dataset.chunkId;
                this.toggleChunk(chunkId);
            });
        });

        // Chunk headers (clickable)
        const chunkHeaders = this.container.querySelectorAll('.chunk-header');
        chunkHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const chunkId = header.dataset.chunkId;
                this.toggleChunk(chunkId);
            });
        });
    }

    /**
     * Preview document content
     */
    async previewDocument() {
        if (!this.currentDocument) return;

        try {
            const previewResponse = await this.app.apiRequest(`/documents/${this.currentDocument.id}/preview`);
            this.showPreviewModal(this.currentDocument, previewResponse.content);

        } catch (error) {
            console.error('Failed to load document preview:', error);
            this.app.showToast('error', 'Error', 'Failed to load document preview');
        }
    }

    /**
     * Show preview modal
     */
    showPreviewModal(doc, content) {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content" style="max-width: 900px; max-height: 90vh;">
                <div class="modal-header">
                    <h3>Full Content: ${this.escapeHtml(doc.filename)}</h3>
                    <button class="modal-close preview-modal-close">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body" style="overflow-y: auto; max-height: 70vh;">
                    <pre style="white-space: pre-wrap; font-family: inherit; font-size: 0.9rem; line-height: 1.6; color: var(--color-text-secondary);">${this.escapeHtml(content)}</pre>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary preview-modal-close">Close</button>
                    <button class="btn btn-primary" onclick="window.pdfkbApp.components.documentDetail.exportDocument()">
                        Export Content
                    </button>
                </div>
            </div>
        `;

        // Add close handlers
        const closeButtons = modal.querySelectorAll('.preview-modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');

        const closeModal = () => modal.remove();
        closeButtons.forEach(btn => btn.addEventListener('click', closeModal));
        backdrop.addEventListener('click', closeModal);

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
     * Export document
     */
    async exportDocument() {
        if (!this.currentDocument) return;

        try {
            const previewResponse = await this.app.apiRequest(`/documents/${this.currentDocument.id}/preview`);

            // Create text file content
            const content = `Document: ${this.currentDocument.filename}
Path: ${this.currentDocument.path}
Added: ${this.app.formatDate(this.currentDocument.added_at)}
Pages: ${this.currentDocument.page_count}
Chunks: ${this.currentDocument.chunk_count}

Content:
${previewResponse.content}`;

            // Download as text file
            this.downloadTextFile(content, `${this.currentDocument.filename}.txt`);

            this.app.showToast('success', 'Export Complete', 'Document exported successfully');

        } catch (error) {
            console.error('Failed to export document:', error);
            this.app.showToast('error', 'Export Failed', 'Failed to export document');
        }
    }

    /**
     * Download text file
     */
    downloadTextFile(content, filename) {
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const link = document.createElement('a');

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    /**
     * Confirm document removal
     */
    confirmRemoveDocument() {
        if (!this.currentDocument) return;

        const title = 'Remove Document';
        const message = `Are you sure you want to remove "${this.currentDocument.filename}" from the knowledgebase? This action cannot be undone.`;

        this.app.showConfirmModal(title, message, async () => {
            await this.removeDocument();
        });
    }

    /**
     * Remove document
     */
    async removeDocument() {
        if (!this.currentDocument) return;

        try {
            await this.app.removeDocument(this.currentDocument.id);

            // Navigate back to documents list
            this.app.navigateTo('documents');

            // Note: Don't show toast here - the app.removeDocument() handles the client-initiated action marking
            // and prevents duplicate notifications from WebSocket events

        } catch (error) {
            console.error('Failed to remove document:', error);
            this.app.showToast('error', 'Remove Failed', 'Failed to remove document. Please try again.');
        }
    }

    /**
     * Toggle chunk expansion
     */
    toggleChunk(chunkId) {
        const chunkEl = this.container.querySelector(`[data-chunk-id="${chunkId}"].document-chunk`);
        const contentEl = chunkEl?.querySelector('.chunk-content');
        const toggleIcon = chunkEl?.querySelector('.chunk-toggle-icon polyline');

        if (!chunkEl || !contentEl) return;

        const isExpanded = this.expandedChunks.has(chunkId);

        if (isExpanded) {
            this.expandedChunks.delete(chunkId);
            contentEl.classList.remove('expanded');
            contentEl.classList.add('collapsed');
            if (toggleIcon) {
                toggleIcon.setAttribute('points', '9,18 15,12 9,6');
            }
        } else {
            this.expandedChunks.add(chunkId);
            contentEl.classList.remove('collapsed');
            contentEl.classList.add('expanded');
            if (toggleIcon) {
                toggleIcon.setAttribute('points', '6,9 12,15 18,9');
            }
        }
    }

    /**
     * Expand all chunks
     */
    expandAllChunks() {
        const chunkEls = this.container.querySelectorAll('.document-chunk');
        chunkEls.forEach(chunkEl => {
            const chunkId = chunkEl.dataset.chunkId;
            if (!this.expandedChunks.has(chunkId)) {
                this.expandedChunks.add(chunkId);
                const contentEl = chunkEl.querySelector('.chunk-content');
                const toggleIcon = chunkEl.querySelector('.chunk-toggle-icon polyline');

                if (contentEl) {
                    contentEl.classList.remove('collapsed');
                    contentEl.classList.add('expanded');
                }
                if (toggleIcon) {
                    toggleIcon.setAttribute('points', '6,9 12,15 18,9');
                }
            }
        });
    }

    /**
     * Collapse all chunks
     */
    collapseAllChunks() {
        const chunkEls = this.container.querySelectorAll('.document-chunk');
        chunkEls.forEach(chunkEl => {
            const chunkId = chunkEl.dataset.chunkId;
            if (this.expandedChunks.has(chunkId)) {
                this.expandedChunks.delete(chunkId);
                const contentEl = chunkEl.querySelector('.chunk-content');
                const toggleIcon = chunkEl.querySelector('.chunk-toggle-icon polyline');

                if (contentEl) {
                    contentEl.classList.remove('expanded');
                    contentEl.classList.add('collapsed');
                }
                if (toggleIcon) {
                    toggleIcon.setAttribute('points', '9,18 15,12 9,6');
                }
            }
        });
    }

    /**
     * Scroll to specific chunk
     */
    scrollToChunk(chunkId) {
        const chunkEl = this.container.querySelector(`[data-chunk-id="${chunkId}"].document-chunk`);
        if (chunkEl) {
            chunkEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Expand the chunk if it's not already expanded
            if (!this.expandedChunks.has(chunkId)) {
                this.toggleChunk(chunkId);
            }
        }
    }

    /**
     * Highlight search terms in text
     */
    highlightSearchTerms(text, query) {
        if (!query || !text) return this.escapeHtml(text);

        const terms = query.toLowerCase().split(/\s+/).filter(term => term.length > 0);
        let highlightedText = this.escapeHtml(text);

        terms.forEach(term => {
            const regex = new RegExp(`(${this.escapeRegex(term)})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        });

        return highlightedText;
    }

    /**
     * Get document status
     */
    getDocumentStatus(document) {
        if (!document.has_embeddings) return 'Processing';
        if (document.chunk_count === 0) return 'Error';
        return 'Indexed';
    }

    /**
     * Get status CSS class
     */
    getStatusClass(document) {
        return this.getDocumentStatus(document).toLowerCase();
    }

    /**
     * Get status icon
     */
    getStatusIcon(document) {
        const status = this.getDocumentStatus(document);
        const icons = {
            'Indexed': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20,6 9,17 4,12"/></svg>',
            'Processing': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9c1.08 0 2.13.19 3.1.54"/></svg>',
            'Error': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        };
        return icons[status] || icons['Error'];
    }

    /**
     * Clear current document
     */
    clear() {
        this.currentDocument = null;
        this.currentOptions = {};
        this.expandedChunks.clear();
        if (this.container) {
            this.container.innerHTML = '';
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

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}

// Initialize when app is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for app to be initialized
    const initComponent = () => {
        if (window.pdfkbApp) {
            new DocumentDetail(window.pdfkbApp);
        } else {
            setTimeout(initComponent, 100);
        }
    };
    initComponent();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DocumentDetail;
}
