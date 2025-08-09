/**
 * FileUpload Component
 * Handles file upload via drag-and-drop, file picker, and path input
 */

class FileUpload {
    constructor(app) {
        this.app = app;
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.fileSelectBtn = document.getElementById('file-select');
        this.pathInput = document.getElementById('path-input');
        this.addPathBtn = document.getElementById('add-path-button');
        this.uploadQueue = document.getElementById('upload-queue');
        this.uploadList = document.getElementById('upload-list');

        this.activeUploads = new Map(); // Track active uploads
        this.supportedTypes = ['.pdf'];

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        this.setupEventListeners();

        // Register with app
        this.app.components.fileUpload = this;

        console.log('FileUpload component initialized');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Drag and drop
        if (this.dropZone) {
            this.setupDragAndDrop();
        }

        // File picker
        if (this.fileSelectBtn && this.fileInput) {
            this.fileSelectBtn.addEventListener('click', () => {
                this.fileInput.click();
            });

            this.fileInput.addEventListener('change', (e) => {
                this.handleFileSelect(e.target.files);
            });
        }

        // Add by path
        if (this.addPathBtn && this.pathInput) {
            this.addPathBtn.addEventListener('click', () => {
                this.handleAddByPath();
            });

            this.pathInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.handleAddByPath();
                }
            });
        }
    }

    /**
     * Set up drag and drop functionality
     */
    setupDragAndDrop() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => {
                this.dropZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => {
                this.dropZone.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        this.dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFileSelect(files);
        }, false);
    }

    /**
     * Prevent default drag behaviors
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Handle file selection (from drag-drop or file picker)
     */
    handleFileSelect(files) {
        const fileArray = Array.from(files);
        const validFiles = fileArray.filter(file => this.validateFile(file));

        if (validFiles.length === 0) {
            this.app.showToast('warning', 'Invalid Files', 'Please select valid PDF files.');
            return;
        }

        if (validFiles.length !== fileArray.length) {
            const invalidCount = fileArray.length - validFiles.length;
            this.app.showToast('warning', 'Some Files Skipped',
                `${invalidCount} invalid file(s) were skipped. Only PDF files are supported.`);
        }

        // Add files to upload queue
        validFiles.forEach(file => {
            this.addToUploadQueue(file);
        });

        // Start uploading
        this.processUploadQueue();
    }

    /**
     * Validate file
     */
    validateFile(file) {
        // Check file type
        if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
            return false;
        }

        // Check file size (optional: set a reasonable limit, e.g., 100MB)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            this.app.showToast('warning', 'File Too Large',
                `${file.name} is too large. Maximum file size is 100MB.`);
            return false;
        }

        return true;
    }

    /**
     * Add file to upload queue
     */
    addToUploadQueue(file) {
        const uploadId = this.generateUploadId();

        const uploadItem = {
            id: uploadId,
            file: file,
            status: 'queued',
            progress: 0,
            error: null
        };

        this.activeUploads.set(uploadId, uploadItem);
        this.renderUploadQueue();
    }

    /**
     * Handle add by path
     */
    async handleAddByPath() {
        const path = this.pathInput?.value?.trim();

        if (!path) {
            this.app.showToast('warning', 'Path Required', 'Please enter a file path.');
            return;
        }

        if (!path.toLowerCase().endsWith('.pdf')) {
            this.app.showToast('warning', 'Invalid File Type', 'Only PDF files are supported.');
            return;
        }

        const uploadId = this.generateUploadId();
        const uploadItem = {
            id: uploadId,
            path: path,
            status: 'uploading',
            progress: 0,
            error: null
        };

        this.activeUploads.set(uploadId, uploadItem);
        this.renderUploadQueue();

        try {
            this.updateUploadProgress(uploadId, 'uploading', 50);

            const result = await this.app.addDocumentByPath(path);

            if (result.success) {
                this.updateUploadProgress(uploadId, 'completed', 100);
                this.app.showToast('success', 'Upload Success',
                    `${result.filename} has been added successfully.`);

                // Clear path input
                if (this.pathInput) {
                    this.pathInput.value = '';
                }

                // Remove from queue after delay
                setTimeout(() => {
                    this.removeFromQueue(uploadId);
                }, 3000);

            } else {
                this.updateUploadProgress(uploadId, 'error', 0, result.error);
                this.app.showToast('error', 'Upload Failed',
                    `Failed to add ${path}: ${result.error || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('Add by path failed:', error);
            this.updateUploadProgress(uploadId, 'error', 0, error.message);
            this.app.showToast('error', 'Upload Failed',
                `Failed to add ${path}: ${error.message}`);
        }
    }

    /**
     * Process upload queue
     */
    async processUploadQueue() {
        const queuedUploads = Array.from(this.activeUploads.values())
            .filter(upload => upload.status === 'queued' && upload.file);

        if (queuedUploads.length === 0) return;

        // Process uploads one at a time to avoid overwhelming the server
        for (const upload of queuedUploads) {
            await this.uploadFile(upload);
        }
    }

    /**
     * Upload a single file
     */
    async uploadFile(uploadItem) {
        const { id, file } = uploadItem;

        try {
            this.updateUploadProgress(id, 'uploading', 10);

            const result = await this.app.uploadDocument(file);

            if (result.success) {
                this.updateUploadProgress(id, 'completed', 100);
                this.app.showToast('success', 'Upload Success',
                    `${file.name} has been uploaded successfully.`);

                // Remove from queue after delay
                setTimeout(() => {
                    this.removeFromQueue(id);
                }, 3000);

            } else {
                this.updateUploadProgress(id, 'error', 0, result.error);
                this.app.showToast('error', 'Upload Failed',
                    `Failed to upload ${file.name}: ${result.error || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('File upload failed:', error);
            this.updateUploadProgress(id, 'error', 0, error.message);
            this.app.showToast('error', 'Upload Failed',
                `Failed to upload ${file.name}: ${error.message}`);
        }
    }

    /**
     * Update upload progress
     */
    updateUploadProgress(uploadId, status, progress, error = null) {
        const uploadItem = this.activeUploads.get(uploadId);
        if (!uploadItem) return;

        uploadItem.status = status;
        uploadItem.progress = progress;
        uploadItem.error = error;

        this.renderUploadQueue();
    }

    /**
     * Remove item from upload queue
     */
    removeFromQueue(uploadId) {
        this.activeUploads.delete(uploadId);
        this.renderUploadQueue();
    }

    /**
     * Render upload queue
     */
    renderUploadQueue() {
        if (!this.uploadList) return;

        const uploads = Array.from(this.activeUploads.values());

        if (uploads.length === 0) {
            if (this.uploadQueue) {
                this.uploadQueue.classList.add('hidden');
            }
            return;
        }

        if (this.uploadQueue) {
            this.uploadQueue.classList.remove('hidden');
        }

        this.uploadList.innerHTML = uploads.map(upload =>
            this.createUploadItemHTML(upload)
        ).join('');

        this.setupUploadItemListeners();
    }

    /**
     * Create HTML for upload item
     */
    createUploadItemHTML(upload) {
        const name = upload.file ? upload.file.name : upload.path.split('/').pop();
        const size = upload.file ? this.app.formatFileSize(upload.file.size) : 'Unknown size';

        let statusIcon = '';
        let statusClass = '';

        switch (upload.status) {
            case 'queued':
                statusIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12,6 12,12 16,14"/>
                </svg>`;
                statusClass = 'queued';
                break;
            case 'uploading':
                statusIcon = `<svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9c1.08 0 2.13.19 3.1.54"/>
                </svg>`;
                statusClass = 'uploading';
                break;
            case 'completed':
                statusIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20,6 9,17 4,12"/>
                </svg>`;
                statusClass = 'completed';
                break;
            case 'error':
                statusIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>`;
                statusClass = 'error';
                break;
        }

        return `
            <div class="upload-item ${statusClass}" data-upload-id="${upload.id}">
                <div class="upload-item-info">
                    <div class="upload-item-name" title="${this.escapeHtml(name)}">
                        ${this.escapeHtml(name)}
                    </div>
                    <div class="upload-item-size">${size}</div>
                    ${upload.error ? `<div class="upload-item-error">${this.escapeHtml(upload.error)}</div>` : ''}
                </div>

                <div class="upload-progress">
                    <div class="upload-progress-bar" style="width: ${upload.progress}%"></div>
                </div>

                <div class="upload-status">
                    ${statusIcon}
                    <span class="upload-status-text">${this.getStatusText(upload)}</span>
                </div>

                <div class="upload-actions">
                    ${upload.status === 'error' || upload.status === 'completed' ?
                        `<button class="upload-action-remove" data-upload-id="${upload.id}" title="Remove">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>` : ''
                    }
                    ${upload.status === 'queued' || upload.status === 'uploading' ?
                        `<button class="upload-action-cancel" data-upload-id="${upload.id}" title="Cancel">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>` : ''
                    }
                </div>
            </div>
        `;
    }

    /**
     * Set up upload item listeners
     */
    setupUploadItemListeners() {
        // Remove buttons
        const removeButtons = this.uploadList.querySelectorAll('.upload-action-remove');
        removeButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const uploadId = button.dataset.uploadId;
                this.removeFromQueue(uploadId);
            });
        });

        // Cancel buttons
        const cancelButtons = this.uploadList.querySelectorAll('.upload-action-cancel');
        cancelButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const uploadId = button.dataset.uploadId;
                this.cancelUpload(uploadId);
            });
        });
    }

    /**
     * Cancel upload
     */
    cancelUpload(uploadId) {
        const uploadItem = this.activeUploads.get(uploadId);
        if (!uploadItem) return;

        // Cancel the upload if it's in progress
        uploadItem.status = 'cancelled';
        this.removeFromQueue(uploadId);

        this.app.showToast('info', 'Upload Cancelled',
            `Upload of ${uploadItem.file?.name || uploadItem.path} was cancelled.`);
    }

    /**
     * Get status text for upload
     */
    getStatusText(upload) {
        switch (upload.status) {
            case 'queued':
                return 'Queued';
            case 'uploading':
                return `${upload.progress}%`;
            case 'completed':
                return 'Complete';
            case 'error':
                return 'Failed';
            case 'cancelled':
                return 'Cancelled';
            default:
                return 'Unknown';
        }
    }

    /**
     * Generate unique upload ID
     */
    generateUploadId() {
        return 'upload_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Clear completed uploads
     */
    clearCompleted() {
        const completed = Array.from(this.activeUploads.entries())
            .filter(([id, upload]) => upload.status === 'completed');

        completed.forEach(([id]) => {
            this.activeUploads.delete(id);
        });

        this.renderUploadQueue();
    }

    /**
     * Clear all uploads
     */
    clearAll() {
        this.activeUploads.clear();
        this.renderUploadQueue();
    }

    /**
     * Handle WebSocket upload events
     */
    onUploadStarted(data) {
        // Find upload by filename and update status
        const uploads = Array.from(this.activeUploads.values());
        const upload = uploads.find(u =>
            (u.file && u.file.name === data.filename) ||
            (u.path && u.path.endsWith(data.filename))
        );

        if (upload) {
            this.updateUploadProgress(upload.id, 'uploading', 25);
        }
    }

    onUploadCompleted(data) {
        const uploads = Array.from(this.activeUploads.values());
        const upload = uploads.find(u =>
            (u.file && u.file.name === data.filename) ||
            (u.path && u.path.endsWith(data.filename))
        );

        if (upload) {
            this.updateUploadProgress(upload.id, 'completed', 100);

            // Auto-remove after delay
            setTimeout(() => {
                this.removeFromQueue(upload.id);
            }, 3000);
        }
    }

    onUploadFailed(data) {
        const uploads = Array.from(this.activeUploads.values());
        const upload = uploads.find(u =>
            (u.file && u.file.name === data.filename) ||
            (u.path && u.path.endsWith(data.filename))
        );

        if (upload) {
            this.updateUploadProgress(upload.id, 'error', 0, data.error);
        }
    }

    /**
     * Reset component state
     */
    reset() {
        this.clearAll();

        if (this.pathInput) {
            this.pathInput.value = '';
        }

        if (this.fileInput) {
            this.fileInput.value = '';
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
            new FileUpload(window.pdfkbApp);
        } else {
            setTimeout(initComponent, 100);
        }
    };
    initComponent();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FileUpload;
}
