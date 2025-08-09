/**
 * StatusBar Component
 * Handles real-time status updates, system metrics, and connection management
 */

class StatusBar {
    constructor(app) {
        this.app = app;
        this.connectionStatus = document.getElementById('connection-status');
        this.statusIndicator = document.querySelector('.status-indicator');
        this.statusText = document.querySelector('.status-text');
        this.systemStats = document.getElementById('system-stats');
        this.documentsCount = document.getElementById('documents-count');
        this.chunksCount = document.getElementById('chunks-count');

        this.systemMetrics = {
            documents: 0,
            chunks: 0,
            uptime: 0,
            lastUpdated: null
        };

        this.connectionState = 'disconnected';
        this.updateInterval = null;

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        this.setupEventListeners();
        this.startPeriodicUpdates();

        // Register with app
        this.app.components.statusBar = this;

        console.log('StatusBar component initialized');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Connection status click for details
        if (this.connectionStatus) {
            this.connectionStatus.addEventListener('click', () => {
                this.showConnectionDetails();
            });
        }

        // Listen to app events
        if (this.app.websocket) {
            this.updateConnectionStatus('connected');
        }
    }

    /**
     * Start periodic updates
     */
    startPeriodicUpdates() {
        // Update system stats every 30 seconds
        this.updateInterval = setInterval(() => {
            this.updateSystemStats();
        }, 30000);

        // Initial update
        this.updateSystemStats();
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(status, message = '') {
        this.connectionState = status;

        if (this.statusIndicator) {
            this.statusIndicator.setAttribute('data-status', status);
        }

        if (this.statusText) {
            const statusMessages = {
                connecting: 'Connecting...',
                connected: 'Connected',
                disconnected: 'Disconnected',
                error: 'Connection Error'
            };

            this.statusText.textContent = message || statusMessages[status] || status;
        }

        // Add animation for status changes
        if (this.connectionStatus) {
            this.connectionStatus.classList.add('status-updated');
            setTimeout(() => {
                this.connectionStatus.classList.remove('status-updated');
            }, 1000);
        }
    }

    /**
     * Update system statistics
     */
    async updateSystemStats() {
        try {
            const status = await this.app.apiRequest('/status');

            this.systemMetrics = {
                documents: status.documents_count || 0,
                chunks: status.chunks_count || 0,
                uptime: status.uptime || 0,
                lastUpdated: new Date()
            };

            this.renderSystemStats();

        } catch (error) {
            console.error('Failed to update system stats:', error);
            // Don't show error toast for status updates as it would be too noisy
        }
    }

    /**
     * Render system statistics
     */
    renderSystemStats() {
        if (this.documentsCount) {
            this.animateCounterUpdate(this.documentsCount, this.systemMetrics.documents);
        }

        if (this.chunksCount) {
            this.animateCounterUpdate(this.chunksCount, this.systemMetrics.chunks);
        }

        // Show system stats if there's data
        if (this.systemStats && (this.systemMetrics.documents > 0 || this.systemMetrics.chunks > 0)) {
            this.systemStats.classList.remove('hidden');
        }
    }

    /**
     * Animate counter update
     */
    animateCounterUpdate(element, newValue) {
        const currentValue = parseInt(element.textContent) || 0;

        if (currentValue === newValue) return;

        // Simple animation - could be enhanced
        element.classList.add('counter-updating');

        setTimeout(() => {
            element.textContent = newValue.toLocaleString();
            element.classList.remove('counter-updating');
        }, 150);
    }

    /**
     * Show connection details modal
     */
    showConnectionDetails() {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content" style="max-width: 600px;">
                <div class="modal-header">
                    <h3>Connection & System Status</h3>
                    <button class="modal-close status-modal-close">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="modal-body">
                    ${this.renderConnectionDetails()}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary status-modal-close">Close</button>
                    <button class="btn btn-primary" onclick="window.pdfkbApp.components.statusBar.refreshStatus()">
                        Refresh Status
                    </button>
                </div>
            </div>
        `;

        // Add close handlers
        const closeButtons = modal.querySelectorAll('.status-modal-close');
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
     * Render connection details
     */
    renderConnectionDetails() {
        const uptime = this.formatUptime(this.systemMetrics.uptime);
        const lastUpdated = this.systemMetrics.lastUpdated ?
            this.systemMetrics.lastUpdated.toLocaleTimeString() : 'Never';

        const wsState = this.app.websocket ? this.app.websocket.readyState : -1;
        const wsStatusText = this.getWebSocketStatusText(wsState);

        return `
            <div class="status-details">
                <div class="status-section">
                    <h4>Connection Status</h4>
                    <div class="status-grid">
                        <div class="status-item">
                            <span class="status-label">WebSocket:</span>
                            <span class="status-value ${this.connectionState}">
                                <div class="status-indicator" data-status="${this.connectionState}"></div>
                                ${wsStatusText}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">API Endpoint:</span>
                            <span class="status-value">${this.app.apiBaseUrl}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">WebSocket URL:</span>
                            <span class="status-value">${this.app.wsUrl}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Reconnect Attempts:</span>
                            <span class="status-value">${this.app.wsReconnectAttempts}</span>
                        </div>
                    </div>
                </div>

                <div class="status-section">
                    <h4>System Metrics</h4>
                    <div class="status-grid">
                        <div class="status-item">
                            <span class="status-label">Documents:</span>
                            <span class="status-value">${this.systemMetrics.documents.toLocaleString()}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Chunks:</span>
                            <span class="status-value">${this.systemMetrics.chunks.toLocaleString()}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Server Uptime:</span>
                            <span class="status-value">${uptime}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Last Updated:</span>
                            <span class="status-value">${lastUpdated}</span>
                        </div>
                    </div>
                </div>

                <div class="status-section">
                    <h4>Browser Info</h4>
                    <div class="status-grid">
                        <div class="status-item">
                            <span class="status-label">Online:</span>
                            <span class="status-value ${navigator.onLine ? 'connected' : 'disconnected'}">
                                <div class="status-indicator" data-status="${navigator.onLine ? 'connected' : 'disconnected'}"></div>
                                ${navigator.onLine ? 'Yes' : 'No'}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">User Agent:</span>
                            <span class="status-value" title="${navigator.userAgent}">
                                ${this.getBrowserName()}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Theme:</span>
                            <span class="status-value">${this.app.theme}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Local Storage:</span>
                            <span class="status-value">${this.isLocalStorageAvailable() ? 'Available' : 'Not Available'}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Get WebSocket status text
     */
    getWebSocketStatusText(readyState) {
        switch (readyState) {
            case WebSocket.CONNECTING:
                return 'Connecting';
            case WebSocket.OPEN:
                return 'Connected';
            case WebSocket.CLOSING:
                return 'Closing';
            case WebSocket.CLOSED:
                return 'Disconnected';
            default:
                return 'Unknown';
        }
    }

    /**
     * Format uptime duration
     */
    formatUptime(seconds) {
        if (!seconds) return 'Unknown';

        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        const parts = [];
        if (days > 0) parts.push(`${days}d`);
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0) parts.push(`${minutes}m`);
        if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

        return parts.join(' ');
    }

    /**
     * Get browser name
     */
    getBrowserName() {
        const ua = navigator.userAgent;

        if (ua.includes('Firefox')) return 'Firefox';
        if (ua.includes('Chrome') && !ua.includes('Edg')) return 'Chrome';
        if (ua.includes('Safari') && !ua.includes('Chrome')) return 'Safari';
        if (ua.includes('Edg')) return 'Edge';
        if (ua.includes('Opera')) return 'Opera';

        return 'Unknown';
    }

    /**
     * Check if localStorage is available
     */
    isLocalStorageAvailable() {
        try {
            localStorage.setItem('test', 'test');
            localStorage.removeItem('test');
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Refresh status manually
     */
    async refreshStatus() {
        await this.updateSystemStats();

        // Test WebSocket connection
        if (this.app.websocket && this.app.websocket.readyState === WebSocket.OPEN) {
            try {
                this.app.websocket.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            } catch (error) {
                console.error('Failed to ping WebSocket:', error);
            }
        }

        this.app.showToast('success', 'Status Updated', 'System status has been refreshed');
    }

    /**
     * Handle real-time events
     */
    onDocumentAdded(data) {
        this.systemMetrics.documents++;
        this.renderSystemStats();
        this.showActivityIndicator('document-added');
    }

    onDocumentRemoved(data) {
        this.systemMetrics.documents--;
        this.renderSystemStats();
        this.showActivityIndicator('document-removed');
    }

    onProcessingStarted(data) {
        this.showActivityIndicator('processing');
    }

    onProcessingCompleted(data) {
        if (data.chunks_created) {
            this.systemMetrics.chunks += data.chunks_created;
            this.renderSystemStats();
        }
        this.showActivityIndicator('processing-complete');
    }

    onSearchPerformed(data) {
        this.showActivityIndicator('search');
    }

    onWebSocketConnected() {
        this.updateConnectionStatus('connected');
    }

    onWebSocketDisconnected() {
        this.updateConnectionStatus('disconnected');
    }

    onWebSocketConnecting() {
        this.updateConnectionStatus('connecting');
    }

    onWebSocketError(error) {
        this.updateConnectionStatus('error', 'Connection Error');
    }

    /**
     * Show activity indicator
     */
    showActivityIndicator(activity) {
        if (!this.connectionStatus) return;

        // Add a small pulse animation to show activity
        this.connectionStatus.classList.add('activity-pulse');

        setTimeout(() => {
            this.connectionStatus.classList.remove('activity-pulse');
        }, 1000);
    }

    /**
     * Handle system status changes
     */
    onSystemStatusChanged(data) {
        if (data.status === 'shutting_down') {
            this.updateConnectionStatus('disconnected', 'Server Shutting Down');
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Get current status summary
     */
    getStatusSummary() {
        return {
            connection: this.connectionState,
            documents: this.systemMetrics.documents,
            chunks: this.systemMetrics.chunks,
            uptime: this.systemMetrics.uptime,
            lastUpdated: this.systemMetrics.lastUpdated,
            websocketState: this.app.websocket ? this.app.websocket.readyState : -1,
            browserOnline: navigator.onLine
        };
    }

    /**
     * Export status report
     */
    async exportStatusReport() {
        try {
            const status = this.getStatusSummary();
            const timestamp = new Date().toISOString();

            const report = {
                timestamp,
                connection: status.connection,
                metrics: {
                    documents: status.documents,
                    chunks: status.chunks,
                    uptime: status.uptime
                },
                technical: {
                    websocket_state: status.websocketState,
                    browser_online: status.browserOnline,
                    user_agent: navigator.userAgent,
                    api_endpoint: this.app.apiBaseUrl,
                    websocket_url: this.app.wsUrl
                }
            };

            const reportText = JSON.stringify(report, null, 2);
            const filename = `pdfkb_status_report_${timestamp.replace(/[:.]/g, '-')}.json`;

            this.downloadTextFile(reportText, filename);
            this.app.showToast('success', 'Report Exported', 'Status report exported successfully');

        } catch (error) {
            console.error('Failed to export status report:', error);
            this.app.showToast('error', 'Export Failed', 'Failed to export status report');
        }
    }

    /**
     * Download text file
     */
    downloadTextFile(content, filename) {
        const blob = new Blob([content], { type: 'application/json' });
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
}

// Initialize when app is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for app to be initialized
    const initComponent = () => {
        if (window.pdfkbApp) {
            new StatusBar(window.pdfkbApp);
        } else {
            setTimeout(initComponent, 100);
        }
    };
    initComponent();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StatusBar;
}
