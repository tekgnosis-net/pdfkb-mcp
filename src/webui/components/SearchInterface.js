/**
 * SearchInterface Component
 * Handles document search functionality and results display
 */

class SearchInterface {
    constructor(app) {
        this.app = app;
        this.searchInput = document.getElementById('search-input');
        this.searchButton = document.getElementById('search-button');
        this.searchLimit = document.getElementById('search-limit');
        this.minScore = document.getElementById('min-score');
        this.minScoreValue = document.getElementById('min-score-value');
        this.resultsContainer = document.getElementById('search-results');
        this.loadingState = document.getElementById('search-results-loading');
        this.emptyState = document.getElementById('search-empty');

        this.currentQuery = '';
        this.searchOptions = {
            limit: 10,
            min_score: 0.0,
            include_chunks: true
        };

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        this.setupEventListeners();
        this.updateMinScoreDisplay();

        // Register with app
        this.app.components.searchInterface = this;

        console.log('SearchInterface component initialized');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Search input - trigger on Enter key
        if (this.searchInput) {
            this.searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.performSearch();
                }
            });

            // Real-time search suggestions (debounced)
            this.searchInput.addEventListener('input', this.app.debounce((e) => {
                this.handleSearchInput(e.target.value);
            }, 500));
        }

        // Search button
        if (this.searchButton) {
            this.searchButton.addEventListener('click', () => {
                this.performSearch();
            });
        }

        // Search options
        if (this.searchLimit) {
            this.searchLimit.addEventListener('change', (e) => {
                this.searchOptions.limit = parseInt(e.target.value);
                if (this.currentQuery) {
                    this.performSearch();
                }
            });
        }

        if (this.minScore) {
            this.minScore.addEventListener('input', (e) => {
                this.searchOptions.min_score = parseFloat(e.target.value);
                this.updateMinScoreDisplay();
                if (this.currentQuery) {
                    this.performSearch();
                }
            });
        }
    }

    /**
     * Handle search input changes
     */
    handleSearchInput(value) {
        // Could implement search suggestions here
        // For now, just store the value
        this.currentQuery = value.trim();

        // Clear results if input is empty
        if (!this.currentQuery) {
            this.clearResults();
        }
    }

    /**
     * Perform search
     */
    async performSearch() {
        const query = this.searchInput?.value?.trim();

        if (!query) {
            this.showEmptyMessage('Please enter a search query');
            return;
        }

        this.currentQuery = query;
        this.showLoading();

        try {
            const results = await this.app.searchDocuments(query, this.searchOptions);
            // Results are handled by updateResults method called from app

        } catch (error) {
            console.error('Search failed:', error);
            this.showEmptyMessage('Search failed. Please try again.');
            this.app.showToast('error', 'Search Error', 'Failed to search documents');
        }
    }

    /**
     * Update search results display
     */
    updateResults(results) {
        if (!this.resultsContainer) return;

        this.hideLoading();

        if (!results || results.length === 0) {
            this.showEmptyMessage('No results found for your search');
            return;
        }

        this.showResults();
        this.renderResults(results);
    }

    /**
     * Render search results
     */
    renderResults(results) {
        if (!this.resultsContainer) return;

        this.resultsContainer.innerHTML = '';

        results.forEach((result, index) => {
            const resultCard = this.createResultCard(result, index);
            this.resultsContainer.appendChild(resultCard);
        });
    }

    /**
     * Create a search result card
     */
    createResultCard(result, index) {
        const card = document.createElement('div');
        card.className = 'search-result';
        card.setAttribute('data-result-index', index);
        card.setAttribute('data-document-id', result.document_id);

        // Highlight search terms in the content
        const highlightedContent = this.highlightSearchTerms(result.chunk_text, this.currentQuery);

        card.innerHTML = `
            <div class="search-result-header">
                <div>
                    <h3 class="search-result-title">${this.escapeHtml(result.document_title || 'Untitled Document')}</h3>
                    <p class="search-result-path">${this.escapeHtml(result.document_path)}</p>
                </div>
                <div class="search-result-score" title="Relevance Score">
                    ${(result.score * 100).toFixed(1)}%
                </div>
            </div>

            <div class="search-result-content">
                ${highlightedContent}
            </div>

            <div class="search-result-meta">
                ${result.page_number ? `<span>Page ${result.page_number}</span>` : ''}
                <span>Chunk ${result.chunk_index + 1}</span>
                <span>Document ID: ${result.document_id}</span>
            </div>
        `;

        // Add click handler to view document detail
        card.addEventListener('click', () => {
            this.viewDocumentFromResult(result);
        });

        return card;
    }

    /**
     * View document from search result
     */
    async viewDocumentFromResult(result) {
        try {
            // Find document in app's documents array
            let document = this.app.documents.find(d => d.id === result.document_id);

            // If not found, try to load it from API
            if (!document) {
                const response = await this.app.getDocumentDetail(result.document_id, false);
                document = response.document;
            }

            if (document && this.app.components.documentDetail) {
                this.app.currentDocument = document;

                // Load full details with chunks
                const detailResponse = await this.app.getDocumentDetail(result.document_id, true);
                this.app.components.documentDetail.render(detailResponse, {
                    highlightChunk: result.chunk_id,
                    searchQuery: this.currentQuery
                });

                this.app.navigateTo('document-detail');
            }

        } catch (error) {
            console.error('Failed to view document from search result:', error);
            this.app.showToast('error', 'Error', 'Failed to load document details');
        }
    }

    /**
     * Highlight search terms in text
     */
    highlightSearchTerms(text, query) {
        if (!query || !text) return this.escapeHtml(text);

        // Split query into individual terms
        const terms = query.toLowerCase().split(/\s+/).filter(term => term.length > 0);

        let highlightedText = this.escapeHtml(text);

        // Highlight each term
        terms.forEach(term => {
            const regex = new RegExp(`(${this.escapeRegex(term)})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        });

        return highlightedText;
    }

    /**
     * Escape regex special characters
     */
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    /**
     * Get search suggestions (placeholder for future implementation)
     */
    async getSearchSuggestions(query) {
        try {
            const response = await this.app.apiRequest(`/search/suggestions?query=${encodeURIComponent(query)}`);
            return response.suggestions || [];
        } catch (error) {
            console.error('Failed to get search suggestions:', error);
            return [];
        }
    }

    /**
     * Show search suggestions (placeholder for future implementation)
     */
    showSearchSuggestions(suggestions) {
        // Could implement a dropdown with suggestions
        console.log('Search suggestions:', suggestions);
    }

    /**
     * Update min score display
     */
    updateMinScoreDisplay() {
        if (this.minScoreValue && this.minScore) {
            this.minScoreValue.textContent = this.minScore.value;
        }
    }

    /**
     * Show loading state
     */
    showLoading() {
        if (this.loadingState) this.loadingState.classList.remove('hidden');
        if (this.resultsContainer) this.resultsContainer.classList.add('hidden');
        if (this.emptyState) this.emptyState.classList.add('hidden');
    }

    /**
     * Hide loading state
     */
    hideLoading() {
        if (this.loadingState) this.loadingState.classList.add('hidden');
    }

    /**
     * Show results container
     */
    showResults() {
        if (this.resultsContainer) this.resultsContainer.classList.remove('hidden');
        if (this.emptyState) this.emptyState.classList.add('hidden');
    }

    /**
     * Show empty state with message
     */
    showEmptyMessage(message) {
        this.hideLoading();

        if (this.resultsContainer) this.resultsContainer.classList.add('hidden');
        if (this.emptyState) {
            this.emptyState.classList.remove('hidden');

            // Update empty state message
            const emptyText = this.emptyState.querySelector('p');
            if (emptyText) {
                emptyText.textContent = message;
            }
        }
    }

    /**
     * Clear search results
     */
    clearResults() {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = '';
            this.resultsContainer.classList.add('hidden');
        }
        if (this.emptyState) this.emptyState.classList.add('hidden');
        if (this.loadingState) this.loadingState.classList.add('hidden');

        this.currentQuery = '';
    }

    /**
     * Export search results (placeholder for future implementation)
     */
    async exportResults() {
        if (!this.app.searchResults || this.app.searchResults.length === 0) {
            this.app.showToast('warning', 'No Results', 'No search results to export');
            return;
        }

        try {
            // Create CSV content
            const csvContent = this.createCSVFromResults(this.app.searchResults);

            // Download CSV file
            this.downloadCSV(csvContent, `search_results_${this.currentQuery}.csv`);

            this.app.showToast('success', 'Export Complete', 'Search results exported successfully');

        } catch (error) {
            console.error('Failed to export search results:', error);
            this.app.showToast('error', 'Export Failed', 'Failed to export search results');
        }
    }

    /**
     * Create CSV content from search results
     */
    createCSVFromResults(results) {
        const headers = ['Document Title', 'Document Path', 'Score', 'Page', 'Chunk Index', 'Content'];
        const csvRows = [headers.join(',')];

        results.forEach(result => {
            const row = [
                this.escapeCsvField(result.document_title || 'Untitled'),
                this.escapeCsvField(result.document_path),
                (result.score * 100).toFixed(2) + '%',
                result.page_number || '',
                result.chunk_index + 1,
                this.escapeCsvField(result.chunk_text)
            ];
            csvRows.push(row.join(','));
        });

        return csvRows.join('\n');
    }

    /**
     * Escape CSV field
     */
    escapeCsvField(field) {
        if (field === null || field === undefined) return '';

        const stringField = String(field);

        // If field contains comma, quote, or newline, wrap in quotes and escape quotes
        if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
            return '"' + stringField.replace(/"/g, '""') + '"';
        }

        return stringField;
    }

    /**
     * Download CSV file
     */
    downloadCSV(csvContent, filename) {
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
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
     * Reset search form
     */
    reset() {
        if (this.searchInput) this.searchInput.value = '';
        if (this.searchLimit) this.searchLimit.value = '10';
        if (this.minScore) {
            this.minScore.value = '0';
            this.updateMinScoreDisplay();
        }

        this.searchOptions = {
            limit: 10,
            min_score: 0.0,
            include_chunks: true
        };

        this.clearResults();
    }

    /**
     * Set search query and perform search
     */
    async searchFor(query) {
        if (this.searchInput) {
            this.searchInput.value = query;
        }
        this.currentQuery = query;
        await this.performSearch();
    }

    /**
     * Handle real-time search events from WebSocket
     */
    onSearchPerformed(data) {
        // Could update search analytics or recent searches
        console.log('Search performed:', data);
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
            new SearchInterface(window.pdfkbApp);
        } else {
            setTimeout(initComponent, 100);
        }
    };
    initComponent();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchInterface;
}
