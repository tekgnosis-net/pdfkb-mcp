/**
 * Performance Optimization Utilities
 * Additional performance enhancements for the PDF KB frontend
 */

// Performance monitoring and optimization utilities
class PerformanceOptimizer {
    constructor() {
        this.observers = new Map();
        this.loadTimes = new Map();
        this.init();
    }

    init() {
        // Initialize performance monitoring
        if ('performance' in window) {
            this.startLoadTimeTracking();
            this.initIntersectionObserver();
            this.initIdleCallback();
        }
    }

    /**
     * Track page load performance
     */
    startLoadTimeTracking() {
        const startTime = performance.now();

        window.addEventListener('load', () => {
            const loadTime = performance.now() - startTime;
            this.loadTimes.set('page', loadTime);
            console.log(`Page load time: ${loadTime.toFixed(2)}ms`);
        });

        // Track navigation timing
        window.addEventListener('load', () => {
            if (performance.getEntriesByType) {
                const [navigation] = performance.getEntriesByType('navigation');
                if (navigation) {
                    console.log('Navigation timing:', {
                        dns: navigation.domainLookupEnd - navigation.domainLookupStart,
                        connection: navigation.connectEnd - navigation.connectStart,
                        ttfb: navigation.responseStart - navigation.requestStart,
                        download: navigation.responseEnd - navigation.responseStart,
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
                        complete: navigation.loadEventEnd - navigation.navigationStart
                    });
                }
            }
        });
    }

    /**
     * Initialize intersection observer for lazy loading
     */
    initIntersectionObserver() {
        if ('IntersectionObserver' in window) {
            const lazyObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const element = entry.target;

                        // Lazy load images
                        if (element.dataset.src) {
                            element.src = element.dataset.src;
                            element.removeAttribute('data-src');
                        }

                        // Lazy load components
                        if (element.dataset.lazyLoad) {
                            this.loadComponent(element);
                        }

                        lazyObserver.unobserve(element);
                    }
                });
            }, {
                rootMargin: '50px',
                threshold: 0.1
            });

            this.observers.set('lazy', lazyObserver);
        }
    }

    /**
     * Initialize idle callback for non-critical tasks
     */
    initIdleCallback() {
        if ('requestIdleCallback' in window) {
            // Schedule non-critical tasks during idle time
            requestIdleCallback(() => {
                this.preloadCriticalResources();
                this.cleanupOldCache();
                this.optimizeMemoryUsage();
            }, { timeout: 5000 });
        }
    }

    /**
     * Preload critical resources
     */
    preloadCriticalResources() {
        // Preload component files
        const componentFiles = [
            '/components/DocumentList.js',
            '/components/SearchInterface.js',
            '/components/DocumentDetail.js',
            '/components/FileUpload.js',
            '/components/StatusBar.js'
        ];

        componentFiles.forEach(url => {
            const link = document.createElement('link');
            link.rel = 'prefetch';
            link.href = url;
            document.head.appendChild(link);
        });
    }

    /**
     * Clean up old cache entries
     */
    cleanupOldCache() {
        try {
            // Clean up old localStorage entries
            const keysToCheck = ['pdfkb_cache_', 'pdfkb_temp_'];
            const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);

            keysToCheck.forEach(prefix => {
                Object.keys(localStorage).forEach(key => {
                    if (key.startsWith(prefix)) {
                        try {
                            const data = JSON.parse(localStorage.getItem(key));
                            if (data.timestamp && data.timestamp < oneWeekAgo) {
                                localStorage.removeItem(key);
                            }
                        } catch (e) {
                            // Remove invalid entries
                            localStorage.removeItem(key);
                        }
                    }
                });
            });
        } catch (error) {
            console.warn('Cache cleanup failed:', error);
        }
    }

    /**
     * Optimize memory usage
     */
    optimizeMemoryUsage() {
        // Remove unused event listeners
        this.cleanupEventListeners();

        // Clear unused references
        if (window.pdfkbApp) {
            // Clean up old search results if too many
            if (window.pdfkbApp.searchResults && window.pdfkbApp.searchResults.length > 100) {
                window.pdfkbApp.searchResults = window.pdfkbApp.searchResults.slice(0, 50);
            }
        }
    }

    /**
     * Clean up unused event listeners
     */
    cleanupEventListeners() {
        // Remove observers for elements that no longer exist
        this.observers.forEach((observer, key) => {
            // Implementation would depend on specific observer tracking
        });
    }

    /**
     * Load component lazily
     */
    async loadComponent(element) {
        const componentName = element.dataset.lazyLoad;

        try {
            // Dynamic import for modern browsers
            if ('import' in window) {
                await import(`/components/${componentName}.js`);
            }

            element.classList.add('loaded');
        } catch (error) {
            console.error(`Failed to load component ${componentName}:`, error);
        }
    }

    /**
     * Measure and report performance metrics
     */
    getPerformanceMetrics() {
        const metrics = {};

        // Memory usage (if available)
        if ('memory' in performance) {
            metrics.memory = {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576), // MB
                total: Math.round(performance.memory.totalJSHeapSize / 1048576), // MB
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) // MB
            };
        }

        // Connection info
        if ('connection' in navigator) {
            metrics.connection = {
                type: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt,
                saveData: navigator.connection.saveData
            };
        }

        // Load times
        metrics.loadTimes = Object.fromEntries(this.loadTimes);

        return metrics;
    }

    /**
     * Optimize images for performance
     */
    optimizeImages() {
        const images = document.querySelectorAll('img[data-src]');

        images.forEach(img => {
            // Add to lazy loading observer
            if (this.observers.has('lazy')) {
                this.observers.get('lazy').observe(img);
            }
        });
    }

    /**
     * Debounce function for performance
     */
    static debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    }

    /**
     * Throttle function for performance
     */
    static throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Virtual scrolling implementation for large lists
     */
    static createVirtualScroller(container, items, renderItem, itemHeight = 50) {
        let scrollTop = 0;
        let startIndex = 0;
        let endIndex = 0;

        const containerHeight = container.clientHeight;
        const visibleCount = Math.ceil(containerHeight / itemHeight) + 2; // Buffer

        const updateVisibleItems = PerformanceOptimizer.throttle(() => {
            startIndex = Math.floor(scrollTop / itemHeight);
            endIndex = Math.min(startIndex + visibleCount, items.length);

            // Clear container
            container.innerHTML = '';

            // Create spacer for items above
            if (startIndex > 0) {
                const spacer = document.createElement('div');
                spacer.style.height = `${startIndex * itemHeight}px`;
                container.appendChild(spacer);
            }

            // Render visible items
            for (let i = startIndex; i < endIndex; i++) {
                const itemElement = renderItem(items[i], i);
                container.appendChild(itemElement);
            }

            // Create spacer for items below
            if (endIndex < items.length) {
                const spacer = document.createElement('div');
                spacer.style.height = `${(items.length - endIndex) * itemHeight}px`;
                container.appendChild(spacer);
            }
        }, 16); // ~60fps

        container.addEventListener('scroll', (e) => {
            scrollTop = e.target.scrollTop;
            updateVisibleItems();
        });

        // Initial render
        updateVisibleItems();

        return {
            update: (newItems) => {
                items = newItems;
                updateVisibleItems();
            }
        };
    }
}

// Web Workers for heavy computations
class WebWorkerHelper {
    static isSupported() {
        return typeof Worker !== 'undefined';
    }

    static createWorker(workerFunction) {
        if (!this.isSupported()) {
            return null;
        }

        const blob = new Blob([`(${workerFunction.toString()})()`], {
            type: 'application/javascript'
        });

        return new Worker(URL.createObjectURL(blob));
    }

    static processSearchInWorker(documents, query) {
        if (!this.isSupported()) {
            return Promise.resolve([]);
        }

        return new Promise((resolve, reject) => {
            const worker = this.createWorker(() => {
                self.onmessage = function(e) {
                    const { documents, query } = e.data;

                    // Simple text search in worker
                    const results = documents.filter(doc =>
                        doc.title.toLowerCase().includes(query.toLowerCase()) ||
                        doc.content.toLowerCase().includes(query.toLowerCase())
                    );

                    self.postMessage(results);
                };
            });

            worker.onmessage = (e) => {
                resolve(e.data);
                worker.terminate();
            };

            worker.onerror = (error) => {
                reject(error);
                worker.terminate();
            };

            worker.postMessage({ documents, query });
        });
    }
}

// Initialize performance optimization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.performanceOptimizer = new PerformanceOptimizer();

    // Expose utilities globally
    window.PerfUtils = {
        debounce: PerformanceOptimizer.debounce,
        throttle: PerformanceOptimizer.throttle,
        createVirtualScroller: PerformanceOptimizer.createVirtualScroller,
        WebWorkerHelper
    };
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PerformanceOptimizer, WebWorkerHelper };
}
