import React, { useState } from 'react';
import { X, ExternalLink, Download, FileText, ChevronLeft, ChevronRight } from 'lucide-react';

export const PdfViewerModal = ({
    isOpen,
    onClose,
    documentName,
    initialPage = 1,
    apiBaseUrl = 'http://127.0.0.1:8000'
}) => {
    const [currentPage, setCurrentPage] = useState(initialPage || 1);

    if (!isOpen || !documentName) return null;

    const encodedDocName = encodeURIComponent(documentName);
    const pdfViewUrl = `${apiBaseUrl}/files/${encodedDocName}/view#page=${currentPage}`;
    const pdfDownloadUrl = `${apiBaseUrl}/files/${encodedDocName}/download`;

    return (
        <div className="modal-backdrop pdf-viewer-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
            <div className="pdf-viewer-card" onClick={(e) => e.stopPropagation()}>
                <header className="pdf-viewer-header">
                    <div className="pdf-viewer-title-group">
                        <div className="pdf-viewer-icon">
                            <FileText size={18} color="var(--accent-primary)" />
                        </div>
                        <div>
                            <h3 className="pdf-viewer-filename">{documentName}</h3>
                            <span className="pdf-viewer-page-badge">Viewing Page {currentPage}</span>
                        </div>
                    </div>

                    <div className="pdf-viewer-actions">
                        <div className="pdf-viewer-nav-group">
                            <button
                                className="pdf-nav-btn"
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                disabled={currentPage <= 1}
                                title="Previous page"
                            >
                                <ChevronLeft size={16} />
                            </button>
                            <span className="pdf-nav-page-text">p. {currentPage}</span>
                            <button
                                className="pdf-nav-btn"
                                onClick={() => setCurrentPage(p => p + 1)}
                                title="Next page"
                            >
                                <ChevronRight size={16} />
                            </button>
                        </div>

                        <a
                            href={pdfViewUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-secondary btn-sm"
                            title="Open in new browser tab"
                        >
                            <ExternalLink size={14} />
                            <span>Open Tab</span>
                        </a>

                        <a
                            href={pdfDownloadUrl}
                            download={documentName}
                            className="btn-secondary btn-sm"
                            title="Download original file"
                        >
                            <Download size={14} />
                            <span>Download</span>
                        </a>

                        <button className="pdf-viewer-close-btn" onClick={onClose} title="Close PDF viewer" aria-label="Close PDF viewer">
                            <X size={20} />
                        </button>
                    </div>
                </header>

                <div className="pdf-viewer-body">
                    <iframe
                        key={`${documentName}-${currentPage}`}
                        src={pdfViewUrl}
                        title={`PDF Preview for ${documentName}`}
                        className="pdf-viewer-iframe"
                    />
                </div>
            </div>
        </div>
    );
};

export default PdfViewerModal;
