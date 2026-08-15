import { useCallback, useEffect, useRef, useState } from 'react';
import {
    Activity, Brain, Check, Database, ExternalLink, FileSpreadsheet, FileText,
    HardDrive, Laptop, Loader2, Moon, Paperclip, RefreshCw, Sparkles,
    Sun, Trash2, UploadCloud, X, Zap
} from 'lucide-react';

import {
    cleanupOrphanedDocuments, deleteFile, fetchFiles,
    fetchSystemDiagnostics, fetchSystemStatus, getFileUrl, uploadFile
} from '../api';

const ACCEPTED_DOCUMENTS = '.pdf,.txt,.docx,.xlsx,.xls,.md,.csv';

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function getFileIcon(type, filename = '') {
    const ext = (type || (filename ? filename.split('.').pop() : '') || '').toUpperCase();
    if (['XLSX', 'XLS', 'CSV'].includes(ext)) return <FileSpreadsheet size={18} color="var(--success-primary, #10b981)" />;
    if (['PDF'].includes(ext)) return <FileText size={18} color="var(--danger-primary, #ef4444)" />;
    if (['DOCX', 'DOC', 'TXT', 'MD'].includes(ext)) return <FileText size={18} color="var(--accent-primary, #3b82f6)" />;
    return <FileText size={18} color="var(--text-secondary, #94a3b8)" />;
}

const SettingsView = ({
    themeMode = 'system',
    setThemeMode,
    responseMode = 'balanced',
    setResponseMode,
    files = [],
    onFilesChange,
    onDeleteDocument,
    showToast,
}) => {
    const [systemStatus, setSystemStatus] = useState(null);
    const [diagnostics, setDiagnostics] = useState(null);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [isCleaning, setIsCleaning] = useState(false);
    const [showCleanupModal, setShowCleanupModal] = useState(false);
    const [deleteTarget, setDeleteTarget] = useState(null); // { filename, document_id }
    const [isDeleting, setIsDeleting] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(null);
    const [localFiles, setLocalFiles] = useState(files);
    const fileInputRef = useRef(null);

    const loadAllStatus = useCallback(async (isManual = false) => {
        setLoadingStatus(true);
        try {
            const [statusData, diagData, filesData] = await Promise.all([
                fetchSystemStatus(),
                fetchSystemDiagnostics().catch(() => null),
                fetchFiles().catch(() => ({ files: [] })),
            ]);
            setSystemStatus(statusData);
            setDiagnostics(diagData);
            setLocalFiles(filesData.files || []);
            onFilesChange?.();

            if (isManual) {
                const docs = statusData?.rag_engine?.total_documents_indexed ?? 0;
                const chunks = statusData?.rag_engine?.total_chunks_indexed ?? 0;
                showToast?.(`System status updated: ${docs} document(s) & ${chunks} chunk(s) indexed.`, 'success');
            }
        } catch (e) {
            console.error('Status check error:', e);
            setSystemStatus({ status: 'error', database: { connected: false } });
            if (isManual) {
                showToast?.(`Unable to check system status: ${e.message}`, 'error');
            }
        } finally {
            setLoadingStatus(false);
        }
    }, [onFilesChange, showToast]);

    useEffect(() => {
        setLocalFiles(files);
    }, [files]);

    useEffect(() => {
        void loadAllStatus();
    }, [loadAllStatus]);

    const handleFileUpload = async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile || isUploading) return;

        setIsUploading(true);
        setUploadProgress({ name: selectedFile.name, size: selectedFile.size, stage: 'Uploading...' });

        try {
            setUploadProgress((prev) => ({ ...prev, stage: 'Processing & Indexing...' }));
            await uploadFile(selectedFile);
            showToast?.(`Document "${selectedFile.name}" uploaded and indexed successfully.`, 'success');
            await loadAllStatus(false);
        } catch (error) {
            console.error('Upload failed:', error);
            showToast?.(`Upload failed: ${error.message}`, 'error');
        } finally {
            setIsUploading(false);
            setUploadProgress(null);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const handleConfirmDelete = async () => {
        if (!deleteTarget || isDeleting) return;
        setIsDeleting(true);
        const filename = deleteTarget.filename;
        const target = deleteTarget.document_id || deleteTarget.filename;

        try {
            if (onDeleteDocument) {
                await onDeleteDocument(target);
            } else {
                await deleteFile(target);
                showToast?.(`Document "${filename}" permanently deleted.`, 'success');
            }
            setDeleteTarget(null);
            await loadAllStatus(false);
        } catch (error) {
            console.error('Delete error:', error);
            showToast?.(`Delete failed: ${error.message}`, 'error');
        } finally {
            setIsDeleting(false);
        }
    };

    const handleExecuteOrphanCleanup = async () => {
        if (isCleaning) return;
        setIsCleaning(true);
        try {
            const report = await cleanupOrphanedDocuments();
            const docsRemoved = report.documents_removed || 0;
            const chunksRemoved = report.chunks_removed || 0;
            const vectorsRemoved = report.vectors_removed || 0;
            const filesRemoved = report.files_removed || 0;
            const totalPurged = report.records_removed || (docsRemoved + chunksRemoved + vectorsRemoved + filesRemoved);

            if (totalPurged > 0) {
                showToast?.(`Cleanup complete: Removed ${docsRemoved} document(s) and ${chunksRemoved} chunk(s).`, 'success');
            } else {
                showToast?.('Cleanup complete: No orphaned data found.', 'info');
            }
            setShowCleanupModal(false);
            await loadAllStatus(false);
        } catch (error) {
            console.error('Orphan cleanup error:', error);
            showToast?.(`Cleanup failed: ${error.message}`, 'error');
        } finally {
            setIsCleaning(false);
        }
    };

    const readyFiles = localFiles.filter((f) => f.status === 'READY');
    const indexedDocs = readyFiles.length;
    const indexedPages = readyFiles.reduce((acc, f) => acc + (f.total_pages || f.page_count || 0), 0);
    const indexedChunks = readyFiles.reduce((acc, f) => acc + (f.chunk_count || 0), 0);
    const isDbConnected = systemStatus?.database?.connected ?? diagnostics?.database === 'Connected';
    const isStorageAvailable = systemStatus?.storage?.available ?? diagnostics?.storage === 'Available';
    const isRagReady = (systemStatus?.rag_engine?.status === 'Ready') || (diagnostics?.rag_status === 'Ready') || (diagnostics?.rag_status === 'Available');

    return (
        <div className="view-container settings-view-wrapper">
            <header className="view-header">
                <div>
                    <h2>Settings</h2>
                    <p>Customize your AI assistant, manage knowledge documents, and inspect system health.</p>
                </div>
            </header>

            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_DOCUMENTS}
                style={{ display: 'none' }}
                onChange={handleFileUpload}
            />

            {/* ─── SECTION A: APPEARANCE ──────────────────────────────────────── */}
            <section className="settings-section">
                <div className="section-title-wrap">
                    <h3>Appearance</h3>
                    <span className="section-subtitle">Select your preferred color theme across the entire application.</span>
                </div>

                <div className="settings-options-grid three-columns">
                    <button
                        type="button"
                        className={`setting-option-card ${themeMode === 'light' ? 'selected' : ''}`}
                        onClick={() => setThemeMode?.('light')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Sun size={20} /></div>
                            {themeMode === 'light' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">Light</strong>
                            <span className="option-desc">Clean, high-contrast light theme for bright spaces.</span>
                        </div>
                    </button>

                    <button
                        type="button"
                        className={`setting-option-card ${themeMode === 'dark' ? 'selected' : ''}`}
                        onClick={() => setThemeMode?.('dark')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Moon size={20} /></div>
                            {themeMode === 'dark' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">Dark</strong>
                            <span className="option-desc">Sleek dark theme optimized for low-light focus.</span>
                        </div>
                    </button>

                    <button
                        type="button"
                        className={`setting-option-card ${themeMode === 'system' ? 'selected' : ''}`}
                        onClick={() => setThemeMode?.('system')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Laptop size={20} /></div>
                            {themeMode === 'system' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">System</strong>
                            <span className="option-desc">Automatically synchronize with your operating system theme.</span>
                        </div>
                    </button>
                </div>
            </section>

            {/* ─── SECTION B: AI RESPONSE MODE ───────────────────────────────── */}
            <section className="settings-section">
                <div className="section-title-wrap">
                    <h3>AI Response Mode</h3>
                    <span className="section-subtitle">Configure model inference latency, retrieval depth, and reasoning tokens for new chats.</span>
                </div>

                <div className="settings-options-grid three-columns">
                    <button
                        type="button"
                        className={`setting-option-card ${responseMode === 'fast' ? 'selected' : ''}`}
                        onClick={() => setResponseMode?.('fast')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Zap size={20} /></div>
                            {responseMode === 'fast' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">Fast</strong>
                            <span className="option-desc">Prioritizes immediate response speed with concise context.</span>
                        </div>
                    </button>

                    <button
                        type="button"
                        className={`setting-option-card ${responseMode === 'balanced' ? 'selected' : ''}`}
                        onClick={() => setResponseMode?.('balanced')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Sparkles size={20} /></div>
                            {responseMode === 'balanced' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">Balanced</strong>
                            <span className="option-desc">Standard configuration with balanced depth and speed.</span>
                        </div>
                    </button>

                    <button
                        type="button"
                        className={`setting-option-card ${responseMode === 'deep' ? 'selected' : ''}`}
                        onClick={() => setResponseMode?.('deep')}
                    >
                        <div className="option-card-header">
                            <div className="option-icon-wrap"><Brain size={20} /></div>
                            {responseMode === 'deep' && <span className="option-check-badge"><Check size={14} /></span>}
                        </div>
                        <div className="option-card-content">
                            <strong className="option-title">Deep Reasoning</strong>
                            <span className="option-desc">Expanded retrieval context and analytical reasoning instructions.</span>
                        </div>
                    </button>
                </div>
            </section>

            {/* ─── SECTION C: DOCUMENTS & KNOWLEDGE ───────────────────────────── */}
            <section className="settings-section">
                <div className="section-header-flex">
                    <div className="section-title-wrap">
                        <h3>Documents & Knowledge</h3>
                        <span className="section-subtitle">Manage knowledge base files indexed for retrieval-augmented generation.</span>
                    </div>

                    <button
                        type="button"
                        className="btn-primary btn-upload-doc"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                    >
                        {isUploading ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />}
                        <span>{isUploading ? 'Uploading...' : 'Upload Document'}</span>
                    </button>
                </div>

                {/* Knowledge Overview Metrics */}
                <div className="knowledge-metrics-bar">
                    <div className="metric-pill">
                        <span className="metric-label">Documents</span>
                        <strong className="metric-value">{indexedDocs}</strong>
                    </div>
                    <div className="metric-pill">
                        <span className="metric-label">Pages</span>
                        <strong className="metric-value">{diagnostics?.indexed_pages ?? localFiles.reduce((acc, f) => acc + (f.total_pages || f.page_count || 0), 0)}</strong>
                    </div>
                    <div className="metric-pill">
                        <span className="metric-label">Chunks</span>
                        <strong className="metric-value">{indexedChunks}</strong>
                    </div>
                </div>

                {/* Live Upload Progress Feedback */}
                {uploadProgress && (
                    <div className="upload-progress-card">
                        <Loader2 size={20} className="animate-spin" color="var(--accent-primary)" />
                        <div className="progress-details">
                            <strong>{uploadProgress.name}</strong>
                            <span>{formatBytes(uploadProgress.size)} · {uploadProgress.stage}</span>
                        </div>
                    </div>
                )}

                {/* Document List */}
                <div className="documents-container-card">
                    <div className="documents-card-header">
                        <h4>View Documents ({localFiles.length})</h4>
                    </div>

                    {localFiles.length === 0 ? (
                        <div className="documents-empty-state">
                            <FileText size={38} className="empty-state-icon" />
                            <h4>No documents indexed</h4>
                            <p>Upload PDF, TXT, DOCX, or spreadsheet files to enable RAG question-answering across your data.</p>
                            <button
                                type="button"
                                className="btn-secondary"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isUploading}
                            >
                                <Paperclip size={15} /> Upload your first document
                            </button>
                        </div>
                    ) : (
                        <div className="documents-table-wrapper">
                            <table className="documents-table">
                                <thead>
                                    <tr>
                                        <th>Document</th>
                                        <th>Size</th>
                                        <th>Pages</th>
                                        <th>Status</th>
                                        <th>Chunks</th>
                                        <th>Uploaded</th>
                                        <th style={{ textAlign: 'right' }}>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {localFiles.map((doc) => {
                                        const filename = doc.filename || doc.name;
                                        const ext = (doc.file_type || (filename ? filename.split('.').pop() : '') || 'FILE').toUpperCase();
                                        const sizeStr = doc.size || formatBytes(doc.size_bytes || doc.file_size);
                                        const statusStr = doc.status === 'READY' ? 'Ready' : doc.status === 'FAILED' ? 'Failed' : 'Processing';
                                        const pageCount = doc.total_pages || doc.page_count || 0;

                                        return (
                                            <tr key={filename}>
                                                <td className="doc-col-title">
                                                    <div className="doc-icon-col">{getFileIcon(ext, filename)}</div>
                                                    <div className="doc-text-col">
                                                        <span className="doc-filename-text" title={filename}>{filename}</span>
                                                        <span className="doc-ext-badge">{ext}</span>
                                                    </div>
                                                </td>
                                                <td>{sizeStr}</td>
                                                <td>{pageCount > 0 ? `${pageCount} pages` : '—'}</td>
                                                <td>
                                                    <span className={`status-pill ${statusStr === 'Ready' ? 'status-ready' : statusStr === 'Failed' ? 'status-failed' : 'status-processing'}`}>
                                                        {statusStr === 'Ready' ? 'Processed' : statusStr}
                                                    </span>
                                                </td>
                                                <td>{doc.chunk_count ?? 0} chunks</td>
                                                <td>{doc.uploaded_on || 'Recently'}</td>
                                                <td style={{ textAlign: 'right' }}>
                                                    <div className="doc-action-group">
                                                        <button
                                                            type="button"
                                                            className="btn-table-action"
                                                            onClick={() => window.open(getFileUrl(filename), '_blank')}
                                                            title={`View ${filename}`}
                                                        >
                                                            <ExternalLink size={14} />
                                                            <span>View</span>
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="btn-table-action danger"
                                                            onClick={() => setDeleteTarget({ filename, document_id: doc.document_id })}
                                                            title={`Delete ${filename}`}
                                                        >
                                                            <Trash2 size={14} />
                                                            <span>Delete</span>
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </section>

            {/* ─── SECTION D: SYSTEM STATUS & DATA MAINTENANCE ───────────────── */}
            <section className="settings-section">
                <div className="section-header-flex">
                    <div className="section-title-wrap">
                        <h3>System Status</h3>
                        <span className="section-subtitle">Live health status across database, local file storage, and RAG vector indices.</span>
                    </div>

                    <button
                        type="button"
                        className="btn-secondary btn-check-status"
                        onClick={() => loadAllStatus(true)}
                        disabled={loadingStatus}
                    >
                        <RefreshCw size={15} className={loadingStatus ? 'animate-spin' : ''} />
                        <span>{loadingStatus ? 'Checking...' : 'Check Status'}</span>
                    </button>
                </div>

                {/* Status Indicators Grid */}
                <div className="system-health-grid">
                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">Database</span>
                            <Database size={16} className="health-icon" />
                        </div>
                        <strong className={`health-value ${isDbConnected ? 'connected' : 'disconnected'}`}>
                            {isDbConnected ? '● Connected' : '● Disconnected'}
                        </strong>
                        <span className="health-subtext">{systemStatus?.database?.is_local ? 'Local MongoDB' : 'MongoDB'}</span>
                    </div>

                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">Storage</span>
                            <HardDrive size={16} className="health-icon" />
                        </div>
                        <strong className={`health-value ${isStorageAvailable ? 'connected' : 'disconnected'}`}>
                            {isStorageAvailable ? '● Available' : '● Error'}
                        </strong>
                        <span className="health-subtext">Uploads Directory</span>
                    </div>

                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">RAG Engine</span>
                            <Activity size={16} className="health-icon" />
                        </div>
                        <strong className={`health-value ${isRagReady ? 'connected' : 'disconnected'}`}>
                            {isRagReady ? '● Ready' : '● Error'}
                        </strong>
                        <span className="health-subtext">{systemStatus?.rag_engine?.groq_model || 'LLM Ready'}</span>
                    </div>

                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">Documents</span>
                            <FileText size={16} className="health-icon" />
                        </div>
                        <strong className="health-value-num">{indexedDocs}</strong>
                        <span className="health-subtext">Active knowledge docs</span>
                    </div>

                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">Pages</span>
                            <FileText size={16} className="health-icon" />
                        </div>
                        <strong className="health-value-num">{diagnostics?.indexed_pages ?? localFiles.reduce((acc, f) => acc + (f.total_pages || f.page_count || 0), 0)}</strong>
                        <span className="health-subtext">Structured page records</span>
                    </div>

                    <div className="health-card">
                        <div className="health-card-header">
                            <span className="health-label">Chunks</span>
                            <Activity size={16} className="health-icon" />
                        </div>
                        <strong className="health-value-num">{indexedChunks}</strong>
                        <span className="health-subtext">Indexed vector embeddings</span>
                    </div>
                </div>

                {/* Data Maintenance Subsection */}
                <div className="maintenance-box">
                    <div className="maintenance-info">
                        <strong>Data Maintenance</strong>
                        <p>Scan and clean up any orphaned database records, chunks, or vector embeddings that no longer have valid source files.</p>
                    </div>
                    <button
                        type="button"
                        className="btn-secondary btn-maintenance"
                        onClick={() => setShowCleanupModal(true)}
                        disabled={isCleaning}
                    >
                        <Trash2 size={15} />
                        <span>Clean up orphaned data</span>
                    </button>
                </div>
            </section>

            {/* ─── MODAL: DOCUMENT DELETE CONFIRMATION ────────────────────────── */}
            {deleteTarget && (
                <div className="modal-backdrop" role="presentation" onClick={() => !isDeleting && setDeleteTarget(null)}>
                    <div className="modal-card modal-compact" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header-simple">
                            <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Delete &ldquo;{deleteTarget.filename}&rdquo;?</h3>
                        </div>
                        <p className="modal-body-text">
                            This will permanently remove the uploaded file and its indexed knowledge from your assistant.
                        </p>
                        <div className="file-target-highlight">
                            📄 {deleteTarget.filename}
                        </div>
                        <div className="modal-actions-flex">
                            <button
                                type="button"
                                className="btn-secondary"
                                onClick={() => setDeleteTarget(null)}
                                disabled={isDeleting}
                            >
                                Cancel
                            </button>
                            <button
                                type="button"
                                className="btn-danger"
                                onClick={handleConfirmDelete}
                                disabled={isDeleting}
                            >
                                {isDeleting ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
                                <span>{isDeleting ? 'Deleting...' : 'Delete'}</span>
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ─── MODAL: ORPHAN CLEANUP CONFIRMATION ──────────────────────────── */}
            {showCleanupModal && (
                <div className="modal-backdrop" role="presentation" onClick={() => !isCleaning && setShowCleanupModal(false)}>
                    <div className="modal-card modal-compact" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header-simple">
                            <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Clean up orphaned data?</h3>
                        </div>
                        <p className="modal-body-text">
                            This will scan physical disk files, MongoDB document metadata, chunk records, and vector embeddings to remove any unlinked or inconsistent records.
                        </p>
                        <div className="modal-actions-flex">
                            <button
                                type="button"
                                className="btn-secondary"
                                onClick={() => setShowCleanupModal(false)}
                                disabled={isCleaning}
                            >
                                Cancel
                            </button>
                            <button
                                type="button"
                                className="btn-primary"
                                onClick={handleExecuteOrphanCleanup}
                                disabled={isCleaning}
                            >
                                {isCleaning ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                                <span>{isCleaning ? 'Cleaning...' : 'Clean Up'}</span>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SettingsView;
