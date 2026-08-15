import { useState } from 'react';
import {
    AlertCircle, Check, ChevronDown, ExternalLink, File, FileSpreadsheet,
    FileText, MoreVertical, Plus, RefreshCw, Search, Trash2, X
} from 'lucide-react';
import { deleteFile, fetchFileDetails, getFileUrl, reprocessFile } from '../api';
import UploadModal from './UploadModal';

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function formatDate(isoString) {
    if (!isoString) return 'Unknown date';
    try {
        const d = new Date(isoString);
        return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
        return isoString;
    }
}

const DocumentsView = ({ files, onFilesChange, onSummarize }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [typeFilter, setTypeFilter] = useState('All Types');
    const [statusFilter, setStatusFilter] = useState('All Statuses');
    const [showTypeDropdown, setShowTypeDropdown] = useState(false);
    const [showStatusDropdown, setShowStatusDropdown] = useState(false);
    const [showActionMenu, setShowActionMenu] = useState(null);
    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [reprocessingFile, setReprocessingFile] = useState(null);
    const [selectedDetailFile, setSelectedDetailFile] = useState(null);
    const [fileDetailsData, setFileDetailsData] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [selectedFiles, setSelectedFiles] = useState([]);

    const getFileIcon = (filename) => {
        const ext = filename.split('.').pop().toLowerCase();
        if (['xlsx', 'xls', 'csv'].includes(ext)) return <FileSpreadsheet size={18} color="var(--success-primary)" />;
        if (['pdf'].includes(ext)) return <FileText size={18} color="var(--danger-primary)" />;
        if (['docx', 'doc'].includes(ext)) return <FileText size={18} color="var(--accent-primary)" />;
        return <File size={18} color="var(--text-secondary)" />;
    };

    const handleDelete = async (filename) => {
        if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
            try {
                await deleteFile(filename);
                onFilesChange();
                setShowActionMenu(null);
            } catch (error) {
                console.error("Delete error:", error);
                window.alert(`Delete failed: ${error.message}`);
            }
        }
    };

    const handleReprocess = async (filename) => {
        setReprocessingFile(filename);
        try {
            await reprocessFile(filename);
            await onFilesChange();
            setShowActionMenu(null);
        } catch (error) {
            console.error('Reprocess error:', error);
            window.alert(`Could not reprocess ${filename}: ${error.message}`);
        } finally {
            setReprocessingFile(null);
        }
    };

    const handleOpenDetails = async (filename) => {
        setSelectedDetailFile(filename);
        setLoadingDetails(true);
        try {
            const data = await fetchFileDetails(filename);
            setFileDetailsData(data);
        } catch (error) {
            console.error('Failed to load file details:', error);
            setFileDetailsData(null);
        } finally {
            setLoadingDetails(false);
        }
    };

    const toggleSelectFile = (filename) => {
        if (selectedFiles.includes(filename)) {
            setSelectedFiles(selectedFiles.filter(f => f !== filename));
        } else {
            setSelectedFiles([...selectedFiles, filename]);
        }
    };

    const toggleSelectAll = () => {
        if (selectedFiles.length === filteredFiles.length) {
            setSelectedFiles([]);
        } else {
            setSelectedFiles(filteredFiles.map(f => f.name));
        }
    };

    const handleBatchDelete = async () => {
        if (selectedFiles.length === 0) return;
        if (window.confirm(`Are you sure you want to delete ${selectedFiles.length} selected document(s)?`)) {
            for (const fn of selectedFiles) {
                try { await deleteFile(fn); } catch (e) { console.error(e); }
            }
            setSelectedFiles([]);
            onFilesChange();
        }
    };

    const filteredFiles = files.filter(file => {
        const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase());
        const ext = file.name.split('.').pop().toUpperCase();
        const matchesType = typeFilter === 'All Types' || ext === typeFilter;
        const normalizedStatus = (file.status || '').toUpperCase();
        const matchesStatus = statusFilter === 'All Statuses' || normalizedStatus === statusFilter.toUpperCase();
        return matchesSearch && matchesType && matchesStatus;
    });

    const fileTypes = ['All Types', ...new Set(files.map(f => f.name.split('.').pop().toUpperCase()))];
    const statusTypes = ['All Statuses', 'READY', 'PROCESSING', 'FAILED'];

    return (
        <div className="view-container">
            <header className="view-header">
                <div>
                    <h2>Documents</h2>
                    <p>Manage and inspect your uploaded knowledge base documents</p>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    {selectedFiles.length > 0 && (
                        <button className="btn-danger" onClick={handleBatchDelete}>
                            <Trash2 size={15} /> Delete {selectedFiles.length} item(s)
                        </button>
                    )}
                    <button className="btn-primary" onClick={() => setIsUploadModalOpen(true)}>
                        <Plus size={18} /> Upload document
                    </button>
                </div>
            </header>

            <div className="documents-toolbar">
                <div className="toolbar-left">
                    <div className="doc-search-box">
                        <Search size={17} color="var(--text-muted)" />
                        <input
                            type="text"
                            placeholder="Search documents…"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                        {searchTerm && <button onClick={() => setSearchTerm('')}><X size={14} /></button>}
                    </div>

                    {/* Status Dropdown */}
                    <div className="custom-select-wrap">
                        <button className="select-btn" onClick={() => setShowStatusDropdown(!showStatusDropdown)}>
                            <span>{statusFilter}</span> <ChevronDown size={15} />
                        </button>
                        {showStatusDropdown && (
                            <div className="select-dropdown-menu">
                                {statusTypes.map(st => (
                                    <button key={st} onClick={() => { setStatusFilter(st); setShowStatusDropdown(false); }}>
                                        <span>{st}</span> {statusFilter === st && <Check size={14} color="var(--accent-primary)" />}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Type Dropdown */}
                    <div className="custom-select-wrap">
                        <button className="select-btn" onClick={() => setShowTypeDropdown(!showTypeDropdown)}>
                            <span>{typeFilter}</span> <ChevronDown size={15} />
                        </button>
                        {showTypeDropdown && (
                            <div className="select-dropdown-menu">
                                {fileTypes.map(ft => (
                                    <button key={ft} onClick={() => { setTypeFilter(ft); setShowTypeDropdown(false); }}>
                                        <span>{ft}</span> {typeFilter === ft && <Check size={14} color="var(--accent-primary)" />}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {filteredFiles.length === 0 ? (
                <div className="empty-files-box" style={{ padding: '60px 20px', border: '1px dashed var(--border-subtle)', borderRadius: 'var(--radius-lg)' }}>
                    <FileText size={42} color="var(--text-muted)" />
                    <p style={{ margin: '12px 0 4px', fontWeight: 600, fontSize: '1rem' }}>No documents found</p>
                    <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                        {searchTerm ? 'Try adjusting your search or filters.' : 'Upload documents to build your RAG knowledge base.'}
                    </span>
                    {!searchTerm && (
                        <button className="btn-primary" style={{ marginTop: '16px' }} onClick={() => setIsUploadModalOpen(true)}>
                            <Plus size={16} /> Upload your first document
                        </button>
                    )}
                </div>
            ) : (
                <div className="documents-table-wrap">
                    <table className="documents-table">
                        <thead>
                            <tr>
                                <th style={{ width: '40px' }}>
                                    <input
                                        type="checkbox"
                                        checked={selectedFiles.length === filteredFiles.length && filteredFiles.length > 0}
                                        onChange={toggleSelectAll}
                                    />
                                </th>
                                <th>Name</th>
                                <th>Size</th>
                                <th>Uploaded</th>
                                <th>Status</th>
                                <th style={{ textAlign: 'right' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredFiles.map((file) => {
                                const isSelected = selectedFiles.includes(file.name);
                                const statusClass = (file.status || 'Ready').toLowerCase();
                                return (
                                    <tr key={file.name} className={isSelected ? 'selected-row' : ''}>
                                        <td>
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleSelectFile(file.name)}
                                            />
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                                {getFileIcon(file.name)}
                                                <span style={{ fontWeight: 600 }}>{file.name}</span>
                                            </div>
                                        </td>
                                        <td>{formatBytes(file.size_bytes)}</td>
                                        <td>{formatDate(file.uploaded_on)}</td>
                                        <td>
                                            <span className={`status-badge ${statusClass}`}>
                                                ● {file.status || 'Ready'}
                                            </span>
                                        </td>
                                        <td style={{ textAlign: 'right' }}>
                                            <div style={{ position: 'relative', display: 'inline-block' }}>
                                                <button
                                                    className="btn-icon"
                                                    onClick={() => setShowActionMenu(showActionMenu === file.name ? null : file.name)}
                                                >
                                                    <MoreVertical size={16} />
                                                </button>

                                                {showActionMenu === file.name && (
                                                    <div className="select-dropdown-menu" style={{ right: 0, left: 'auto' }}>
                                                        <button onClick={() => { onSummarize(file.name); setShowActionMenu(null); }}>
                                                            <span>Summarize in Chat</span>
                                                        </button>
                                                        <button onClick={() => { window.open(getFileUrl(file.name), '_blank'); setShowActionMenu(null); }}>
                                                            <span><ExternalLink size={14} /> Open / Download</span>
                                                        </button>
                                                        <button onClick={() => { handleOpenDetails(file.name); setShowActionMenu(null); }}>
                                                            <span>View Chunks</span>
                                                        </button>
                                                        <button onClick={() => handleReprocess(file.name)}>
                                                            <span><RefreshCw size={14} /> Reprocess</span>
                                                        </button>
                                                        <button className="danger" onClick={() => handleDelete(file.name)}>
                                                            <span style={{ color: 'var(--danger-primary)' }}>Delete</span>
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Custom Upload Modal */}
            <UploadModal
                isOpen={isUploadModalOpen}
                onClose={() => setIsUploadModalOpen(false)}
                onUploadSuccess={() => onFilesChange()}
            />

            {/* Details Modal */}
            {selectedDetailFile && (
                <div className="modal-backdrop" role="presentation" onClick={() => setSelectedDetailFile(null)}>
                    <div className="modal-card" style={{ maxWidth: '640px' }} onClick={(e) => e.stopPropagation()}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                            <h3 style={{ margin: 0 }}>Document Details: {selectedDetailFile}</h3>
                            <button className="btn-icon" onClick={() => setSelectedDetailFile(null)}><X size={18} /></button>
                        </div>
                        {loadingDetails ? (
                            <p>Loading document details…</p>
                        ) : fileDetailsData ? (
                            <div style={{ maxHeight: '360px', overflowY: 'auto' }}>
                                <p><strong>Filename:</strong> {fileDetailsData.filename}</p>
                                <p><strong>Status:</strong> {fileDetailsData.status}</p>
                                <p><strong>Indexed Chunks:</strong> {fileDetailsData.chunks?.length || 0}</p>
                                <h4>Chunks Sample:</h4>
                                {fileDetailsData.chunks?.slice(0, 5).map((chunk, idx) => (
                                    <div key={idx} style={{ padding: '10px', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--bg-muted)', marginBottom: '8px', fontSize: '0.82rem' }}>
                                        {chunk.content || chunk.text || 'No text snippet available.'}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p>Could not load details for this document.</p>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default DocumentsView;
