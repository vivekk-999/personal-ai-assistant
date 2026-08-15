import { useEffect, useRef, useState } from 'react';
import { File, FileSpreadsheet, FileText, Folder, Image as ImageIcon, Loader2, Paperclip, Trash2, Unlink, X } from 'lucide-react';
import { deleteFile, detachFileFromChat, fetchChatFiles, uploadFile } from '../api';

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

const FilesInChatDrawer = ({ isOpen, onClose, chatId, onRemoveDocument, onFilesChange, onFileUploaded }) => {
    const [files, setFiles] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [confirmAction, setConfirmAction] = useState(null); // { type: 'detach' | 'delete', filename: string }
    const [isProcessingAction, setIsProcessingAction] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');
    const fileInputRef = useRef(null);

    const loadChatFiles = async () => {
        if (!chatId) {
            setFiles([]);
            return;
        }
        setIsLoading(true);
        setErrorMsg('');
        try {
            const data = await fetchChatFiles(chatId);
            setFiles(data.files || []);
        } catch (error) {
            console.error('Error fetching chat files:', error);
            setErrorMsg(error.message || 'Unable to load chat files');
            setFiles([]);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen && chatId) {
            void loadChatFiles();
        } else {
            setFiles([]);
            setConfirmAction(null);
            setErrorMsg('');
        }
    }, [isOpen, chatId]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape' && isOpen) {
                if (confirmAction) setConfirmAction(null);
                else onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, confirmAction, onClose]);

    if (!isOpen) return null;

    const handleConfirmAction = async () => {
        if (!confirmAction || isProcessingAction) return;
        const { type, filename, document_id } = confirmAction;
        const target = document_id || filename;
        setIsProcessingAction(true);
        setErrorMsg('');
        try {
            if (type === 'delete') {
                await deleteFile(target);
            } else {
                await detachFileFromChat(chatId, target);
            }
            setFiles((prev) => prev.filter((f) => (f.document_id || f.filename) !== target && f.filename !== filename));
            setConfirmAction(null);
            onRemoveDocument?.(filename);
            onFilesChange?.();
        } catch (error) {
            console.error(`Failed to ${type} file:`, error);
            setErrorMsg(`Could not ${type} file: ${error.message}`);
        } finally {
            setIsProcessingAction(false);
        }
    };


    const handleFileSelect = async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile || isUploading) return;
        setIsUploading(true);
        setErrorMsg('');
        try {
            const uploadRes = await uploadFile(selectedFile, chatId);
            await loadChatFiles();
            onFileUploaded?.(uploadRes);
            onFilesChange?.();
        } catch (error) {
            console.error('Upload failed:', error);
            setErrorMsg(error.message || 'Upload failed. Please try again.');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };


    const getIcon = (type) => {
        const t = (type || '').toUpperCase();
        if (['XLSX', 'XLS', 'CSV'].includes(t)) return <FileSpreadsheet size={20} color="var(--success-primary, #10b981)" />;
        if (['PDF'].includes(t)) return <FileText size={20} color="var(--danger-primary, #ef4444)" />;
        if (['DOCX', 'DOC'].includes(t)) return <FileText size={20} color="var(--accent-primary, #3b82f6)" />;
        if (['JPG', 'JPEG', 'PNG', 'WEBP', 'GIF'].includes(t)) return <ImageIcon size={20} color="#a855f7" />;
        return <File size={20} color="var(--text-secondary, #94a3b8)" />;
    };

    return (
        <div className="files-drawer-overlay" role="presentation" onClick={onClose}>
            <div className="files-drawer-card" onClick={(e) => e.stopPropagation()}>
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                />

                <header className="files-drawer-header">
                    <div className="files-drawer-title">
                        <Folder size={18} color="var(--accent-primary, #3b82f6)" />
                        <h3 style={{ textTransform: 'none', fontSize: '0.95rem' }}>
                            Files in this chat ({files.length})
                        </h3>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                            className="btn-icon"
                            title="Upload file to chat"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                            aria-label="Upload file"
                        >
                            {isUploading ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
                        </button>
                        <button className="btn-icon" onClick={onClose} aria-label="Close drawer"><X size={18} /></button>
                    </div>
                </header>

                <div className="files-drawer-body">
                    {errorMsg && (
                        <div style={{ padding: '10px 14px', marginBottom: '14px', borderRadius: '6px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171', fontSize: '0.82rem' }}>
                            {errorMsg}
                        </div>
                    )}

                    {isLoading ? (
                        <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                            <Loader2 size={24} className="animate-spin" style={{ margin: '0 auto 8px' }} />
                            <div>Loading attached files…</div>
                        </div>
                    ) : files.length === 0 ? (
                        <div className="empty-files-drawer">
                            <FileText size={48} color="var(--text-muted)" style={{ opacity: 0.5 }} />
                            <h4 style={{ margin: '14px 0 4px', fontSize: '1rem' }}>Files in this chat (0)</h4>
                            <p>No files uploaded to this conversation.</p>
                            <button
                                className="btn-upload-in-drawer"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isUploading}
                            >
                                {isUploading ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
                                {isUploading ? 'Uploading...' : 'Upload file'}
                            </button>
                        </div>
                    ) : (
                        <div className="files-drawer-list">
                            {files.map((file) => {
                                const extFormatted = (file.file_type || 'FILE').toUpperCase();
                                const sizeFormatted = formatBytes(file.size_bytes);
                                const pagesFormatted = (file.total_pages || file.page_count) ? ` • ${file.total_pages || file.page_count} pages` : '';
                                const statusLabel = file.status === 'PROCESSING' 
                                    ? (file.status_details || (file.stage ? `${file.stage}...` : 'Processing...'))
                                    : file.status === 'FAILED' 
                                    ? (file.error ? `Failed: ${file.error}` : 'Failed') 
                                    : '✓ Ready';

                                return (
                                    <div className="drawer-file-item" key={file.document_id || file.filename}>
                                        <div className="drawer-file-icon">{getIcon(file.file_type)}</div>
                                        <div className="drawer-file-info">
                                            <div className="drawer-filename" title={file.filename}>{file.filename}</div>
                                            <div className="drawer-file-meta">
                                                <span>{extFormatted} • {sizeFormatted}{pagesFormatted}</span>
                                                <span style={{ margin: '0 6px' }}>•</span>
                                                <span className={`status-${(file.status || 'Ready').toLowerCase()}`} title={statusLabel}>{statusLabel}</span>
                                            </div>
                                        </div>

                                        <div className="drawer-file-actions" style={{ display: 'flex', gap: '6px' }}>
                                            <button
                                                className="btn-file-remove"
                                                title={`Detach ${file.filename} from this chat only`}
                                                onClick={() => setConfirmAction({ type: 'detach', filename: file.filename, document_id: file.document_id })}
                                                disabled={isProcessingAction}
                                            >
                                                <Unlink size={13} />
                                                <span>Detach</span>
                                            </button>
                                            <button
                                                className="btn-file-remove"
                                                style={{ borderColor: 'rgba(239, 68, 68, 0.3)', color: 'var(--danger-primary, #ef4444)' }}
                                                title={`Permanently delete ${file.filename}`}
                                                onClick={() => setConfirmAction({ type: 'delete', filename: file.filename, document_id: file.document_id })}
                                                disabled={isProcessingAction}
                                            >
                                                <Trash2 size={13} />
                                                <span>Delete</span>
                                            </button>
                                        </div>
                                    </div>
                                );

                            })}
                        </div>
                    )}
                </div>
            </div>

            {/* Action Confirmation Dialog */}
            {confirmAction && (
                <div className="modal-backdrop" role="presentation" onClick={() => !isProcessingAction && setConfirmAction(null)}>
                    <div className="modal-card" style={{ maxWidth: '420px', padding: '24px' }} onClick={(e) => e.stopPropagation()}>
                        <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '8px', color: 'var(--text-primary)' }}>
                            {confirmAction.type === 'delete' ? 'Delete document permanently?' : 'Detach file from chat?'}
                        </h2>
                        <p style={{ fontSize: '0.88rem', lineHeight: '1.5', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                            {confirmAction.type === 'delete'
                                ? 'This will permanently remove the physical file, all indexed chunks, and vector embeddings across the entire system.'
                                : 'This document will be detached from this chat session, but will remain stored in your documents library.'}
                        </p>
                        <div className="file-target-highlight">
                            📄 {confirmAction.filename}
                        </div>
                        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
                            <button
                                className="btn-secondary"
                                style={{ padding: '8px 16px', fontSize: '0.88rem' }}
                                onClick={() => setConfirmAction(null)}
                                disabled={isProcessingAction}
                            >
                                Cancel
                            </button>
                            <button
                                className={confirmAction.type === 'delete' ? 'btn-danger' : 'btn-primary'}
                                style={{ padding: '8px 16px', fontSize: '0.88rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                                onClick={handleConfirmAction}
                                disabled={isProcessingAction}
                            >
                                {isProcessingAction ? <Loader2 size={14} className="animate-spin" /> : confirmAction.type === 'delete' ? <Trash2 size={14} /> : <Unlink size={14} />}
                                <span>{isProcessingAction ? 'Processing...' : confirmAction.type === 'delete' ? 'Delete Permanently' : 'Detach'}</span>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>

    );
};

export default FilesInChatDrawer;
