import { useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, FileText, Loader2, UploadCloud, X } from 'lucide-react';
import { uploadFile } from '../api';

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

const ACCEPTED_EXTENSIONS = '.pdf,.txt,.docx,.xlsx,.xls,.md,.csv';

const UploadModal = ({ isOpen, onClose, onUploadSuccess }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [uploadingFile, setUploadingFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null); // 'success' | 'error'
    const [errorMessage, setErrorMessage] = useState('');
    const fileInputRef = useRef(null);

    if (!isOpen) return null;

    const handleFile = async (file) => {
        if (!file) return;
        setUploadingFile(file);
        setIsUploading(true);
        setUploadStatus(null);
        setErrorMessage('');

        try {
            await uploadFile(file);
            setUploadStatus('success');
            setTimeout(() => {
                onUploadSuccess?.(file.name);
                onClose();
                setUploadingFile(null);
                setUploadStatus(null);
            }, 1200);
        } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus('error');
            setErrorMessage(error.message || 'File upload failed. Please try again.');
        } finally {
            setIsUploading(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void handleFile(file);
    };

    return (
        <div className="modal-backdrop" role="presentation" onClick={onClose}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700 }}>Upload Document</h3>
                    <button className="btn-icon" onClick={onClose} aria-label="Close modal"><X size={18} /></button>
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    className="visually-hidden-file-input"
                    accept={ACCEPTED_EXTENSIONS}
                    onChange={(e) => handleFile(e.target.files?.[0])}
                />

                <div
                    className={`upload-dropzone ${isDragging ? 'dragging' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <div className="upload-icon-wrap">
                        <UploadCloud size={28} />
                    </div>
                    <p style={{ margin: '0 0 6px', fontWeight: 600, fontSize: '0.95rem' }}>
                        Drag & drop file here, or <span style={{ color: 'var(--accent-primary)', textDecoration: 'underline' }}>browse</span>
                    </p>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        Maximum file size: 50 MB
                    </span>
                    <div className="format-tags">
                        <span>PDF</span> • <span>DOCX</span> • <span>TXT</span> • <span>MD</span> • <span>XLSX</span> • <span>CSV</span>
                    </div>
                </div>

                {uploadingFile && (
                    <div style={{ marginTop: '16px', padding: '12px', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--bg-muted)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <FileText size={20} color="var(--accent-primary)" />
                            <div>
                                <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{uploadingFile.name}</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{formatBytes(uploadingFile.size)}</div>
                            </div>
                        </div>
                        <div>
                            {isUploading && <Loader2 className="animate-spin" size={18} color="var(--accent-primary)" />}
                            {uploadStatus === 'success' && <CheckCircle2 size={18} color="var(--success-primary)" />}
                            {uploadStatus === 'error' && <AlertCircle size={18} color="var(--danger-primary)" />}
                        </div>
                    </div>
                )}

                {errorMessage && (
                    <div style={{ marginTop: '12px', fontSize: '0.82rem', color: 'var(--danger-primary)', textAlign: 'center' }}>
                        {errorMessage}
                    </div>
                )}
            </div>
        </div>
    );
};

export default UploadModal;
