import { useCallback, useEffect, useRef, useState } from 'react';
import {
    Archive, Bot, Brain, Camera, Check, ChevronDown, Copy, Edit2, FileText, Folder,
    Image as ImageIcon, Maximize2, Menu, Mic, Moon, MoreVertical, Paperclip, Pencil, Pin, Plus,
    RotateCw, Send, Share2, Sparkles, Square, Sun, ThumbsDown, ThumbsUp, Trash2, X
} from 'lucide-react';
import { attachFileToChat, createChat, fetchChatMessages, getFileUrl, streamChat, uploadFile } from '../api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import FilesInChatDrawer from './FilesInChatDrawer';
import PdfViewerModal from './PdfViewerModal';

const ACCEPTED_DOCUMENTS = '.pdf,.txt,.docx,.xlsx,.xls,.md,.csv';

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function normalizeSource(source) {
    if (typeof source === 'string') return { name: source, pages: [] };
    return { name: source?.name || source?.source || 'Document', pages: source?.pages || [] };
}


const ChatArea = ({
    messages, setMessages, chatId, chatTitle, files,
    selectedDocuments = [], onSelectedDocumentsChange,
    responseMode = 'balanced', onResponseModeChange,
    onNewChat, onMessageSent, onFirstMessage, onFileUploaded, onOpenDocuments, onToggleSidebar,
    isDarkMode, setIsDarkMode, onToggleTheme, onThemeChange, preloadedMessage, clearPreloadedMessage,
    onDeleteChat, onTogglePinChat, onToggleArchiveChat, isPinned, isArchived, onDeleteDocument,
    onChatIdCreated, showToast,
}) => {
    const selectedDocument = selectedDocuments[0] || null;
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [showVoiceOverlay, setShowVoiceOverlay] = useState(false);
    const [showToolsMenu, setShowToolsMenu] = useState(false);
    const [showDocumentMenu, setShowDocumentMenu] = useState(false);
    const [showModeMenu, setShowModeMenu] = useState(false);
    const [showChatMenu, setShowChatMenu] = useState(false);
    const [isFilesDrawerOpen, setIsFilesDrawerOpen] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [activeSelectorDocMenu, setActiveSelectorDocMenu] = useState(null);
    const [selectorDeleteTarget, setSelectorDeleteTarget] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageError, setImageError] = useState('');
    const [isDragging, setIsDragging] = useState(false);
    const [copiedId, setCopiedId] = useState(null);
    const [feedback, setFeedback] = useState({});
    const [previewImageModal, setPreviewImageModal] = useState(null);
    const [editingMessageId, setEditingMessageId] = useState(null);
    const [editText, setEditText] = useState('');
    const [pdfViewerState, setPdfViewerState] = useState({ isOpen: false, documentName: null, initialPage: 1 });

    const handleOpenPdfViewer = (docName, pageNum = 1) => {
        const cleanName = typeof docName === 'string'
            ? docName.replace(/\s*\(\s*(?:pages?|p\.)\s*[\d\s,–-]+(?:\s*total\s*:\s*\d+\s*pages?)?\s*\)/gi, '').trim()
            : docName;
        setPdfViewerState({
            isOpen: true,
            documentName: cleanName || selectedDocument || 'Document',
            initialPage: Number(pageNum) || 1
        });
    };

    const messagesEndRef = useRef(null);
    const messagesRef = useRef(messages);
    const recognitionRef = useRef(null);
    const abortControllerRef = useRef(null);
    const activeMessageIdRef = useRef(null);
    const fileInputRef = useRef(null);
    const imageInputRef = useRef(null);
    const cameraInputRef = useRef(null);
    const textareaRef = useRef(null);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => { messagesRef.current = messages; }, [messages]);

    useEffect(() => () => {
        abortControllerRef.current?.abort();
        recognitionRef.current?.stop();
    }, []);

    const prevChatIdRef = useRef(chatId);
    useEffect(() => {
        const prevId = prevChatIdRef.current;
        prevChatIdRef.current = chatId;
        if (prevId === chatId) return undefined;
        if (isLoading) return undefined;
        if (!chatId) {
            setMessages([]);
            return undefined;
        }
        let isCurrentChat = true;
        const loadMessages = async () => {
            try {
                const data = await fetchChatMessages(chatId);
                if (isCurrentChat) setMessages(data.messages || []);
            } catch (error) {
                console.error('Error loading chat messages:', error);
            }
        };
        void loadMessages();
        return () => { isCurrentChat = false; };
    }, [chatId, isLoading, setMessages]);

    useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

    useEffect(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.style.height = 'auto';
        textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
    }, [input]);

    const clearSelectedImage = () => {
        setSelectedImage(null);
        setImageError('');
    };

    const prepareImage = (file) => {
        if (!file) return;
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            setImageError('Choose a JPG, PNG, or WEBP image.');
            return;
        }
        if (file.size > 8 * 1024 * 1024) {
            setImageError('Images must be 8 MB or smaller.');
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            const preview = String(reader.result || '');
            const data = preview.split(',', 2)[1];
            if (!data) return setImageError('This image could not be read. Please choose another one.');
            setSelectedImage({ preview, data, mimeType: file.type, name: file.name });
            setImageError('');
        };
        reader.onerror = () => setImageError('This image could not be read. Please choose another one.');
        reader.readAsDataURL(file);
    };

    const handleImageSelect = (event) => {
        prepareImage(event.target.files?.[0]);
        event.target.value = '';
    };

    const [uploadingFileName, setUploadingFileName] = useState(null);
    const [uploadError, setUploadError] = useState(null);

    const handleFileUpload = async (file) => {
        if (!file) return;
        setUploadError(null);
        const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.xlsx', '.xls', '.csv', '.md'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            setUploadError(`PDF upload failed: Invalid file type '${ext}'. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`);
            return;
        }

        const MAX_SIZE = 25 * 1024 * 1024; // 25 MB
        if (file.size > MAX_SIZE) {
            setUploadError(`PDF upload failed: File too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum size is 25 MB.`);
            return;
        }

        try {
            setUploadingFileName(file.name);
            let activeChatId = chatId;
            if (!activeChatId) {
                try {
                    const created = await createChat();
                    if (created?.conversation_id) {
                        activeChatId = created.conversation_id;
                        onChatIdCreated?.(activeChatId);
                    }
                } catch (e) {
                    console.error('Failed to create chat session during upload:', e);
                }
            }

            const uploadRes = await uploadFile(file, activeChatId);
            setUploadingFileName(null);
            const docIdentifier = uploadRes?.filename || file.name || uploadRes?.document_id;
            onFileUploaded(docIdentifier);

            // Auto-attach file to active chat
            if (activeChatId) {
                try {
                    await attachFileToChat(activeChatId, uploadRes?.document_id || docIdentifier);
                } catch (e) {
                    console.error('Failed to attach file to chat:', e);
                }
            }

        } catch (error) {
            setUploadingFileName(null);
            console.error('Upload error:', error);
            let msg = error.message || 'Upload failed';
            if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('Network request failed')) {
                msg = 'Backend server is unavailable';
            }
            setUploadError(`PDF upload failed: ${msg}`);
        }
    };

    const handleDocumentSelect = (event) => {
        void handleFileUpload(event.target.files?.[0]);
        event.target.value = '';
    };

    const handleStopGenerating = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        const activeId = activeMessageIdRef.current;
        if (activeId) {
            setMessages((previous) => previous.map((message) => message.id === activeId ? { ...message, isStreaming: false, interrupted: true } : message));
        }
        setIsLoading(false);
    };

    const handleStartEdit = (message) => {
        setEditingMessageId(message.id);
        setEditText(message.text || '');
    };

    const handleCancelEdit = () => {
        setEditingMessageId(null);
        setEditText('');
    };

    const handleSend = useCallback(async (messageOverride = null, options = {}) => {
        const textToSend = messageOverride ?? input;
        const imageToSend = selectedImage;
        if ((!textToSend.trim() && !imageToSend) || isLoading) return;

        const userMessage = textToSend.trim();
        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const { editTurnId = null, replaceFromMessageId = null } = options;

        setIsLoading(true);
        setInput('');
        clearSelectedImage();
        setEditingMessageId(null);
        setEditText('');

        const aiMessageId = `ai_${Date.now()}`;

        try {
            let activeChatId = chatId;
            if (!activeChatId) {
                const created = await createChat();
                if (created?.conversation_id) {
                    activeChatId = created.conversation_id;
                    onChatIdCreated?.(activeChatId);
                } else {
                    throw new Error("Unable to create a new chat session.");
                }
            }

            const cleanDocIdent = selectedDocRecord?.document_id || selectedDocRecord?.filename || (typeof selectedDocument === 'string' ? selectedDocument.replace(/\s*\(\s*(?:pages?|p\.)\s*[\d\s,–-]+(?:\s*total\s*:\s*\d+\s*pages?)?\s*\)/gi, '').trim() : null);

            if (cleanDocIdent && activeChatId) {
                try {
                    await attachFileToChat(activeChatId, cleanDocIdent);
                } catch (e) {
                    console.error('Auto-attach context file failed:', e);
                }
            }

            const docsToSend = cleanDocIdent ? [cleanDocIdent] : (Array.isArray(selectedDocuments) && selectedDocuments.length > 0 ? selectedDocuments : []);

            const isFirstUserMessage = !messagesRef.current.some((message) => message.sender === 'user');
            if (isFirstUserMessage && userMessage) onFirstMessage(userMessage);

            if (replaceFromMessageId) {
                // Editing an existing message turn or regenerating
                setMessages((previous) => {
                    const targetIdx = previous.findIndex(m => m.id === replaceFromMessageId);
                    if (targetIdx !== -1) {
                        const truncated = previous.slice(0, targetIdx + 1).map((m, idx) => {
                            if (idx === targetIdx && m.sender === 'user') {
                                return { ...m, text: userMessage, time: now };
                            }
                            return m;
                        });
                        return [...truncated, { id: aiMessageId, turn_id: editTurnId, text: '', sender: 'ai', time: now, isStreaming: true }];
                    }
                    return previous;
                });
            } else {
                setMessages((previous) => [...previous, {
                    id: `user_${Date.now()}`,
                    text: userMessage,
                    sender: 'user',
                    time: now,
                    imagePreview: imageToSend?.preview,
                    imageName: imageToSend?.name,
                }, { id: aiMessageId, text: '', sender: 'ai', time: now, isStreaming: true }]);
            }

            activeMessageIdRef.current = aiMessageId;
            abortControllerRef.current = new AbortController();

            await streamChat(
                userMessage,
                activeChatId,
                imageToSend ? { data: imageToSend.data, mime_type: imageToSend.mimeType, name: imageToSend.name } : null,
                docsToSend,
                responseMode,
                (token) => setMessages((previous) => previous.map((message) => message.id === aiMessageId ? { ...message, text: message.text + token } : message)),
                (fullText, sources, sourceDetails) => {
                    const finalOutput = fullText || "I didn't receive a response from the AI model. Please try again.";
                    const isErr = !fullText;
                    setMessages((previous) => previous.map((message) => message.id === aiMessageId ? {
                        ...message, text: finalOutput, sources, sourceDetails, isStreaming: false, isError: isErr
                    } : message));
                    onMessageSent();
                },
                abortControllerRef.current.signal,
                editTurnId
            );
        } catch (error) {
            if (error.name === 'AbortError') {
                setMessages((previous) => previous.map((message) => message.id === aiMessageId ? { ...message, isStreaming: false, interrupted: true } : message));
            } else {
                console.error('Chat error:', error);
                const errorText = error.message || 'An unexpected error occurred while communicating with Personal AI.';
                setMessages((previous) => {
                    const exists = previous.some(m => m.id === aiMessageId);
                    if (exists) {
                        return previous.map((message) => message.id === aiMessageId ? {
                            ...message, text: errorText, isStreaming: false, isError: true,
                        } : message);
                    } else {
                        return [...previous, { id: aiMessageId, text: errorText, sender: 'ai', time: now, isStreaming: false, isError: true }];
                    }
                });
            }
        } finally {
            setIsLoading(false);
            activeMessageIdRef.current = null;
            abortControllerRef.current = null;
        }
    }, [chatId, input, isLoading, onChatIdCreated, onFirstMessage, onMessageSent, selectedDocument, selectedDocuments, selectedImage, setMessages, responseMode]);

    const handleSaveEdit = useCallback((messageId, turnId, newText) => {
        if (!newText.trim() || isLoading) return;
        void handleSend(newText, { editTurnId: turnId, replaceFromMessageId: messageId });
    }, [handleSend, isLoading]);

    useEffect(() => {
        if (!preloadedMessage) return undefined;
        const timer = window.setTimeout(() => {
            void handleSend(preloadedMessage);
            clearPreloadedMessage();
        }, 0);
        return () => window.clearTimeout(timer);
    }, [clearPreloadedMessage, handleSend, preloadedMessage]);

    const startListening = () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            window.alert('Speech recognition is not supported in this browser.');
            return;
        }
        recognitionRef.current?.stop();
        setShowVoiceOverlay(true);
        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.maxAlternatives = 1;
        let finalTranscript = '';
        recognition.onstart = () => setIsListening(true);
        recognition.onresult = (event) => {
            let interimTranscript = '';
            for (let index = event.resultIndex; index < event.results.length; index += 1) {
                if (event.results[index].isFinal) finalTranscript += event.results[index][0].transcript;
                else interimTranscript += event.results[index][0].transcript;
            }
            setInput(finalTranscript + interimTranscript);
        };
        recognition.onerror = (event) => {
            if (event.error !== 'no-speech') { setIsListening(false); setShowVoiceOverlay(false); }
        };
        recognition.onend = () => setIsListening(false);
        recognition.start();
    };

    const stopListening = () => {
        recognitionRef.current?.stop();
        setShowVoiceOverlay(false);
        setIsListening(false);
    };

    const copyAnswer = async (id, text) => {
        try {
            await navigator.clipboard.writeText(text);
            setCopiedId(id);
            window.setTimeout(() => setCopiedId(null), 1600);
        } catch {
            window.alert('Copy is not available in this browser.');
        }
    };

    const handleRegenerate = useCallback((aiMessageId) => {
        if (isLoading) return;
        const msgList = messagesRef.current;
        const index = msgList.findIndex((m) => m.id === aiMessageId);
        let userMsg = null;
        if (index > 0) {
            for (let i = index - 1; i >= 0; i--) {
                if (msgList[i].sender === 'user') {
                    userMsg = msgList[i];
                    break;
                }
            }
        }
        if (!userMsg) {
            userMsg = [...msgList].reverse().find((m) => m.sender === 'user');
        }
        if (userMsg) {
            const aiMsg = msgList[index];
            const turnId = aiMsg?.turn_id || userMsg?.turn_id;
            void handleSend(userMsg.text, { editTurnId: turnId, replaceFromMessageId: userMsg.id });
        }
    }, [handleSend, isLoading]);


    const handleShare = async () => {
        try {
            if (navigator.share) {
                await navigator.share({
                    title: chatTitle || 'Personal AI',
                    url: window.location.href,
                });
            } else {
                await navigator.clipboard.writeText(window.location.href);
                showToast?.('Conversation link copied to clipboard!');
            }
        } catch (e) {
            console.error('Share error:', e);
        }
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setIsDragging(false);
        const file = event.dataTransfer.files?.[0];
        if (!file) return;
        if (file.type.startsWith('image/')) prepareImage(file);
        else void handleFileUpload(file);
    };

    const handlePaste = (event) => {
        const image = [...(event.clipboardData?.files || [])].find((file) => file.type.startsWith('image/'));
        if (image) {
            event.preventDefault();
            prepareImage(image);
        }
    };

    const hasVisibleConversation = messages.length > 0;
    const sanitizedSelectedDoc = typeof selectedDocument === 'string'
        ? selectedDocument.replace(/\s*\(\s*(?:pages?|p\.)\s*[\d\s,–-]+(?:\s*total\s*:\s*\d+\s*pages?)?\s*\)/gi, '').trim()
        : selectedDocument;

    const selectedDocRecord = files.find((file) => 
        file.document_id === selectedDocument || 
        file.id === selectedDocument || 
        file.filename === selectedDocument || 
        file.name === selectedDocument ||
        (sanitizedSelectedDoc && (
            file.document_id === sanitizedSelectedDoc || 
            file.id === sanitizedSelectedDoc || 
            file.filename === sanitizedSelectedDoc || 
            file.name === sanitizedSelectedDoc
        ))
    );
    const displayDocName = selectedDocRecord?.filename || selectedDocRecord?.name || sanitizedSelectedDoc || selectedDocument;

    let rawStatus = 'Not found';
    if (uploadingFileName === selectedDocument || (sanitizedSelectedDoc && uploadingFileName === sanitizedSelectedDoc)) {
        rawStatus = 'Uploading';
    } else if (selectedDocRecord) {
        rawStatus = String(selectedDocRecord.status || (selectedDocRecord.processing_status === 'ready' ? 'READY' : 'PROCESSING')).toUpperCase();
    } else if (!selectedDocument) {
        rawStatus = '';
    }

    const isReady = (rawStatus === 'READY') && ((selectedDocRecord?.chunk_count > 0) || (selectedDocRecord?.page_count > 0) || (selectedDocRecord?.total_pages > 0));
    const selectedDocumentStatus = rawStatus === 'Uploading' ? 'Uploading...' : 
        (rawStatus === 'PROCESSING' || rawStatus === 'UPLOADING') ? 'Processing...' : 
        rawStatus === 'FAILED' ? 'Failed' : 
        rawStatus === 'Not found' ? 'Not found' :
        isReady ? 'Ready' : (rawStatus ? 'Processing...' : '');

    const selectedDocumentError = selectedDocRecord?.error;

    return (
        <main className="main-content" onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={(event) => { if (event.currentTarget === event.target) setIsDragging(false); }} onDrop={handleDrop}>
            <header className="chat-header">
                <div className="chat-header-left">
                    <button className="mobile-menu-button" onClick={onToggleSidebar} title="Open navigation" aria-label="Open navigation"><Menu size={20} /></button>
                    <div>
                        <h1>{chatTitle || 'New chat'}</h1>
                        <p>{selectedDocument ? `Context: ${displayDocName}` : 'Personal AI'}</p>
                    </div>
                </div>

                <div className="chat-header-actions">
                    {/* Compact Theme Toggle Icon Button */}
                    <button
                        className="header-theme-toggle"
                        onClick={() => {
                            if (typeof onToggleTheme === 'function') onToggleTheme();
                            else if (typeof onThemeChange === 'function') onThemeChange();
                            else if (typeof setIsDarkMode === 'function') setIsDarkMode(!isDarkMode);
                        }}
                        title={isDarkMode ? 'Use light theme' : 'Use dark theme'}
                        aria-label="Toggle theme"
                    >
                        {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
                    </button>

                    {/* Share Button with Icon + Label */}
                    <button
                        className={`header-share-btn ${(!chatId || messages.length === 0) ? 'disabled' : ''}`}
                        onClick={handleShare}
                        disabled={!chatId || messages.length === 0}
                        title={chatId && messages.length > 0 ? "Share conversation" : "No conversation to share"}
                        aria-label="Share conversation"
                    >
                        <Share2 size={15} />
                        <span className="share-btn-label">Share</span>
                    </button>

                    {/* 3-Dot Options Button + Anchored Dropdown */}
                    <div className="header-menu-container">
                        <button className="header-menu-btn" onClick={() => setShowChatMenu(!showChatMenu)} title="Chat options" aria-label="Chat options">
                            <MoreVertical size={18} />
                        </button>

                        {showChatMenu && (
                            <div className="chat-menu-dropdown">
                                <button onClick={() => { fileInputRef.current?.click(); setShowChatMenu(false); }}>
                                    <Paperclip size={15} color="var(--accent-primary)" /> <span>Upload file</span>
                                </button>
                                <button onClick={() => { setIsFilesDrawerOpen(true); setShowChatMenu(false); }}>
                                    <Folder size={15} color="var(--accent-primary)" /> <span>View files in chat</span>
                                </button>

                                {chatId && (
                                    <button onClick={() => { onTogglePinChat?.(chatId); setShowChatMenu(false); }}>
                                        <Pin size={15} /> <span>{isPinned ? 'Unpin chat' : 'Pin chat'}</span>
                                    </button>
                                )}

                                {chatId && (
                                    <button onClick={() => { onToggleArchiveChat?.(chatId); setShowChatMenu(false); }}>
                                        <Archive size={15} /> <span>{isArchived ? 'Unarchive' : 'Archive'}</span>
                                    </button>
                                )}

                                {chatId && (
                                    <button className="danger" onClick={() => { setShowDeleteModal(true); setShowChatMenu(false); }}>
                                        <Trash2 size={15} color="var(--danger-primary)" /> <span style={{ color: 'var(--danger-primary)' }}>Delete</span>
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </header>

            <div className="chat-scroll-area">
                <div className="chat-messages">
                    {!hasVisibleConversation && !isLoading ? (
                        <WelcomeHero
                            onChoose={(msg) => void handleSend(msg)}
                            onUpload={() => fileInputRef.current?.click()}
                            onSelectDocument={() => setShowDocumentMenu(true)}
                        />
                    ) : messages.map((message, index) => (
                        <MessageItem
                            key={message.id || `${message.sender}-${index}`}
                            message={message}
                            isEditing={editingMessageId === message.id}
                            editText={editText}
                            onEditTextChange={setEditText}
                            onStartEdit={handleStartEdit}
                            onCancelEdit={handleCancelEdit}
                            onSaveEdit={handleSaveEdit}
                            copiedId={copiedId}
                            feedback={feedback[message.id]}
                            onCopy={copyAnswer}
                            onFeedback={(value) => setFeedback((current) => ({ ...current, [message.id]: value }))}
                            onRegenerate={handleRegenerate}
                            onRetry={(msgId) => {
                                const msgIdx = messages.findIndex(m => m.id === msgId);
                                if (msgIdx > 0 && messages[msgIdx - 1]?.sender === 'user') {
                                    const userMsg = messages[msgIdx - 1];
                                    void handleSend(userMsg.text, { editTurnId: userMsg.turn_id, replaceFromMessageId: userMsg.id });
                                }
                            }}
                            onImageClick={(src) => setPreviewImageModal(src)}
                            onOpenDocument={handleOpenPdfViewer}
                        />
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Composer Shell */}
            <div className="composer-shell">
                <div className="composer-wrap">
                    {uploadError && (
                        <div className="upload-error-banner" style={{ color: 'var(--danger-primary)', fontSize: '0.82rem', marginBottom: '8px', padding: '6px 10px', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span>⚠️ {uploadError}</span>
                            <button type="button" onClick={() => setUploadError(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', padding: '0 4px', fontSize: '0.9rem' }}>✕</button>
                        </div>
                    )}

                    {selectedDocument && (
                        <div className="composer-document-attachment">
                            <div className="attachment-icon-badge">
                                <FileText size={18} color="var(--accent-primary)" />
                            </div>
                            <div className="attachment-details">
                                <span className="attachment-name" title={displayDocName}>{displayDocName}</span>
                                <div className="attachment-subtext">
                                    <span>{selectedDocRecord?.file_type || 'PDF'}</span>
                                    {selectedDocRecord?.size_bytes > 0 && <span> · {formatBytes(selectedDocRecord.size_bytes)}</span>}
                                    <span> · </span>
                                    <span className={`file-status-badge ${selectedDocumentStatus.includes('Uploading') || selectedDocumentStatus.includes('Processing') ? 'status-processing' : selectedDocumentStatus === 'Failed' ? 'status-failed' : 'status-ready'}`} title={selectedDocumentError || selectedDocumentStatus}>
                                        {selectedDocumentStatus === 'Ready' ? 'Indexed ✓' : selectedDocumentStatus}
                                    </span>
                                </div>
                            </div>
                            <div className="attachment-action-buttons">
                                {selectedDocRecord && (
                                    <button
                                        type="button"
                                        className="btn-attachment-action"
                                        onClick={() => handleOpenPdfViewer(selectedDocRecord?.filename || displayDocName, 1)}
                                        title="View document preview"
                                    >
                                        View
                                    </button>
                                )}
                                <button
                                    type="button"
                                    className="btn-attachment-action danger"
                                    onClick={async () => {
                                        if (window.confirm(`Are you sure you want to permanently delete "${displayDocName}"?\n\nThis will physically remove the file, all chunks, and vector index records.`)) {
                                            await onDeleteDocument?.(selectedDocRecord?.document_id || displayDocName);
                                            onSelectedDocumentsChange?.(null);
                                        }
                                    }}
                                    title="Delete document permanently"
                                >
                                    <Trash2 size={14} />
                                </button>
                                <button
                                    type="button"
                                    className="btn-attachment-action"
                                    onClick={() => onSelectedDocumentsChange?.(null)}
                                    title="Detach from current composer"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        </div>
                    )}


                    {selectedImage && (
                        <div className="composer-image-preview" style={{ padding: '8px', display: 'flex', gap: '10px', alignItems: 'center', backgroundColor: 'var(--bg-muted)', borderRadius: 'var(--radius-md)', marginBottom: '8px' }}>
                            <img src={selectedImage.preview} alt="Photo ready to analyze" style={{ width: '40px', height: '40px', borderRadius: '6px', objectFit: 'cover' }} />
                            <div style={{ flex: 1, fontSize: '0.82rem' }}>
                                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedImage.name || 'Attached photo'}</div>
                                <div style={{ color: 'var(--text-muted)' }}>Ready for visual analysis</div>
                            </div>
                            <button className="composer-action-btn" onClick={clearSelectedImage} title="Remove attached photo" aria-label="Remove attached photo">
                                <X size={16} />
                            </button>
                        </div>
                    )}

                    {imageError && <div style={{ color: 'var(--danger-primary)', fontSize: '0.8rem', marginBottom: '6px' }}>{imageError}</div>}

                    <div className="composer-container">
                        <div className="composer-input-row">
                            <button className="composer-plus-btn" onClick={() => setShowToolsMenu((value) => !value)} title="Add attachment" aria-label="Add attachment"><Plus size={22} /></button>

                            {showToolsMenu && (
                                <div className="select-dropdown-menu" style={{ bottom: '54px', left: '12px', top: 'auto', minWidth: '200px' }}>
                                    <button onClick={() => { fileInputRef.current?.click(); setShowToolsMenu(false); }}><Paperclip size={15} /> Upload document</button>
                                    <button onClick={() => { setShowDocumentMenu(true); setShowToolsMenu(false); }}><FileText size={15} /> Attach existing document</button>
                                    <button onClick={() => { imageInputRef.current?.click(); setShowToolsMenu(false); }}><ImageIcon size={15} /> Upload photo</button>
                                    <button onClick={() => { cameraInputRef.current?.click(); setShowToolsMenu(false); }}><Camera size={15} /> Use camera</button>
                                </div>
                            )}

                            {/* Hidden file inputs */}
                            <input ref={fileInputRef} type="file" className="visually-hidden-file-input" accept={ACCEPTED_DOCUMENTS} onChange={handleDocumentSelect} />
                            <input ref={imageInputRef} type="file" className="visually-hidden-file-input" accept="image/jpeg,image/png,image/webp" onChange={handleImageSelect} />
                            <input ref={cameraInputRef} type="file" className="visually-hidden-file-input" accept="image/jpeg,image/png,image/webp" capture="environment" onChange={handleImageSelect} />

                            <textarea
                                ref={textareaRef}
                                className="composer-textarea"
                                value={input}
                                onChange={(event) => setInput(event.target.value)}
                                onPaste={handlePaste}
                                onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void handleSend(); } }}
                                placeholder={selectedDocument ? `Ask about ${selectedDocument}…` : 'Ask anything...'}
                                rows={1}
                            />

                            <div className="composer-actions-group">
                                {/* Think Reasoning Mode Control */}
                                <button
                                    type="button"
                                    className={`composer-think-btn ${responseMode === 'deep' ? 'active' : ''}`}
                                    onClick={() => onResponseModeChange(responseMode === 'deep' ? 'balanced' : 'deep')}
                                    title="Toggle Think reasoning mode"
                                    aria-label="Toggle Think reasoning mode"
                                >
                                    <Brain size={15} />
                                    <span>Think</span>
                                </button>

                                <div style={{ position: 'relative' }}>
                                    <button className={`composer-action-btn ${selectedDocument ? 'active' : ''}`} onClick={() => setShowDocumentMenu((v) => !v)} title="Select document context" aria-label="Select document context">
                                        <FileText size={18} />
                                    </button>
                                    {showDocumentMenu && (
                                        <div className="select-dropdown-menu" style={{ right: 0, left: 'auto', bottom: '54px', top: 'auto', minWidth: '180px', padding: '4px' }}>
                                            <button
                                                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', borderRadius: 'var(--radius-sm)' }}
                                                onClick={() => { onSelectedDocumentsChange(null); setShowDocumentMenu(false); }}
                                            >
                                                <span style={{ fontWeight: !selectedDocument ? 600 : 400, fontSize: '0.88rem' }}>All documents</span>
                                                {!selectedDocument && <Check size={14} color="var(--accent-primary)" />}
                                            </button>
                                        </div>
                                    )}
                                </div>

                                <button className={`composer-action-btn ${isListening ? 'active' : ''}`} onClick={startListening} title="Voice dictation" aria-label="Voice dictation">
                                    <Mic size={18} />
                                </button>

                                {isLoading ? (
                                    <button className="send-button" onClick={handleStopGenerating} title="Stop generating" aria-label="Stop generating">
                                        <Square size={14} fill="currentColor" />
                                    </button>
                                ) : (
                                    <button className="send-button" onClick={() => void handleSend()} disabled={!input.trim() && !selectedImage} title="Send message" aria-label="Send message">
                                        <Send size={18} />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Right-side Files in Chat Drawer */}
            <FilesInChatDrawer
                isOpen={isFilesDrawerOpen}
                onClose={() => setIsFilesDrawerOpen(false)}
                chatId={chatId}
                onRemoveDocument={(docName) => {
                    if (selectedDocument === docName) onSelectedDocumentsChange(null);
                }}
                onFilesChange={onMessageSent}
                onFileUploaded={onFileUploaded}
            />

            {/* Destructive Delete Confirmation Modal */}
            {showDeleteModal && (
                <div className="modal-backdrop" role="presentation" onClick={() => setShowDeleteModal(false)}>
                    <div className="modal-card" style={{ maxWidth: '420px' }} onClick={(e) => e.stopPropagation()}>
                        <h3 style={{ margin: '0 0 10px', fontSize: '1.2rem', color: 'var(--text-primary)' }}>Delete conversation?</h3>
                        <p style={{ margin: '0 0 20px', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                            This will permanently delete this conversation and its messages. Attached documents will remain in your document library.
                        </p>
                        <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                            <button className="btn-secondary" onClick={() => setShowDeleteModal(false)}>Cancel</button>
                            <button className="btn-danger" onClick={() => { onDeleteChat?.(chatId); setShowDeleteModal(false); }}>Delete</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Image Lightbox Preview Modal */}
            {previewImageModal && (
                <div className="modal-backdrop image-lightbox-backdrop" role="presentation" onClick={() => setPreviewImageModal(null)}>
                    <div className="image-lightbox-card" onClick={(e) => e.stopPropagation()}>
                        <button className="image-lightbox-close" onClick={() => setPreviewImageModal(null)} title="Close preview" aria-label="Close preview">
                            <X size={20} />
                        </button>
                        <img src={previewImageModal} alt="Enlarged attachment preview" />
                    </div>
                </div>
            )}

            {/* Embedded PDF Viewer Modal with Jump-to-Page */}
            <PdfViewerModal
                isOpen={pdfViewerState.isOpen}
                onClose={() => setPdfViewerState({ isOpen: false, documentName: null, initialPage: 1 })}
                documentName={pdfViewerState.documentName}
                initialPage={pdfViewerState.initialPage}
            />

            {isDragging && <div className="modal-backdrop"><div><Paperclip size={28} /><p>Drop document or image here to upload</p></div></div>}
            {showVoiceOverlay && <VoiceOverlay onClose={stopListening} />}
        </main>
    );
};

/* Personal AI Landing Hero View */
const WelcomeHero = ({ onChoose, onUpload, onSelectDocument }) => {
    return (
        <section className="welcome-hero">
            <h1 className="welcome-greeting">How can I help you?</h1>
            <p className="welcome-description">

                Ask anything, write, code, analyze, or work with your files.
            </p>

            <div className="welcome-action-grid">
                <div className="welcome-card" onClick={() => onChoose('Learn a new concept or topic')}>
                    <div className="welcome-card-icon"><Sparkles size={20} /></div>
                    <div>
                        <h3>✨ Learn something</h3>
                        <span>Learn a concept or topic</span>
                    </div>
                </div>

                <div className="welcome-card" onClick={() => onChoose('Help me write, debug, or refactor code')}>
                    <div className="welcome-card-icon"><Bot size={20} /></div>
                    <div>
                        <h3>💻 Help me code</h3>
                        <span>Build, debug, or explain code</span>
                    </div>
                </div>

                <div className="welcome-card" onClick={() => onChoose('Draft an article or improve my writing')}>
                    <div className="welcome-card-icon"><FileText size={20} /></div>
                    <div>
                        <h3>📝 Write something</h3>
                        <span>Create or improve content</span>
                    </div>
                </div>

                <div className="welcome-card" onClick={onUpload}>
                    <div className="welcome-card-icon"><Paperclip size={20} /></div>
                    <div>
                        <h3>📄 Work with a file</h3>
                        <span>Analyze, summarize, or ask questions about a file</span>
                    </div>
                </div>

                <div className="welcome-card" onClick={() => onChoose('Let us brainstorm ideas for my project')}>
                    <div className="welcome-card-icon"><Brain size={20} /></div>
                    <div>
                        <h3>🧠 Brainstorm ideas</h3>
                        <span>Generate and refine ideas</span>
                    </div>
                </div>
            </div>
        </section>
    );
};

const MessageItem = ({
    message,
    isEditing,
    editText,
    onEditTextChange,
    onStartEdit,
    onCancelEdit,
    onSaveEdit,
    copiedId,
    feedback,
    onCopy,
    onFeedback,
    onRegenerate,
    onRetry,
    onImageClick,
    onOpenDocument
}) => {
    const sourceItems = message.sourceDetails?.length ? message.sourceDetails : message.sources || [];
    const isUser = message.sender === 'user';

    return (
        <article className={`message-row ${isUser ? 'from-user' : 'from-assistant'}`}>
            {!isUser && (
                <div className="assistant-avatar" title="Personal AI" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    <img src="/logo.png" alt="Personal AI" style={{ width: '22px', height: '22px', objectFit: 'contain' }} />
                </div>
            )}
            <div className="message-content">
                <div className={`message-bubble ${isUser ? 'user-message' : 'assistant-message'} ${message.isError ? 'has-error' : ''}`}>
                    {message.imagePreview && (
                        <div className="chat-image-wrapper">
                            <img
                                className="chat-image-preview"
                                src={message.imagePreview}
                                alt={message.imageName || 'Uploaded photo'}
                                onClick={() => onImageClick?.(message.imagePreview)}
                                title="Click to view full image"
                            />
                        </div>
                    )}
                    {message.isError ? (
                        <div className="message-error-box" style={{ padding: '4px 0', color: 'var(--danger-primary)', fontSize: '0.88rem' }}>
                            <div style={{ fontWeight: 600, marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>⚠️ Request Failed</div>
                            <div>{message.text}</div>
                            <button
                                className="btn-secondary"
                                style={{ marginTop: '10px', padding: '4px 12px', fontSize: '0.8rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                                onClick={() => onRetry?.(message.id)}
                            >
                                <RotateCw size={13} /> Retry request
                            </button>
                        </div>
                    ) : isEditing ? (
                        <div className="inline-message-editor">
                            <textarea
                                className="inline-edit-textarea"
                                value={editText}
                                onChange={(e) => onEditTextChange(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        onSaveEdit(message.id, message.turn_id, editText);
                                    } else if (e.key === 'Escape') {
                                        onCancelEdit();
                                    }
                                }}
                                rows={2}
                                autoFocus
                            />
                            <div className="inline-edit-actions">
                                <button type="button" className="btn-secondary btn-sm" onClick={onCancelEdit}>Cancel</button>
                                <button type="button" className="btn-primary btn-sm" onClick={() => onSaveEdit(message.id, message.turn_id, editText)} disabled={!editText.trim()}>Save & Submit</button>
                            </div>
                        </div>
                    ) : (
                        <>
                            {message.text && <MarkdownContent text={message.text} onOpenDocument={onOpenDocument} />}
                            {!isUser && message.isStreaming && !message.text && <ResponsePulse />}
                        </>
                    )}
                    {!isUser && !message.isError && sourceItems.length > 0 && (
                        <div className="sources-section">
                            <div className="sources-title">Sources</div>
                            <div className="source-list">
                                {sourceItems.map((source, index) => {
                                    const detail = normalizeSource(source);
                                    const primaryPage = detail.pages && detail.pages.length > 0 ? detail.pages[0] : 1;
                                    return (
                                        <button
                                            type="button"
                                            className="source-card source-card-interactive"
                                            key={`${detail.name}-${index}`}
                                            onClick={() => onOpenDocument?.(detail.name, primaryPage)}
                                            title={`Click to view ${detail.name} at page ${primaryPage}`}
                                        >
                                            <FileText size={14} color="var(--accent-primary)" />
                                            <strong>{detail.name}</strong>
                                            {detail.pages.length > 0 && <span className="source-page-badge">(p. {detail.pages.join(', ')})</span>}
                                            <span className="source-view-action">View page ↗</span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>
                {isUser && !message.isError && !isEditing && (
                    <div className="message-actions user-message-actions">
                        <button
                            className="action-btn"
                            onClick={() => onStartEdit?.(message)}
                            title="Edit message"
                            aria-label="Edit message"
                        >
                            <Pencil size={13} />
                            <span>Edit</span>
                        </button>
                        <button
                            className="action-btn"
                            onClick={() => onCopy(message.id, message.text)}
                            title="Copy message"
                            aria-label="Copy message"
                        >
                            {copiedId === message.id ? <><Check size={13} /> <span>Copied</span></> : <><Copy size={13} /> <span>Copy</span></>}
                        </button>
                    </div>
                )}
                {!isUser && !message.isStreaming && !message.isError && (
                    <div className="message-actions">
                        <button
                            className={`action-btn ${feedback === 'up' ? 'active' : ''}`}
                            onClick={() => onFeedback('up')}
                            title="Like response"
                            aria-label="Like response"
                        >
                            <ThumbsUp size={13} />
                            <span>Like</span>
                        </button>
                        <button
                            className={`action-btn ${feedback === 'down' ? 'active' : ''}`}
                            onClick={() => onFeedback('down')}
                            title="Dislike response"
                            aria-label="Dislike response"
                        >
                            <ThumbsDown size={13} />
                            <span>Dislike</span>
                        </button>
                        <button
                            className="action-btn"
                            onClick={() => onCopy(message.id, message.text)}
                            title="Copy response"
                            aria-label="Copy response"
                        >
                            {copiedId === message.id ? <><Check size={13} /> <span>Copied</span></> : <><Copy size={13} /> <span>Copy</span></>}
                        </button>
                        <button
                            className="action-btn"
                            onClick={() => onRegenerate?.(message.id)}
                            title="Regenerate response"
                            aria-label="Regenerate response"
                        >
                            <RotateCw size={13} />
                            <span>Regenerate</span>
                        </button>
                    </div>
                )}
            </div>
        </article>
    );
};

/* Code block with copy button and syntax highlighting */
const CodeBlock = ({ language, codeText }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(codeText);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1600);
        } catch {
            /* ignore fallback */
        }
    };

    const displayLang = (language || 'code').toLowerCase();

    return (
        <div className="code-block-wrapper">
            <div className="code-block-header">
                <span className="code-block-lang">{displayLang}</span>
                <button className="code-copy-btn" onClick={handleCopy} title="Copy code" aria-label="Copy code">
                    {copied ? <><Check size={13} /> Copied</> : <><Copy size={13} /> Copy</>}
                </button>
            </div>
            <div className="code-block-body">
                <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={displayLang !== 'code' && displayLang !== 'text' ? displayLang : 'text'}
                    PreTag="div"
                    customStyle={{
                        margin: 0,
                        padding: '14px 16px',
                        background: '#141720',
                        fontSize: '0.86rem',
                        lineHeight: '1.5',
                        fontFamily: "'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace",
                        overflowX: 'auto',
                    }}
                    codeTagProps={{
                        style: {
                            fontFamily: "'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace",
                        }
                    }}
                >
                    {codeText}
                </SyntaxHighlighter>
            </div>
        </div>
    );
};

const MarkdownContent = ({ text, onOpenDocument }) => {
    // Process in-text citations like [Document: filename, Page X] into clickable markdown links
    const processedText = (text || '').replace(/\[Document:\s*([^,\]]+)(?:,\s*Page\s*([0-9]+))?\]/gi, (match, docName, pageNum) => {
        const p = pageNum || '1';
        const cleanDoc = docName.trim();
        return `[📄 ${cleanDoc} (p. ${p})](#cite:${encodeURIComponent(cleanDoc)}:${p})`;
    });

    return (
        <div className="markdown-content">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    pre({ children }) {
                        return <>{children}</>;
                    },
                    code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '');
                        const codeText = String(children).replace(/\n$/, '');
                        const isMultiLine = codeText.includes('\n');

                        if (!match && !isMultiLine && inline !== false) {
                            return <code className="inline-code-badge" {...props}>{children}</code>;
                        }

                        const lang = match ? match[1] : '';
                        return <CodeBlock language={lang} codeText={codeText} />;
                    },
                    table({ children }) {
                        return <div className="table-wrapper"><table className="markdown-table">{children}</table></div>;
                    },
                    a({ href, children }) {
                        if (href && href.startsWith('#cite:')) {
                            const [, encDoc, pageStr] = href.split(':');
                            const pageNum = parseInt(pageStr, 10) || 1;
                            const docName = decodeURIComponent(encDoc || '');
                            return (
                                <button
                                    type="button"
                                    className="citation-pill-btn"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        onOpenDocument?.(docName, pageNum);
                                    }}
                                    title={`Click to view ${docName} at page ${pageNum}`}
                                >
                                    <span>{children}</span>
                                </button>
                            );
                        }
                        return <a href={href} target="_blank" rel="noopener noreferrer" className="markdown-link">{children}</a>;
                    }
                }}
            >
                {processedText}
            </ReactMarkdown>
        </div>
    );
};

const ResponsePulse = () => <div className="response-pulse" style={{ display: 'flex', gap: '4px', padding: '8px 0' }}><span>•</span><span>•</span><span>•</span></div>;

const VoiceOverlay = ({ onClose }) => <div className="modal-backdrop"><div className="modal-card" style={{ textCenter: 'center' }}><Mic size={36} color="var(--accent-primary)" /><p>Listening…</p><button className="btn-secondary" onClick={onClose}>Stop listening</button></div></div>;

export default ChatArea;
