import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import DocumentsView from './components/DocumentsView';
import HistoryView from './components/HistoryView';
import SettingsView from './components/SettingsView';
import ViewFilesModal from './components/ViewFilesModal';
import { createChat, deleteChat, deleteFile, fetchChatMessages, fetchChats, fetchFiles, fetchHistory, renameChat, updateChat } from './api';
import './App.css';

const ACTIVE_CHAT_KEY = 'rag_active_conversation_id';
const CHAT_META_KEY = 'rag_chat_metadata';

function readStoredValue(key, fallback = null) {
    if (typeof window === 'undefined') return fallback;
    try {
        const value = window.localStorage.getItem(key);
        return value ? JSON.parse(value) : fallback;
    } catch {
        return fallback;
    }
}

function readStoredChatId() {
    if (typeof window === 'undefined') return null;
    try {
        return window.localStorage.getItem(ACTIVE_CHAT_KEY);
    } catch {
        return null;
    }
}

function AppShell() {
    const navigate = useNavigate();
    const [files, setFiles] = useState([]);
    const [history, setHistory] = useState([]);
    const [conversations, setConversations] = useState([]);
    const [chatId, setChatId] = useState(() => readStoredChatId());
    const [messages, setMessages] = useState([]);
    const [chatMeta, setChatMeta] = useState(() => readStoredValue(CHAT_META_KEY, {}));
    const [pendingDocument, setPendingDocument] = useState(null);
    const [themeMode, setThemeMode] = useState(() => readStoredValue('rag_theme_mode', 'system'));
    const [responseMode, setResponseMode] = useState(() => readStoredValue('rag_response_mode', 'balanced'));
    const [isDarkMode, setIsDarkMode] = useState(() => {
        const storedTheme = readStoredValue('rag_theme_mode', 'system');
        if (storedTheme === 'dark') return true;
        if (storedTheme === 'light') return false;
        return typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    });
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => readStoredValue('rag_sidebar_collapsed', typeof window !== 'undefined' && window.innerWidth <= 768));
    const [preloadedMessage, setPreloadedMessage] = useState(null);
    const [viewFilesChat, setViewFilesChat] = useState(null);
    const [toast, setToast] = useState(null); // { id, message, type }

    useEffect(() => {
        try { window.localStorage.setItem('rag_sidebar_collapsed', JSON.stringify(isSidebarCollapsed)); } catch {}
    }, [isSidebarCollapsed]);

    const showToast = useCallback((msg, type = 'success') => {
        const text = typeof msg === 'string' ? msg : (msg?.message || msg?.text || 'Notification');
        const toastType = (typeof msg === 'object' && msg?.type) ? msg.type : type;
        const id = Date.now();
        setToast({ id, message: text, type: toastType });
    }, []);

    useEffect(() => {
        if (!toast) return;
        const timer = setTimeout(() => {
            setToast((current) => (current?.id === toast.id ? null : current));
        }, 3500);
        return () => clearTimeout(timer);
    }, [toast]);

    const loadMessages = useCallback(async (conversationId) => {
        if (!conversationId) {
            setMessages([]);
            return;
        }
        const data = await fetchChatMessages(conversationId);
        setMessages(data.messages || []);
    }, []);

    const loadData = useCallback(async () => {
        const [filesData, historyData, chatsData] = await Promise.all([fetchFiles(), fetchHistory(), fetchChats(true)]);
        setFiles(filesData.files || []);
        setHistory(historyData.history || []);
        setConversations(chatsData.chats || []);
        return chatsData.chats || [];
    }, []);

    const toggleTheme = useCallback(() => {
        setThemeMode((prev) => {
            if (prev === 'dark') return 'light';
            if (prev === 'light') return 'dark';
            return isDarkMode ? 'light' : 'dark';
        });
    }, [isDarkMode]);

    useEffect(() => {
        const updateTheme = () => {
            let dark = false;
            if (themeMode === 'dark') {
                dark = true;
            } else if (themeMode === 'light') {
                dark = false;
            } else {
                dark = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            setIsDarkMode(dark);
            if (typeof document !== 'undefined') {
                document.body.classList.toggle('dark', dark);
                document.documentElement.classList.toggle('dark', dark);
                document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
            }
        };

        updateTheme();
        try { window.localStorage.setItem('rag_theme_mode', JSON.stringify(themeMode)); } catch {}

        if (themeMode === 'system' && typeof window !== 'undefined' && window.matchMedia) {
            const mql = window.matchMedia('(prefers-color-scheme: dark)');
            const handler = () => updateTheme();
            mql.addEventListener('change', handler);
            return () => mql.removeEventListener('change', handler);
        }
    }, [themeMode]);

    const handleResponseModeChange = useCallback((mode) => {
        setResponseMode(mode);
        try { window.localStorage.setItem('rag_response_mode', JSON.stringify(mode)); } catch {}
    }, []);

    useEffect(() => {
        try {
            if (chatId) window.localStorage.setItem(ACTIVE_CHAT_KEY, chatId);
            else window.localStorage.removeItem(ACTIVE_CHAT_KEY);
        } catch {}
    }, [chatId]);

    useEffect(() => {
        try { window.localStorage.setItem(CHAT_META_KEY, JSON.stringify(chatMeta)); } catch {}
    }, [chatMeta]);


    useEffect(() => {
        let cancelled = false;
        const initialize = async () => {
            try {
                const serverChats = await loadData();
                if (cancelled) return;
                const storedId = readStoredChatId();
                const validStoredChat = serverChats.find((chat) => chat.conversation_id === storedId && !chat.archived);
                const firstActiveChat = serverChats.find((chat) => !chat.archived);
                const nextId = validStoredChat?.conversation_id || firstActiveChat?.conversation_id || null;
                setChatId(nextId);
                await loadMessages(nextId);
            } catch (error) {
                console.error('Unable to load data from the server:', error);
                if (!cancelled) setMessages([]);
            }
        };
        void initialize();
        return () => { cancelled = true; };
    }, [loadData, loadMessages]);

    // Automatic status polling for processing documents
    useEffect(() => {
        const isProcessing = files.some((file) => {
            const st = (file.status || '').toLowerCase();
            return ['uploading', 'processing', 'extracting text', 'chunking', 'generating summary', 'generating embeddings', 'indexing'].includes(st);
        });
        if (!isProcessing) return;

        const interval = setInterval(() => {
            void loadData();
        }, 1500);

        return () => clearInterval(interval);
    }, [files, loadData]);

    const updateChatMeta = useCallback((id, changes) => {
        if (!id) return;
        setChatMeta((current) => ({ ...current, [id]: { ...current[id], ...changes } }));
    }, []);

    const startNewChat = useCallback(() => {
        setChatId(null);
        setMessages([]);
        setPreloadedMessage(null);
        setPendingDocument(null);
        navigate('/');
        if (window.innerWidth <= 768) setIsSidebarCollapsed(true);
        return null;
    }, [navigate]);

    const selectChat = useCallback(async (conversationId) => {
        if (!conversationId) return;
        try {
            setChatId(conversationId);
            setPreloadedMessage(null);
            await loadMessages(conversationId);
            navigate('/');
            if (window.innerWidth <= 768) setIsSidebarCollapsed(true);
        } catch (error) {
            console.error('Unable to load conversation:', error);
            showToast(error.message || 'Unable to load this conversation.', 'error');
        }
    }, [loadMessages, navigate, showToast]);

    const handleSelectDocuments = useCallback((docs) => {

        const normalized = Array.isArray(docs) ? docs : (docs ? [docs] : []);
        if (!chatId) {
            setPendingDocument(normalized.length > 0 ? normalized : null);
            return;
        }
        updateChatMeta(chatId, { selectedDocuments: normalized });
    }, [chatId, updateChatMeta]);

    const handleFileUploaded = useCallback((filename) => {
        if (filename) {
            handleSelectDocuments([filename]);
        }
        void loadData();
    }, [handleSelectDocuments, loadData]);

    const handleSummarize = useCallback(async (filename) => {
        try {
            const created = await createChat();
            if (created?.conversation_id) {
                setChatId(created.conversation_id);
                setMessages([]);
                updateChatMeta(created.conversation_id, { selectedDocuments: [filename] });
                setPreloadedMessage(`Summarize the document: ${filename}`);
                navigate('/');
            }
        } catch (error) {
            console.error('Summarize failed:', error);
        }
    }, [navigate, updateChatMeta]);

    const extractChatId = useCallback((param) => {
        if (!param) return null;
        if (typeof param === 'string') return param;
        return param.conversationId || param.id || param.conversation_id || null;
    }, []);

    const handleRenameChat = useCallback(async (conversationId, title) => {
        const targetId = extractChatId(conversationId);
        if (!targetId) return;
        await renameChat(targetId, title);
        await loadData();
    }, [extractChatId, loadData]);

    const handleTogglePin = useCallback(async (chat) => {
        const targetId = extractChatId(chat);
        if (!targetId) return;
        try {
            const conversation = conversations.find(c => (c.conversation_id === targetId || c.id === targetId));
            const currentPinned = typeof chat === 'object' && chat.pinned !== undefined ? chat.pinned : Boolean(conversation?.pinned);
            await updateChat(targetId, { pinned: !currentPinned });
            await loadData();
            showToast(!currentPinned ? 'Chat pinned.' : 'Chat unpinned.');
        } catch (error) {
            console.error('Unable to update pin state:', error);
            showToast('Unable to update pin status.');
        }
    }, [conversations, extractChatId, loadData, showToast]);

    const handleToggleArchive = useCallback(async (chat) => {
        const targetId = extractChatId(chat);
        if (!targetId) return;
        try {
            const conversation = conversations.find(c => (c.conversation_id === targetId || c.id === targetId));
            const currentArchived = typeof chat === 'object' && chat.archived !== undefined ? chat.archived : Boolean(conversation?.archived);
            await updateChat(targetId, { archived: !currentArchived });
            await loadData();
            showToast(!currentArchived ? 'Chat archived.' : 'Chat unarchived.');
            if (!currentArchived && targetId === chatId) {
                setChatId(null);
                setMessages([]);
                navigate('/');
            }
        } catch (error) {
            console.error('Unable to update archive state:', error);
            showToast('Unable to archive chat. Please try again.');
        }
    }, [chatId, conversations, extractChatId, loadData, navigate, showToast]);

    const handleDeleteChat = useCallback(async (chat) => {
        const targetId = extractChatId(chat);
        if (!targetId) return;
        try {
            console.log('Initiating delete for conversation:', targetId);
            await deleteChat(targetId);
            setChatMeta((current) => {
                const next = { ...current };
                delete next[targetId];
                return next;
            });
            if (targetId === chatId) {
                setChatId(null);
                setMessages([]);
                setPendingDocument(null);
                navigate('/');
            }
            await loadData();
        } catch (error) {
            console.error('Delete conversation failed:', error);
            showToast('Unable to delete conversation. Please try again.');
        }
    }, [chatId, extractChatId, loadData, navigate, showToast]);

    const handleDeleteDocument = useCallback(async (filename) => {
        if (!filename) return;
        try {
            await deleteFile(filename);
            if (pendingDocument === filename || (Array.isArray(pendingDocument) && pendingDocument.includes(filename))) {
                setPendingDocument(null);
            }
            setChatMeta((current) => {
                const next = { ...current };
                for (const key of Object.keys(next)) {
                    if (next[key]?.selectedDocuments) {
                        next[key].selectedDocuments = next[key].selectedDocuments.filter((d) => d !== filename);
                    }
                }
                return next;
            });
            await loadData();
            showToast(`Document "${filename}" permanently deleted.`, 'success');
        } catch (error) {
            console.error('Delete document error:', error);
            showToast(`Unable to delete document: ${error.message}`, 'error');
        }
    }, [loadData, pendingDocument, showToast]);

    const refreshAfterMessage = useCallback(async () => {
        try { await loadData(); } catch (error) { console.error('Unable to refresh conversation list:', error); }
    }, [loadData]);

    const activeServerChat = useMemo(
        () => conversations.find((conversation) => (conversation.conversation_id === chatId || conversation.id === chatId)),
        [chatId, conversations],
    );
    const activeChat = useMemo(() => {
        let rawSelected = chatId
            ? chatMeta[chatId]?.selectedDocuments ?? (chatMeta[chatId]?.selectedDocument ? [chatMeta[chatId]?.selectedDocument] : (activeServerChat?.document_name ? [activeServerChat.document_name] : []))
            : (Array.isArray(pendingDocument) ? pendingDocument : (pendingDocument ? [pendingDocument] : []));
        return {
            title: activeServerChat?.title || 'New chat',
            selectedDocuments: Array.isArray(rawSelected) ? rawSelected : [],
            pinned: Boolean(activeServerChat?.pinned),
            archived: Boolean(activeServerChat?.archived),
        };
    }, [activeServerChat, chatId, chatMeta, pendingDocument]);

    return (
        <div className={`app-container ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
            <Sidebar
                files={files}
                history={history}
                conversations={conversations}
                chatId={chatId}
                isCollapsed={isSidebarCollapsed}
                isDarkMode={isDarkMode}
                onToggle={() => setIsSidebarCollapsed((value) => !value)}
                onNewChat={startNewChat}
                onSelectChat={selectChat}
                onRenameChat={handleRenameChat}
                onDeleteChat={handleDeleteChat}
                onTogglePin={handleTogglePin}
                onToggleArchive={handleToggleArchive}
                onViewFiles={(chat) => setViewFilesChat(chat)}
                onFilesChange={loadData}
                onThemeChange={toggleTheme}
                showToast={showToast}
            />
            <div className="mobile-sidebar-scrim" onClick={() => setIsSidebarCollapsed(true)} aria-hidden="true" />
            <Routes>
                <Route path="/" element={<ChatArea
                    messages={messages}
                    setMessages={setMessages}
                    chatId={chatId}
                    chatTitle={activeChat.title}
                    files={files}
                    selectedDocuments={activeChat.selectedDocuments}
                    onSelectedDocumentsChange={handleSelectDocuments}
                    responseMode={responseMode}
                    onResponseModeChange={handleResponseModeChange}
                    onNewChat={startNewChat}
                    onMessageSent={refreshAfterMessage}
                    onFirstMessage={() => {}}
                    onFileUploaded={handleFileUploaded}
                    onOpenDocuments={() => navigate('/documents')}
                    onToggleSidebar={() => setIsSidebarCollapsed((value) => !value)}
                    isDarkMode={isDarkMode}
                    setIsDarkMode={setIsDarkMode}
                    onToggleTheme={toggleTheme}
                    onThemeChange={toggleTheme}
                    preloadedMessage={preloadedMessage}
                    clearPreloadedMessage={() => setPreloadedMessage(null)}
                    onDeleteChat={handleDeleteChat}
                    onTogglePinChat={handleTogglePin}
                    onToggleArchiveChat={handleToggleArchive}
                    onDeleteDocument={handleDeleteDocument}
                    onChatIdCreated={setChatId}
                    isPinned={activeChat?.pinned}
                    isArchived={activeChat?.archived}
                    showToast={showToast}
                />} />
                <Route path="/documents" element={<DocumentsView files={files} onFilesChange={loadData} onSummarize={handleSummarize} showToast={showToast} />} />
                <Route path="/history" element={<HistoryView history={history} onClearHistory={loadData} />} />
                <Route path="/settings" element={
                    <SettingsView
                        themeMode={themeMode}
                        setThemeMode={setThemeMode}
                        isDarkMode={isDarkMode}
                        setIsDarkMode={setIsDarkMode}
                        responseMode={responseMode}
                        setResponseMode={handleResponseModeChange}
                        files={files}
                        onFilesChange={loadData}
                        onDeleteDocument={handleDeleteDocument}
                        onToggleSidebar={() => setIsSidebarCollapsed((value) => !value)}
                        showToast={showToast}
                    />
                } />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>

            {viewFilesChat && (
                <ViewFilesModal chat={viewFilesChat} onClose={() => setViewFilesChat(null)} onFilesChange={loadData} />
            )}

            {toast && (
                <div className="toast-container" role="status" aria-live="polite">
                    <div className={`toast-card toast-${toast.type || 'success'}`}>
                        <span className="toast-icon">
                            {toast.type === 'error' ? '✕' : toast.type === 'info' ? 'ℹ' : '✓'}
                        </span>
                        <span className="toast-message">{toast.message}</span>
                        <button
                            type="button"
                            className="toast-close-btn"
                            onClick={() => setToast(null)}
                            aria-label="Close notification"
                        >
                            ✕
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

function App() {
    return <BrowserRouter><AppShell /></BrowserRouter>;
}

export default App;

