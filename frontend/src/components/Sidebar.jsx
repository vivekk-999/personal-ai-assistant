import { useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import {
    Archive, ArchiveRestore, BookOpen, Bot, ChevronDown, ChevronRight,
    ChevronsLeft, ChevronsRight, FileText, Folder, LogOut, MessageSquare,
    MoreHorizontal, Moon, Pencil, Pin, PinOff, Plus, Search, Settings, Sun, Trash2, User, X
} from 'lucide-react';

const DAY = 24 * 60 * 60 * 1000;

function chatGroup(dateValue) {
    const date = dateValue ? new Date(dateValue) : new Date();
    const now = new Date();
    const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const delta = startToday - new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
    if (delta <= 0) return 'Today';
    if (delta <= DAY) return 'Yesterday';
    if (delta <= 7 * DAY) return 'Previous 7 days';
    return 'Older';
}

function createConversations(serverConversations) {
    const conversations = new Map();
    for (const conversation of serverConversations || []) {
        if (!conversation.conversation_id && !conversation.id) continue;
        const convId = conversation.conversation_id || conversation.id;

        conversations.set(convId, {
            id: convId,
            conversationId: convId,
            title: conversation.title || 'New chat',
            pinned: Boolean(conversation.pinned),
            archived: Boolean(conversation.archived),
            questions: conversation.preview ? [conversation.preview] : [],
            documents: conversation.selected_document_ids || (conversation.document_name ? [conversation.document_name] : []),
            updatedAt: conversation.updated_at || conversation.created_at || conversation.time,
        });
    }

    return [...conversations.values()]
        .sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0));
}

const Sidebar = ({
    files, conversations: serverConversations, chatId, isCollapsed, isDarkMode, onToggle, onNewChat,
    onSelectChat, onRenameChat, onDeleteChat, onTogglePin, onToggleArchive, onViewFiles, onThemeChange,
}) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [search, setSearch] = useState('');
    const [openMenu, setOpenMenu] = useState(null);
    const [profileOpen, setProfileOpen] = useState(false);
    const [showArchivedView, setShowArchivedView] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(true);
    const [renameChat, setRenameChat] = useState(null);
    const [renameValue, setRenameValue] = useState('');
    const [deleteChat, setDeleteChat] = useState(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const searchInputRef = useRef(null);

    // Close menus when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (openMenu && !event.target.closest('.conversation-more-wrap')) {
                setOpenMenu(null);
            }
            if (profileOpen && !event.target.closest('.profile-wrap')) {
                setProfileOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [openMenu, profileOpen]);

    const conversations = useMemo(() => createConversations(serverConversations), [serverConversations]);

    const matchedChats = useMemo(() => {
        const query = search.trim().toLowerCase();
        if (!query) return conversations;
        return conversations.filter((chat) => [chat.title, ...chat.questions, ...chat.documents]
            .filter(Boolean)
            .some((value) => value.toLowerCase().includes(query)));
    }, [conversations, search]);

    const pinnedChats = useMemo(() => {
        return matchedChats.filter((chat) => !chat.archived && chat.pinned);
    }, [matchedChats]);

    const activeUnpinnedChats = useMemo(() => {
        return matchedChats.filter((chat) => !chat.archived && !chat.pinned);
    }, [matchedChats]);

    const archivedChats = useMemo(() => {
        return matchedChats.filter((chat) => chat.archived);
    }, [matchedChats]);

    const groups = useMemo(() => {
        const grouped = new Map();
        for (const chat of activeUnpinnedChats) {
            const label = chatGroup(chat.updatedAt);
            if (!grouped.has(label)) grouped.set(label, []);
            grouped.get(label).push(chat);
        }
        return ['Today', 'Yesterday', 'Previous 7 days', 'Older']
            .map((label) => [label, grouped.get(label) || []])
            .filter(([, chats]) => chats.length);
    }, [activeUnpinnedChats]);

    const closeDrawerAfterSelection = () => {
        if (window.innerWidth <= 768 && !isCollapsed) onToggle();
    };

    const confirmDelete = async () => {
        if (!deleteChat) return;
        if (!deleteChat.conversationId) {
            setDeleteChat(null);
            return;
        }
        setIsDeleting(true);
        try {
            await onDeleteChat(deleteChat);
            setDeleteChat(null);
        } catch (error) {
            console.error('Unable to delete conversation:', error);
        } finally {
            setIsDeleting(false);
        }
    };

    const renderChatRow = (chat) => (
        <div key={chat.id} className={`conversation-row ${chat.id === chatId && location.pathname === '/' ? 'is-active' : ''}`}>
            <button className="conversation-select" onClick={() => { onSelectChat(chat.id); closeDrawerAfterSelection(); }} title={chat.title}>
                <MessageSquare size={15} />
                <span className="sidebar-label conversation-title">{chat.title}</span>
                {chat.documents.length > 0 && <FileText className="document-dot sidebar-label" size={13} aria-label="Has document sources" />}
            </button>
            <div className="conversation-more-wrap">
                <button
                    className="conversation-more sidebar-label"
                    onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenu(openMenu === chat.id ? null : chat.id);
                    }}
                    aria-label={`More options for ${chat.title}`}
                >
                    <MoreHorizontal size={17} />
                </button>
                {openMenu === chat.id && (
                    <div className="conversation-menu" onClick={(e) => e.stopPropagation()}>
                        <button onClick={() => { onViewFiles(chat); setOpenMenu(null); }}>
                            <BookOpen size={14} /> View files in chat
                        </button>
                        <button onClick={() => { setRenameChat(chat); setRenameValue(chat.title); setOpenMenu(null); }}>
                            <Pencil size={14} /> Rename
                        </button>
                        <button onClick={() => { onTogglePin(chat); setOpenMenu(null); }}>
                            {chat.pinned ? <PinOff size={14} /> : <Pin size={14} />} {chat.pinned ? 'Unpin chat' : 'Pin chat'}
                        </button>
                        <button className="danger" onClick={() => { setDeleteChat(chat); setOpenMenu(null); }}>
                            <Trash2 size={14} /> Delete
                        </button>
                    </div>
                )}
            </div>
        </div>
    );


    return (
        <aside className={`sidebar ${isCollapsed ? 'is-collapsed' : ''}`} aria-label="Main navigation">
            {/* ─── EXPANDED SIDEBAR CONTENT ─── */}
            <div className="sidebar-expanded-content">
                <div className="sidebar-top">
                    <div className="brand-row">
                        <button className="brand" onClick={() => navigate('/')} title="Personal AI home">
                            <span className="brand-mark">
                                <img src="/logo.png" alt="Personal AI Logo" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
                            </span>
                            <span className="sidebar-label brand-title">Personal AI</span>
                        </button>
                        <button
                            className="sidebar-collapse-btn sidebar-label"
                            onClick={onToggle}
                            title="Collapse sidebar («)"
                            aria-label="Collapse sidebar"
                        >
                            <ChevronsLeft size={18} />
                        </button>
                    </div>

                    <button className="sidebar-new-chat" onClick={() => { onNewChat(); closeDrawerAfterSelection(); }} title="Start a new chat" aria-label="Start a new chat">
                        <Plus size={17} /> <span className="sidebar-label">New chat</span>
                    </button>

                    <div className="chat-search">
                        <Search size={16} />
                        <input
                            ref={searchInputRef}
                            value={search}
                            onChange={(event) => setSearch(event.target.value)}
                            placeholder="Search chats…"
                            aria-label="Search chats"
                        />
                        {search && <button className="clear-search-btn" onClick={() => setSearch('')} aria-label="Clear search"><X size={14} /></button>}
                    </div>

                    {/* Navigation */}
                    <div className="sidebar-tools-group">
                        <NavLink to="/" className={({ isActive }) => `sidebar-tool-link ${isActive && location.pathname === '/' ? 'active' : ''}`} onClick={closeDrawerAfterSelection}>
                            <MessageSquare size={17} /> <span>Chats</span>
                        </NavLink>
                        <NavLink to="/settings" className={({ isActive }) => `sidebar-tool-link ${isActive ? 'active' : ''}`} onClick={closeDrawerAfterSelection}>
                            <Settings size={17} /> <span>Settings</span>
                        </NavLink>
                    </div>
                </div>

                {/* History Section */}
                <div className="conversation-list">
                    <div className="history-section-header" onClick={() => setIsHistoryOpen(prev => !prev)}>
                        <div className="history-header-left">
                            <span className="history-section-title">RECENTS</span>
                            <span className="history-chevron">{isHistoryOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span>
                        </div>
                    </div>

                    {isHistoryOpen && (
                        <div className="history-body">
                            {pinnedChats.length > 0 && (
                                <section className="conversation-group pinned-group">
                                    <h2 className="sidebar-label"><Pin size={13} /> Pinned</h2>
                                    {pinnedChats.map(renderChatRow)}
                                </section>
                            )}

                            {groups.length ? groups.map(([label, chats]) => (
                                <section className="conversation-group" key={label}>
                                    <h2 className="sidebar-label">{label}</h2>
                                    {chats.map(renderChatRow)}
                                </section>
                            )) : !pinnedChats.length && (
                                <p className="empty-history sidebar-label">{search ? 'No conversations found.' : 'Your conversations will appear here.'}</p>
                            )}
                        </div>
                    )}
                </div>

                {/* Bottom Footer User Area */}
                <div className="sidebar-bottom">
                    <div className="sidebar-footer-row">
                        {/* Clean Single User Profile Button + Dropdown Popover */}
                        <div className="profile-wrap" style={{ position: 'relative', width: '100%' }}>
                            <button
                                className="user-profile"
                                onClick={() => setProfileOpen((value) => !value)}
                                aria-label="User profile options"
                                style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '10px' }}
                            >
                                <span className="avatar"><User size={16} /></span>
                                <div className="sidebar-label profile-info" style={{ textAlign: 'left', flex: 1, overflow: 'hidden' }}>
                                    <span className="profile-name" style={{ display: 'block', fontWeight: 600, fontSize: '0.88rem', lineHeight: '1.2' }}>Personal AI</span>
                                    <span className="profile-subtitle" style={{ display: 'block', fontSize: '0.72rem', opacity: 0.7 }}>Account</span>
                                </div>
                                <ChevronDown className="sidebar-label" size={14} style={{ opacity: 0.7 }} />
                            </button>

                            {profileOpen && (
                                <div className="conversation-menu" style={{ bottom: '44px', top: 'auto', left: 0, minWidth: '170px' }}>
                                    <button onClick={() => { navigate('/settings'); setProfileOpen(false); }}>
                                        <Settings size={15} /> Settings
                                    </button>
                                    <button onClick={() => { onThemeChange(); setProfileOpen(false); }}>
                                        {isDarkMode ? <Sun size={15} /> : <Moon size={15} />}
                                        <span>{isDarkMode ? 'Light mode' : 'Dark mode'}</span>
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* ─── COLLAPSED RAIL CONTENT (Icon Rail) ─── */}
            <div className="sidebar-rail-content">
                <div className="rail-top">
                    <button className="rail-btn brand-rail-btn" onClick={() => navigate('/')} title="Personal AI home" aria-label="Personal AI home" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4px' }}>
                        <img src="/logo.png" alt="Personal AI" style={{ width: '22px', height: '22px', objectFit: 'contain' }} />
                    </button>
                    <button className="rail-btn" onClick={() => { onNewChat(); closeDrawerAfterSelection(); }} title="New chat" aria-label="New chat">
                        <Plus size={18} />
                    </button>
                    <button className="rail-btn" onClick={() => { onToggle(); setTimeout(() => searchInputRef.current?.focus(), 150); }} title="Search chats" aria-label="Search chats">
                        <Search size={18} />
                    </button>
                </div>

                <div className="rail-middle">
                    <button className={`rail-btn ${location.pathname === '/' ? 'active' : ''}`} onClick={() => navigate('/')} title="Chats" aria-label="Chats">
                        <MessageSquare size={18} />
                    </button>
                    <button className={`rail-btn ${location.pathname === '/settings' ? 'active' : ''}`} onClick={() => navigate('/settings')} title="Settings" aria-label="Settings">
                        <Settings size={18} />
                    </button>
                </div>

                <div className="rail-bottom">
                    {/* Minimal Standalone Expand Button » */}
                    <button
                        className="sidebar-expand-btn"
                        onClick={onToggle}
                        title="Expand sidebar (»)"
                        aria-label="Expand sidebar"
                    >
                        <ChevronsRight size={18} />
                    </button>

                    <button className="rail-avatar-btn" onClick={() => { onToggle(); setProfileOpen(true); }} title="User profile" aria-label="User profile">
                        <span className="avatar"><User size={15} /></span>
                    </button>
                </div>
            </div>

            {/* Modal Dialogs */}
            {renameChat && (
                <div className="modal-backdrop" role="presentation">
                    <form className="modal-card" onSubmit={async (event) => { event.preventDefault(); const title = renameValue.trim(); if (title) await onRenameChat(renameChat.id, title); setRenameChat(null); }}>
                        <h2>Rename conversation</h2>
                        <p>Give this chat a title that makes it easy to find later.</p>
                        <input autoFocus value={renameValue} onChange={(event) => setRenameValue(event.target.value)} maxLength={80} />
                        <div className="modal-actions"><button type="button" className="btn-secondary" onClick={() => setRenameChat(null)}>Cancel</button><button type="submit" className="btn-primary">Save title</button></div>
                    </form>
                </div>
            )}
            {deleteChat && (
                <div className="modal-backdrop" role="presentation">
                    <div className="modal-card">
                        <h2>Delete conversation?</h2>
                        <p>This will permanently delete this conversation and its messages. Documents will remain in your document library.</p>
                        <div className="modal-actions">
                            <button className="btn-secondary" disabled={isDeleting} onClick={() => setDeleteChat(null)}>Cancel</button>
                            <button className="btn-danger" disabled={isDeleting} onClick={confirmDelete}>{isDeleting ? 'Deleting…' : 'Delete'}</button>
                        </div>
                    </div>
                </div>
            )}
        </aside>
    );
};

export default Sidebar;
