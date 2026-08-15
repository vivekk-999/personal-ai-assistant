import { useEffect, useState } from 'react';
import { Search, Calendar, Trash2, ChevronLeft, ChevronRight, Clock } from 'lucide-react';
import { clearHistory, deleteHistoryItem, fetchHistory } from '../api';
const HistoryView = ({ history, onClearHistory }) => {
    const [items, setItems] = useState(history || []);
    const [searchTerm, setSearchTerm] = useState('');
    const [dateFilter, setDateFilter] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const ITEMS_PER_PAGE = 5;

    useEffect(() => {
        const loadHistory = async () => {
            try {
                const data = await fetchHistory();
                setItems(data.history || []);
            } catch (error) {
                console.error("Error loading history:", error);
            }
        };
        loadHistory();
    }, []);

    const filteredHistory = items.filter(item => {
        const question = item.question || '';
        const preview = item.preview || item.answer || '';
        const matchesSearch = question.toLowerCase().includes(searchTerm.toLowerCase()) || 
                             preview.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesDate = !dateFilter || (item.time || '').includes(dateFilter);
        return matchesSearch && matchesDate;
    });

    const totalPages = Math.ceil(filteredHistory.length / ITEMS_PER_PAGE);
    const paginatedHistory = filteredHistory.slice(
        (currentPage - 1) * ITEMS_PER_PAGE,
        currentPage * ITEMS_PER_PAGE
    );

    const handleDelete = async (id) => {
        if (window.confirm("Are you sure you want to delete this conversation?")) {
            try {
                await deleteHistoryItem(id);
                const data = await fetchHistory();
                setItems(data.history || []);
                if (onClearHistory) onClearHistory();
            } catch (e) {
                console.error(e);
            }
        }
    };

    return (
        <div className="view-container">
            <header className="view-header">
                <div className="header-info">
                    <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>History</h2>
                    <p style={{ color: 'var(--text-muted)' }}>View your past conversations</p>
                </div>
                <button 
                    className="new-chat-btn" 
                    style={{ border: '1px solid var(--border)', color: 'var(--text-main)' }}
                    onClick={async () => {
                        try {
                            await clearHistory();
                            setItems([]);
                            if (onClearHistory) onClearHistory();
                        } catch (e) {
                            console.error(e);
                        }
                    }}
                >
                    <Trash2 size={18} style={{ marginRight: '8px' }} />
                    Clear History
                </button>
            </header>

            <div className="filter-bar">
                <div className="search-input-wrapper">
                    <Search size={18} color="var(--text-muted)" />
                    <input 
                        type="text" 
                        placeholder="Search conversations..." 
                        value={searchTerm}
                        onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                    />
                </div>
                <div className="search-dropdown" style={{ minWidth: '180px', padding: '0 8px', position: 'relative' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', width: '100%' }}>
                        <Calendar size={18} color="var(--text-muted)" />
                        <input 
                            type="date" 
                            className="native-date-input"
                            style={{ border: 'none', background: 'transparent', outline: 'none', color: 'var(--text-main)', fontSize: '0.875rem', width: '100%' }}
                            value={dateFilter}
                            onChange={(e) => { setDateFilter(e.target.value); setCurrentPage(1); }}
                        />
                    </div>
                </div>
            </div>

            <div className="table-card">
                <table>
                    <thead>
                        <tr>
                            <th style={{ width: '50px' }}>#</th>
                            <th>Question</th>
                            <th>Response Preview</th>
                            <th>Time</th>
                            <th style={{ textAlign: 'center' }}>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedHistory.map((item, index) => (
                            <tr key={item.id}>
                                <td style={{ fontWeight: 600 }}>{(currentPage - 1) * ITEMS_PER_PAGE + index + 1}</td>
                                <td>
                                    <div style={{ fontWeight: 600, maxWidth: '250px' }}>{item.question || 'Untitled conversation'}</div>
                                </td>
                                <td>
                                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem', maxWidth: '400px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {item.preview || item.answer || 'No preview available'}
                                    </div>
                                </td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                        <Clock size={14} />
                                        {item.time || 'Unknown time'}
                                    </div>
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                    <Trash2 
                                        size={18} 
                                        color="var(--text-muted)" 
                                        style={{ cursor: 'pointer' }} 
                                        onClick={() => handleDelete(item.id)}
                                    />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <div className="pagination">
                    <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                        Showing {paginatedHistory.length > 0 ? (currentPage - 1) * ITEMS_PER_PAGE + 1 : 0} to {(currentPage - 1) * ITEMS_PER_PAGE + paginatedHistory.length} of {filteredHistory.length} conversations
                    </span>
                    <div className="page-controls">
                        <button 
                            className="page-btn" 
                            disabled={currentPage === 1}
                            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                        >
                            <ChevronLeft size={16} />
                        </button>
                        
                        {[...Array(totalPages)].map((_, i) => (
                            <button 
                                key={i + 1}
                                className={`page-btn ${currentPage === i + 1 ? 'active' : ''}`}
                                onClick={() => setCurrentPage(i + 1)}
                            >
                                {i + 1}
                            </button>
                        ))}
                        
                        <button 
                            className="page-btn" 
                            disabled={currentPage === totalPages || totalPages === 0}
                            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                        >
                            <ChevronRight size={16} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HistoryView;
