// Dynamic API base URL resolution to guarantee connectivity across localhost, 127.0.0.1, or custom ports
const getApiBaseUrl = () => {
    if (import.meta.env.VITE_API_BASE_URL) {
        return import.meta.env.VITE_API_BASE_URL;
    }
    if (typeof window !== 'undefined') {
        const hostname = window.location.hostname;
        if (hostname === '127.0.0.1') return 'http://127.0.0.1:8000';
        if (hostname === 'localhost') return 'http://localhost:8000';
        return `http://${hostname}:8000`;
    }
    return 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();

async function responseError(response, fallback) {
    const errorBody = await response.json().catch(() => ({}));
    const statusPrefix = response.status ? `[HTTP ${response.status}] ` : '';
    const detailMsg = errorBody.detail || errorBody.message || fallback;
    return new Error(`${statusPrefix}${detailMsg}`);
}

// ─── File APIs ───────────────────────────────────────────────────────────────
export const fetchFiles = async () => {
    const response = await fetch(`${API_BASE_URL}/files`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch files');
    return response.json();
};

export const fetchFileDetails = async (identifier) => {
    const response = await fetch(`${API_BASE_URL}/files/${encodeURIComponent(identifier)}/details`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch document details');
    return response.json();
};

export const fetchDocumentStatus = async (identifier) => {
    const response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(identifier)}/status`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch document status');
    return response.json();
};

export const fetchSystemStatus = async () => {
    const response = await fetch(`${API_BASE_URL}/status`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch system status');
    return response.json();
};

export const uploadFile = async (file, chatId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (chatId) formData.append('chat_id', chatId);
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        throw await responseError(response, 'Upload failed');
    }
    return response.json();
};

export const deleteFile = async (identifier) => {
    const response = await fetch(`${API_BASE_URL}/files/${encodeURIComponent(identifier)}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw await responseError(response, 'Failed to delete file');
    return response.json();
};

export const getFileUrl = (identifier) => `${API_BASE_URL}/files/${encodeURIComponent(identifier)}/download`;

export const reprocessFile = async (identifier) => {
    const response = await fetch(`${API_BASE_URL}/files/${encodeURIComponent(identifier)}/reprocess`, { method: 'POST' });
    if (!response.ok) {
        throw await responseError(response, 'Reprocessing failed');
    }
    return response.json();
};

// ─── Chat APIs ───────────────────────────────────────────────────────────────
export const streamChat = async (message, conversationId, image, documentNames, responseMode = 'balanced', onToken, onDone, signal, editTurnId = null) => {
    let docList = [];
    if (Array.isArray(documentNames)) {
        docList = documentNames.filter(Boolean);
    } else if (typeof documentNames === 'string' && documentNames) {
        docList = [documentNames];
    }

    const payload = {
        message,
        conversation_id: conversationId,
        image,
        document_ids: docList.length > 0 ? docList : null,
        document_id: docList.length === 1 ? docList[0] : null,
        document_names: docList.length > 0 ? docList : null,
        document_name: docList.length === 1 ? docList[0] : null,
        response_mode: responseMode || 'balanced',
        edit_turn_id: editTurnId || null,
    };

    let response;
    try {
        response = await fetch(`${API_BASE_URL}/chat_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal
        });
    } catch (networkError) {
        if (networkError.name === 'AbortError') throw networkError;
        console.error('Network connection error:', networkError);
        throw new Error(`Unable to reach backend server at ${API_BASE_URL}. Please verify the backend server is running (uvicorn main:app --reload).`);
    }

    if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        console.error('Chat stream request failed', { url: response.url, status: response.status, detail: errorBody.detail });
        const detailStr = String(errorBody.detail || 'Sorry, something went wrong while generating the response.');
        throw new Error(detailStr);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullAnswer = "";
    let buffer = "";
    let isStreamDone = false;

    const processLine = (line) => {
        if (!line.startsWith('data: ')) return;
        let data;
        try {
            const rawData = line.slice(6).trim();
            if (!rawData) return;
            data = JSON.parse(rawData);
        } catch (e) {
            console.error("Error parsing JSON chunk:", e, line);
            return;
        }
        if (data.error) {
            console.error("SSE Data Error:", data.error);
            throw new Error(data.error || 'Sorry, something went wrong while generating the response.');
        }
        if (data.token) {
            fullAnswer += data.token;
            onToken(data.token);
        }
        if (data.done) {
            isStreamDone = true;
            onDone(fullAnswer, data.sources || [], data.source_details || []);
        }
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop();

            for (const part of parts) {
                const lines = part.split('\n');
                for (const line of lines) {
                    processLine(line);
                    if (isStreamDone) return;
                }
            }
        }

        if (buffer) {
            const lines = buffer.split('\n');
            for (const line of lines) {
                processLine(line);
                if (isStreamDone) return;
            }
        }

        if (!isStreamDone) {
            onDone(fullAnswer, [], []);
        }
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error('Chat stream error:', e);
        }
        throw e;
    }
};

export const fetchChatMessages = async (conversationId) => {
    const response = await fetch(`${API_BASE_URL}/chat_messages?conversation_id=${encodeURIComponent(conversationId)}`);
    if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || 'Failed to fetch chat messages');
    }
    return response.json();
};

export const clearSession = async (conversationId) => {
    const response = await fetch(`${API_BASE_URL}/clear_session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId })
    });
    if (!response.ok) throw await responseError(response, 'Failed to clear session');
    return response.json();
};

// ─── History APIs ─────────────────────────────────────────────────────────────
export const fetchHistory = async (conversationId = null) => {
    const url = conversationId
        ? `${API_BASE_URL}/history?conversation_id=${encodeURIComponent(conversationId)}`
        : `${API_BASE_URL}/history`;
    const response = await fetch(url);
    if (!response.ok) throw await responseError(response, 'Failed to fetch history');
    return response.json();
};

export const clearHistory = async () => {
    const response = await fetch(`${API_BASE_URL}/clear_history`, {
        method: 'POST'
    });
    if (!response.ok) throw await responseError(response, 'Failed to clear history');
    return response.json();
};

export const deleteHistoryItem = async (id) => {
    const response = await fetch(`${API_BASE_URL}/history/${id}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw await responseError(response, 'Failed to delete history item');
    return response.json();
};

export const fetchChats = async (includeArchived = false, query = '') => {
    const params = new URLSearchParams();
    if (includeArchived) params.append('include_archived', 'true');
    if (query) params.append('query', query);
    const queryString = params.toString();
    const url = `${API_BASE_URL}/chats${queryString ? `?${queryString}` : ''}`;
    const response = await fetch(url);
    if (!response.ok) throw await responseError(response, 'Failed to fetch chats');
    return response.json();
};

export const createChat = async (title = 'New chat', selectedDocumentIds = []) => {
    const body = { title };
    if (Array.isArray(selectedDocumentIds) && selectedDocumentIds.length > 0) {
        body.selected_document_ids = selectedDocumentIds;
    }
    const response = await fetch(`${API_BASE_URL}/chats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        console.error('Create chat request failed', { url: response.url, status: response.status, detail: errorBody.detail });
        throw new Error(errorBody.detail || 'Failed to create chat');
    }
    return response.json();
};

export const deleteChat = async (chatId) => {
    const response = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(chatId)}`, {
        method: 'DELETE'
    });
    if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        console.error('Delete chat request failed', { url: response.url, status: response.status, detail: errorBody.detail });
        throw new Error(errorBody.detail || 'Failed to delete chat');
    }
    return response.json();
};

export const renameChat = async (chatId, title) => {
    return updateChat(chatId, { title });
};

export const updateChat = async (chatId, updates) => {
    const response = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(chatId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
    });
    if (!response.ok) throw await responseError(response, 'Failed to update chat');
    return response.json();
};

export const fetchChatFiles = async (chatId) => {
    const response = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(chatId)}/files`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch chat files');
    return response.json();
};

export const attachFileToChat = async (chatId, documentIdOrFilename) => {
    const isDocId = typeof documentIdOrFilename === 'string' && documentIdOrFilename.length === 36 && documentIdOrFilename.includes('-');
    const body = isDocId ? { document_id: documentIdOrFilename, filename: documentIdOrFilename } : { filename: documentIdOrFilename, document_id: documentIdOrFilename };
    const response = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(chatId)}/files/attach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) throw await responseError(response, 'Failed to attach file to chat');
    return response.json();
};

export const detachFileFromChat = async (chatId, documentIdOrFilename) => {
    const isDocId = typeof documentIdOrFilename === 'string' && documentIdOrFilename.length === 36 && documentIdOrFilename.includes('-');
    const body = isDocId ? { document_id: documentIdOrFilename, filename: documentIdOrFilename } : { filename: documentIdOrFilename, document_id: documentIdOrFilename };
    const response = await fetch(`${API_BASE_URL}/chats/${encodeURIComponent(chatId)}/files/detach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) throw await responseError(response, 'Failed to detach file from chat');
    return response.json();
};


export const cleanupOrphanedDocuments = async () => {
    const response = await fetch(`${API_BASE_URL}/system/cleanup_orphaned`, {
        method: 'POST'
    });
    if (!response.ok) throw await responseError(response, 'Failed to clean up orphaned documents');
    return response.json();
};

export const fetchSystemDiagnostics = async () => {
    const response = await fetch(`${API_BASE_URL}/system/diagnostics`);
    if (!response.ok) throw await responseError(response, 'Failed to fetch diagnostics');
    return response.json();
};



