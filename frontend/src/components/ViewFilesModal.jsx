import FilesInChatDrawer from './FilesInChatDrawer';

const ViewFilesModal = ({ chat, onClose, onFilesChange }) => {
    const chatId = chat?.id || chat?.conversationId;
    return (
        <FilesInChatDrawer
            isOpen={Boolean(chat)}
            onClose={onClose}
            chatId={chatId}
            onFilesChange={onFilesChange}
        />
    );
};

export default ViewFilesModal;
