import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Drug Interaction'
};

export default function DrugChatPage() {
  return <ChatContainer type="drug" />;
}