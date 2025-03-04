import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Symptom Analysis'
};

export default function SymptomChatPage() {
  return <ChatContainer type="symptom" />;
}