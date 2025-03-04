// src/app/dashboard/chat/clinical/page.tsx
import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Clinical Case Analysis'
};

export default function ClinicalChatPage() {
  return <ChatContainer type="clinical" />;
}