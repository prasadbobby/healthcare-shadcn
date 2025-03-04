// src/app/dashboard/chat/symptom/[sessionId]/page.tsx
import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Symptom Analysis'
};

export default async function SymptomChatSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }> | { sessionId: string };
}) {
  // Await the params if it's a Promise
  const resolvedParams = await Promise.resolve(params);
  
  return <ChatContainer type="symptom" sessionId={resolvedParams.sessionId} />;
}