// src/app/dashboard/chat/clinical/[sessionId]/page.tsx
import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Clinical Case Analysis'
};

export default async function ClinicalChatSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }> | { sessionId: string };
}) {
  // Await the params if it's a Promise
  const resolvedParams = await Promise.resolve(params);
  
  return <ChatContainer type="clinical" sessionId={resolvedParams.sessionId} />;
}