// src/app/dashboard/chat/drug/[sessionId]/page.tsx
import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Drug Interaction'
};

export default async function DrugChatSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }> | { sessionId: string };
}) {
  // Await the params if it's a Promise
  const resolvedParams = await Promise.resolve(params);
  
  return <ChatContainer type="drug" sessionId={resolvedParams.sessionId} />;
}