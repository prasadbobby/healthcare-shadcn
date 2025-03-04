// src/app/dashboard/chat/literature/[sessionId]/page.tsx
import ChatContainer from '@/features/chat/components/chat-container';

export const metadata = {
  title: 'Healthcare Dashboard - Medical Literature Review'
};

export default async function LiteratureChatSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }> | { sessionId: string };
}) {
  // Await the params if it's a Promise
  const resolvedParams = await Promise.resolve(params);
  
  return <ChatContainer type="literature" sessionId={resolvedParams.sessionId} />;
}