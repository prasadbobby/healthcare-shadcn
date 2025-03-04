// src/features/chat/components/chat-container.tsx
'use client';

import { useEffect, useState } from 'react';
import ChatSidebar from './chat-sidebar';
import ChatInterface from './chat-interface';
import { useChatStore } from '../utils/store';

interface ChatContainerProps {
  type: 'clinical' | 'literature' | 'symptom' | 'drug';
  sessionId?: string;
}

export default function ChatContainer({ type, sessionId }: ChatContainerProps) {
  const [isStoreReady, setIsStoreReady] = useState(false);
  
  // Initialize store once
  useEffect(() => {
    useChatStore.persist.rehydrate();
    setIsStoreReady(true);
  }, []);
  
  if (!isStoreReady) {
    return <div className="flex h-full items-center justify-center">Loading...</div>;
  }
  
  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <ChatSidebar type={type} />
      <div className="flex-1">
        <ChatInterface type={type} sessionId={sessionId} />
      </div>
    </div>
  );
}