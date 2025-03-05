// src/features/chat/components/chat-container.tsx
'use client';

import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import ChatSidebar from './chat-sidebar';
import ChatInterface from './chat-interface';
import { useChatStore } from '../utils/store';

interface ChatContainerProps {
  type: 'clinical' | 'literature' | 'symptom' | 'drug';
  sessionId?: string;
}

export default function ChatContainer({ type, sessionId }: ChatContainerProps) {
  const [isStoreReady, setIsStoreReady] = useState(false);
  const {
    sessions,
    activeSessionId,
    createSession,
    setActiveSessionId,
  } = useChatStore();
  const router = useRouter();
  const initRef = useRef(false);
  
  // Initialize store once
  useEffect(() => {
    useChatStore.persist.rehydrate();
    setIsStoreReady(true);
  }, []);
  
  // Improved initialization logic with ref to prevent loops
  useEffect(() => {
    // Skip if already initialized
    if (initRef.current || !isStoreReady) return;
    
    const initializeSession = async () => {
      // Set flag to prevent repeated initialization
      initRef.current = true;
      
      // If sessionId is provided
      if (sessionId) {
        // Check if this session exists
        const sessionExists = sessions.some(s => s.id === sessionId);
        if (sessionExists) {
          // Set as active if it exists
          setActiveSessionId(sessionId);
        } else {
          // Create a new session with this ID
          const newId = createSession(type);
          router.replace(`/dashboard/chat/${type}/${newId}`);
        }
      } else {
        // No sessionId provided, check for existing sessions of this type
        const typeSessions = sessions.filter(s => s.type === type);
        if (typeSessions.length > 0) {
          // Use first existing session
          setActiveSessionId(typeSessions[0].id);
          router.replace(`/dashboard/chat/${type}/${typeSessions[0].id}`);
        } else {
          // Create new session
          const newId = createSession(type);
          router.replace(`/dashboard/chat/${type}/${newId}`);
        }
      }
    };
    
    initializeSession();
  }, [type, sessionId, sessions, setActiveSessionId, createSession, router, isStoreReady]);
  
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