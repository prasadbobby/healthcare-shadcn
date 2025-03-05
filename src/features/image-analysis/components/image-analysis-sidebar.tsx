// src/features/image-analysis/components/image-analysis-sidebar.tsx
'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatStore, ChatSession } from '../../../features/chat/utils/store';
import { ImageIcon, Plus, Trash2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';

interface ImageAnalysisSidebarProps {
  type: 'image';
}

export default function ImageAnalysisSidebar({ type }: ImageAnalysisSidebarProps) {
  const { 
    sessions, 
    activeSessionId, 
    createSession, 
    setActiveSessionId, 
    deleteSession
  } = useChatStore();
  
  const router = useRouter();
  const [typeSessions, setTypeSessions] = useState<ChatSession[]>([]);
  
  // Update type sessions when sessions change
  useEffect(() => {
    setTypeSessions(sessions.filter(s => s.type === type));
  }, [sessions, type]);
  
  // Create a new session
  const handleNewSession = () => {
    const newSessionId = createSession(type);
    router.push(`/dashboard/image-analysis/${newSessionId}`);
  };
  
  // Format session title
  const formatSessionTitle = (session: ChatSession) => {
    // Default title if empty or "New image session"
    if (!session.title || session.title === `New ${type} session`) {
      return `Analysis ${new Date(session.createdAt).toLocaleDateString()}`;
    }
    
    if (session.title.length > 20) {
      return session.title.substring(0, 20) + '...';
    }
    return session.title;
  };

  return (
    <div className="flex h-full w-64 flex-col border-r">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center">
          <ImageIcon className="h-4 w-4" />
          <span className="ml-2 font-medium">
            Image Sessions
          </span>
        </div>
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleNewSession}
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {typeSessions.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No sessions yet
            </div>
          ) : (
            typeSessions.map((session) => (
              <div 
                key={session.id}
                className="flex items-center justify-between group"
              >
                <Button
                  variant="ghost"
                  className={cn(
                    "w-full justify-start text-left font-normal",
                    session.id === activeSessionId && "bg-muted"
                  )}
                  onClick={() => {
                    setActiveSessionId(session.id);
                    router.push(`/dashboard/image-analysis/${session.id}`);
                  }}
                >
                  <div className="truncate">
                    {formatSessionTitle(session)}
                  </div>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 opacity-0 group-hover:opacity-100"
                  onClick={() => {
                    deleteSession(session.id);
                    // Navigate back to type page if active session is deleted
                    if (session.id === activeSessionId) {
                      router.push(`/dashboard/image-analysis`);
                    }
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
      
      <div className="border-t p-2">
        <div className="text-xs text-muted-foreground px-2 py-1">
          {typeSessions.length} session{typeSessions.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
  );
}