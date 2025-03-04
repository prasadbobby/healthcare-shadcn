// src/features/chat/utils/store.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Message = {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  createdAt: Date;
};

export type ChatSession = {
  id: string;
  title: string;
  messages: Message[];
  type: 'clinical' | 'literature' | 'symptom' | 'drug';
  createdAt: Date;
  updatedAt: Date;
};

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  activeSession: ChatSession | null;
  createSession: (type: 'clinical' | 'literature' | 'symptom' | 'drug') => string;
  setActiveSessionId: (id: string) => void;
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'createdAt'>) => void;
  clearSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  getSessionsByType: (type: 'clinical' | 'literature' | 'symptom' | 'drug') => ChatSession[];
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      activeSession: null,
      
      createSession: (type) => {
        // Create a new session without restrictions for any type
        const id = Date.now().toString();
        const newSession: ChatSession = {
          id,
          title: `New ${type} session`,
          messages: [],
          type,
          createdAt: new Date(),
          updatedAt: new Date()
        };
        
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: id,
          activeSession: newSession
        }));
        
        return id;
      },
      
      setActiveSessionId: (id) => {
        // Don't update if already the active session
        if (get().activeSessionId === id) return;
        
        const session = get().sessions.find(s => s.id === id) || null;
        set({ activeSessionId: id, activeSession: session });
      },
      
      addMessage: (sessionId, messageData) => {
        const newMessage: Message = {
          id: Date.now().toString(),
          content: messageData.content,
          role: messageData.role,
          createdAt: new Date()
        };
        
        set((state) => {
          const updatedSessions = state.sessions.map(session => {
            if (session.id === sessionId) {
              return {
                ...session,
                messages: [...session.messages, newMessage],
                updatedAt: new Date(),
                // Update title based on first user message if it's the only message so far
                title: session.messages.length === 0 && messageData.role === 'user' 
                  ? messageData.content.slice(0, 30) + (messageData.content.length > 30 ? '...' : '')
                  : session.title
              };
            }
            return session;
          });
          
          const activeSession = updatedSessions.find(s => s.id === sessionId) || null;
          
          return {
            sessions: updatedSessions,
            activeSession
          };
        });
      },
      
      clearSession: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.map(session => 
            session.id === sessionId
              ? { ...session, messages: [], updatedAt: new Date() }
              : session
          )
        }));
      },
      
      deleteSession: (sessionId) => {
        set((state) => {
          const updatedSessions = state.sessions.filter(session => session.id !== sessionId);
          
          // If the active session is deleted, set a new active session
          let newActiveSessionId = state.activeSessionId;
          let newActiveSession = state.activeSession;
          
          if (state.activeSessionId === sessionId) {
            newActiveSessionId = updatedSessions.length > 0 ? updatedSessions[0].id : null;
            newActiveSession = newActiveSessionId 
              ? updatedSessions.find(s => s.id === newActiveSessionId) || null
              : null;
          }
          
          return {
            sessions: updatedSessions,
            activeSessionId: newActiveSessionId,
            activeSession: newActiveSession
          };
        });
      },
      
      getSessionsByType: (type) => {
        return get().sessions.filter(session => session.type === type);
      }
    }),
    {
      name: 'healthcare-chat-store',
      skipHydration: true
    }
  )
);