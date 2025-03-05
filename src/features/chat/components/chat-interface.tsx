// src/features/chat/components/chat-interface.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatStore, Message } from '../utils/store';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { format } from 'date-fns';
import { XCircle, Send, Trash2, Plus, ArrowUp } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useRouter } from 'next/navigation';

interface ChatInterfaceProps {
  type: 'clinical' | 'literature' | 'symptom' | 'drug';
  sessionId?: string;
}

export default function ChatInterface({ type, sessionId }: ChatInterfaceProps) {
  const {
    sessions,
    activeSession,
    activeSessionId,
    createSession,
    setActiveSessionId,
    addMessage,
    clearSession,
    deleteSession
  } = useChatStore();
  
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const initRef = useRef(false);
  
  // Improved initialization logic with ref to prevent loops
  useEffect(() => {
    // Skip if already initialized
    if (initRef.current) return;
    
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
        } else if (type === 'clinical') {
          // For clinical, find existing clinical sessions first
          const clinicalSessions = sessions.filter(s => s.type === 'clinical');
          if (clinicalSessions.length > 0) {
            // Use first existing clinical session
            setActiveSessionId(clinicalSessions[0].id);
            router.replace(`/dashboard/chat/clinical/${clinicalSessions[0].id}`);
          } else {
            // Only create new session if none exist
            const newId = createSession('clinical');
            router.replace(`/dashboard/chat/clinical/${newId}`);
          }
        } else {
          // For other types, create new session
          const newId = createSession(type);
          router.replace(`/dashboard/chat/${type}/${newId}`);
        }
      } else {
        // No sessionId provided, handle by type
        if (type === 'clinical') {
          // For clinical, find existing clinical sessions first
          const clinicalSessions = sessions.filter(s => s.type === 'clinical');
          if (clinicalSessions.length > 0) {
            // Use first existing clinical session
            setActiveSessionId(clinicalSessions[0].id);
            router.replace(`/dashboard/chat/clinical/${clinicalSessions[0].id}`);
          } else {
            // Only create new if none exist
            const newId = createSession('clinical');
            router.replace(`/dashboard/chat/clinical/${newId}`);
          }
        } else {
          // For other types, check for existing sessions
          const typeSessions = sessions.filter(s => s.type === type);
          if (typeSessions.length > 0) {
            setActiveSessionId(typeSessions[0].id);
            router.replace(`/dashboard/chat/${type}/${typeSessions[0].id}`);
          } else {
            const newId = createSession(type);
            router.replace(`/dashboard/chat/${type}/${newId}`);
          }
        }
      }
    };
    
    initializeSession();
  }, [type, sessionId, sessions, setActiveSessionId, createSession, router]);
  

  useEffect(() => {
    if (activeSession && activeSession.messages.length === 0 && activeSessionId) {
      // Add appropriate welcome message based on type
      let welcomeMessage = '';
      switch (type) {
        case 'clinical':
          welcomeMessage = 'Welcome to Clinical Case Analysis! Describe a clinical case or patient scenario for analysis.';
          break;
        case 'literature':
          welcomeMessage = 'Welcome to Medical Literature Review! Ask about recent medical research or specific conditions.';
          break;
        case 'symptom':
          welcomeMessage = 'Welcome to Symptom Analysis! Describe symptoms for potential causes and recommendations.';
          break;
        case 'drug':
          welcomeMessage = 'Welcome to Drug Interaction Analysis! Enter medications to check for potential interactions.';
          break;
        default:
          welcomeMessage = 'Welcome! How can I assist you today?';
      }
      
      addMessage(activeSessionId, {
        content: welcomeMessage,
        role: 'assistant'
      });
    }
  }, [activeSession, activeSessionId, addMessage, type]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim() || isLoading || !activeSessionId) return;
    
    // Add user message
    addMessage(activeSessionId, {
      content: message,
      role: 'user'
    });
    
    // Clear input
    setMessage('');
    
    // Focus back on textarea
    textareaRef.current?.focus();
    
    // Simulate API request
    setIsLoading(true);
    
    try {
      // Simulate a delay for the API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Add assistant response
      addMessage(activeSessionId, {
        content: `This is a simulated response for your ${type} analysis request.`,
        role: 'assistant'
      });
    } catch (error) {
      console.error('Error getting AI response:', error);
      // Add error message
      addMessage(activeSessionId, {
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant'
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  const getChatTitle = () => {
    switch (type) {
      case 'clinical':
        return 'Clinical Case Analysis';
      case 'literature':
        return 'Medical Literature Review';
      case 'symptom':
        return 'Symptom Analysis';
      case 'drug':
        return 'Drug Interaction';
      default:
        return 'AI Chat';
    }
  };
  
  const getChatIcon = () => {
    switch (type) {
      case 'clinical':
        return 'CA';
      case 'literature':
        return 'LR';
      case 'symptom':
        return 'SA';
      case 'drug':
        return 'DI';
      default:
        return 'AI';
    }
  };
  
  // Function to create a new session
  const handleNewSession = () => {
    // Prevent creation of multiple clinical sessions
    if (type === 'clinical') {
      const clinicalSessions = sessions.filter(s => s.type === 'clinical');
      if (clinicalSessions.length > 0) {
        // Just switch to the first one if it exists
        setActiveSessionId(clinicalSessions[0].id);
        router.push(`/dashboard/chat/clinical/${clinicalSessions[0].id}`);
        return;
      }
    }
    
    const newSessionId = createSession(type);
    router.push(`/dashboard/chat/${type}/${newSessionId}`);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      <Card className="flex flex-1 flex-col overflow-hidden">
        <CardHeader className="border-b">
          <div className="flex justify-between items-center">
            <CardTitle>{getChatTitle()}</CardTitle>
            <div className="flex space-x-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="icon"
                      onClick={handleNewSession}
                      disabled={type === 'clinical' && sessions.filter(s => s.type === 'clinical').length > 0}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>New Session</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="icon" 
                      onClick={() => activeSessionId && clearSession(activeSessionId)}
                      disabled={!activeSessionId || !activeSession?.messages.length}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Clear Chat</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </CardHeader>
        
        <ScrollArea className="flex-1 p-4">
  <div className="space-y-4">
    {activeSession?.messages.map((msg: Message) => (
      <div 
        key={msg.id} 
        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
      >
        <div 
          className={`flex max-w-[80%] items-start space-x-2 rounded-lg p-4
            ${msg.role === 'user' 
              ? 'bg-primary text-primary-foreground' 
              : 'bg-muted'}`}
        >
          {msg.role === 'assistant' && (
            <Avatar className="h-8 w-8">
              <AvatarFallback>{getChatIcon()}</AvatarFallback>
            </Avatar>
          )}
          <div className="space-y-1">
            {/* Handle both direct imageUrl and stored imageKey */}
            {(msg.metadata?.imageUrl || msg.metadata?.imageKey) && (
              <div className="mb-2 overflow-hidden rounded-md border">
                <img 
                  src={msg.metadata?.imageUrl || (msg.metadata?.imageKey ? sessionStorage.getItem(msg.metadata.imageKey) : '')} 
                  alt="Uploaded image" 
                  className="h-auto max-h-60 w-full object-contain" 
                />
              </div>
            )}
            <div className="break-words">{msg.content}</div>
            <div className="text-xs opacity-50">
              {format(new Date(msg.createdAt), 'h:mm a')}
            </div>
          </div>
        </div>
      </div>
    ))}
    <div ref={messagesEndRef} />
  </div>
</ScrollArea>
        
        <CardFooter className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
  <Textarea
    ref={textareaRef}
    value={message}
    onChange={(e) => setMessage(e.target.value)}
    onKeyDown={handleKeyDown}
    placeholder={`Enter your ${type} query...`}
    className="flex-1 min-h-10 resize-none"
    disabled={isLoading || !activeSessionId}
  />
  
  {/* <MicButton 
    onSpeechRecognized={(text) => setMessage(text)}
    disabled={isLoading || !activeSessionId}
  /> */}
  
  <Button 
    type="submit" 
    disabled={!message.trim() || isLoading || !activeSessionId}
    className="shrink-0"
  >
    {isLoading ? (
      <div className="animate-spin">⟳</div>
    ) : (
      <ArrowUp className="h-4 w-4" />
    )}
    <span className="sr-only">Send</span>
  </Button>
</form>
        </CardFooter>
      </Card>
    </div>
  );
}