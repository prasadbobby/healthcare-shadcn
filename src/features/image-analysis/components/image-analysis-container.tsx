// src/features/image-analysis/components/image-analysis-container.tsx
'use client';
import { useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { useChatStore, Message } from '../../../features/chat/utils/store';
import { useRouter } from 'next/navigation';
import { Camera, SendIcon, ImageIcon, XIcon } from 'lucide-react';
import { format } from 'date-fns';
import CameraCapture from './camera-capture';
import ImageAnalysisSidebar from './image-analysis-sidebar';

interface ImageAnalysisContainerProps {
    type: 'image';
    sessionId?: string;
}

export default function ImageAnalysisContainer({ type, sessionId }: ImageAnalysisContainerProps) {
    const {
        sessions,
        activeSession,
        activeSessionId,
        createSession,
        setActiveSessionId,
        addMessage
    } = useChatStore();

    const router = useRouter();
    const [message, setMessage] = useState('');
    const [showCamera, setShowCamera] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const initRef = useRef(false);

    // Initialize session
    useEffect(() => {
        if (initRef.current) return;

        const initializeSession = async () => {
            initRef.current = true;

            if (sessionId) {
                const sessionExists = sessions.some(s => s.id === sessionId);
                if (sessionExists) {
                    setActiveSessionId(sessionId);
                } else {
                    const newId = createSession(type);
                    router.replace(`/dashboard/image-analysis/${newId}`);
                }
            } else {
                const imageSessions = sessions.filter(s => s.type === type);
                if (imageSessions.length > 0) {
                    setActiveSessionId(imageSessions[0].id);
                    router.replace(`/dashboard/image-analysis/${imageSessions[0].id}`);
                } else {
                    const newId = createSession(type);
                    router.replace(`/dashboard/image-analysis/${newId}`);
                }
            }
        };

        initializeSession();
    }, [type, sessionId, sessions, setActiveSessionId, createSession, router]);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [activeSession?.messages]);

    // Initialize with welcome message if this is a new session
    useEffect(() => {
        if (activeSession && activeSession.messages.length === 0 && activeSessionId) {
          addMessage(activeSessionId, {
            content: 'Welcome to Image Analysis! Upload a medical image or take a photo to begin analysis.',
            role: 'assistant'
          });
        }
      }, [activeSession, activeSessionId, addMessage]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if ((!message.trim() && !uploadedImage) || isAnalyzing || !activeSessionId) return;

        // Add user message with image if there is one
        if (uploadedImage) {
            addMessage(activeSessionId, {
                content: uploadedFile ? `Uploaded image: ${uploadedFile.name}` : 'Captured image',
                role: 'user',
                metadata: { imageUrl: uploadedImage }
            });

            // Clear the uploaded image
            setUploadedImage(null);
            setUploadedFile(null);

            // Simulate analysis
            setIsAnalyzing(true);
            await simulateAnalysisResponse();
            setIsAnalyzing(false);
        } else if (message.trim()) {
            // Just text message
            addMessage(activeSessionId, {
                content: message,
                role: 'user'
            });

            // Clear input
            setMessage('');

            // Simulate response
            await simulateTextResponse(message);
        }
    };

    const handleCameraCapture = (imageSrc: string) => {
        setUploadedImage(imageSrc);
        setUploadedFile(null);
        setShowCamera(false);
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            const imageUrl = URL.createObjectURL(file);
            setUploadedImage(imageUrl);
            setUploadedFile(file);
        }
    };

    const simulateAnalysisResponse = async () => {
        // Simulate delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        addMessage(activeSessionId, {
            content: `I've analyzed this medical image. This appears to be a T2-weighted MRI scan of the brain with some abnormal tissue contrast in the temporal lobe region. The confidence score for this assessment is 89%. The apparent abnormality is located in the temporal lobe region showing altered signal intensity which could indicate several possibilities including a focal lesion, inflammation, or tissue changes. Would you like me to explain any specific aspects of this image?`,
            role: 'assistant'
        });
    };

    const simulateTextResponse = async (userMessage: string) => {
        // Simulate delay
        await new Promise(resolve => setTimeout(resolve, 1000));

        const lowerQuestion = userMessage.toLowerCase();
        let response = '';

        if (lowerQuestion.includes('abnormality') || lowerQuestion.includes('issue')) {
            response = 'The apparent abnormality in this image is located in the temporal lobe region. It shows altered signal intensity which could indicate several possibilities including a focal lesion, inflammation, or tissue changes. More specific diagnosis would require clinical correlation and potentially additional imaging sequences.';
        } else if (lowerQuestion.includes('diagnosis') || lowerQuestion.includes('condition')) {
            response = 'Based solely on this single image, I cannot provide a definitive diagnosis. The findings could be consistent with several conditions including low-grade glioma, focal cortical dysplasia, or post-inflammatory changes. Clinical context, patient history, and additional imaging are crucial for accurate diagnosis.';
        } else if (lowerQuestion.includes('recommend') || lowerQuestion.includes('next steps')) {
            response = 'I recommend: 1) Clinical correlation with patient symptoms, 2) Additional MRI sequences including contrast-enhanced T1, FLAIR, and potentially diffusion-weighted imaging, 3) Follow-up imaging to assess for any changes over time, 4) Depending on clinical suspicion, consideration for advanced imaging such as MR spectroscopy.';
        } else {
            response = "To properly address your question about this image, I would need more specific details about what aspect you're interested in. I can provide information about the apparent findings, potential implications, or suggested next steps for clinical assessment.";
        }

        addMessage(activeSessionId, {
            content: response,
            role: 'assistant'
        });
    };

    return (
        <div className="flex h-[calc(100vh-4rem)]">
            {/* Left Sidebar - Sessions */}
            <ImageAnalysisSidebar type={type} />

            {/* Main Chat Area */}
            <div className="flex-1">
                <Card className="flex h-full flex-col overflow-hidden">
                    <CardHeader className="border-b">
                        <div className="flex items-center justify-between">
                            <h3 className="font-medium">Image Analysis</h3>
                        </div>
                    </CardHeader>

                    <CardContent className="flex-1 overflow-auto p-0">
                        <ScrollArea className="h-full">
                            <div className="flex flex-col space-y-4 p-4">
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
          <AvatarFallback>IA</AvatarFallback>
        </Avatar>
      )}
      <div className="space-y-2">
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

                                {/* Camera interface when active */}
                                {showCamera && (
                                    <div className="fixed inset-x-0 bottom-16 z-50 bg-background p-4 shadow-lg border-t">
                                        <div className="relative">
                                            <Button
                                                className="absolute right-0 top-0 z-10"
                                                size="icon"
                                                variant="ghost"
                                                onClick={() => setShowCamera(false)}
                                            >
                                                <XIcon className="h-4 w-4" />
                                            </Button>
                                            <CameraCapture onCapture={handleCameraCapture} />
                                        </div>
                                    </div>
                                )}
                            </div>
                        </ScrollArea>
                    </CardContent>

                    <CardFooter className="border-t p-4">
                        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
                            {uploadedImage && (
                                <div className="relative mr-2">
                                    <div className="h-10 w-10 overflow-hidden rounded-md border">
                                        <img
                                            src={uploadedImage}
                                            alt="Preview"
                                            className="h-full w-full object-cover"
                                        />
                                    </div>
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        className="absolute -right-2 -top-2 h-5 w-5 rounded-full bg-red-500 p-0 text-white hover:bg-red-600"
                                        onClick={() => {
                                            setUploadedImage(null);
                                            setUploadedFile(null);
                                        }}
                                    >
                                        <XIcon className="h-3 w-3" />
                                    </Button>
                                </div>
                            )}

                            <div className="flex items-center space-x-2">
                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => setShowCamera(!showCamera)}
                                >
                                    <Camera className="h-5 w-5" />
                                </Button>

                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <ImageIcon className="h-5 w-5" />
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        className="hidden"
                                        accept="image/*"
                                        onChange={handleFileUpload}
                                    />
                                </Button>

                                {/* <MicButton
                                    onSpeechRecognized={(text) => setMessage(text)}
                                    disabled={isAnalyzing || !activeSessionId}
                                /> */}
                            </div>

                            <Textarea
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                placeholder={uploadedImage
                                    ? "Add a message or click Send to analyze this image..."
                                    : "Type a message or upload an image..."}
                                className="flex-1 min-h-10 resize-none"
                            />

                            <Button
                                type="submit"
                                disabled={isAnalyzing || (!message.trim() && !uploadedImage) || !activeSessionId}
                                size="icon"
                            >
                                <SendIcon className="h-4 w-4" />
                                <span className="sr-only">Send</span>
                            </Button>
                        </form>
                    </CardFooter>
                </Card>
            </div>
        </div>
    );
}