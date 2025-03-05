// src/features/image-analysis/components/camera-capture.tsx
'use client';
import { useRef, useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, RefreshCw } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (imageSrc: string) => void;
}

export default function CameraCapture({ onCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Start camera automatically when component mounts
  useEffect(() => {
    startCamera();
    
    // Cleanup when component unmounts
    return () => {
      stopCamera();
    };
  }, []);
  
  const startCamera = async () => {
    try {
      // First stop any existing streams
      stopCamera();
      
      // Let's ensure the video element is fully loaded before attempting to use it
      if (!videoRef.current) {
        setError("Camera element not ready. Please try again.");
        return;
      }
      
      // Try various camera configurations
      let stream: MediaStream | null = null;
      
      try {
        // First try user camera (front camera) as it's more likely to work on most devices
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' }
        });
      } catch (e) {
        // Then try environment camera (back camera on phones)
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
          });
        } catch (e) {
          // Finally try any camera
          stream = await navigator.mediaDevices.getUserMedia({
            video: true
          });
        }
      }
      
      if (videoRef.current && stream) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        
        // Important for iOS
        videoRef.current.setAttribute('playsinline', 'true');
        videoRef.current.setAttribute('webkit-playsinline', 'true');
        
        // Use the onloadedmetadata event to ensure the video is ready
        videoRef.current.onloadedmetadata = async () => {
          if (videoRef.current) {
            try {
              await videoRef.current.play();
              setCameraActive(true);
              setError(null);
            } catch (e: any) {
              setError(`Error starting camera: ${e.message}`);
            }
          }
        };
      }
    } catch (err: any) {
      console.error('Error accessing camera:', err);
      setError(`Unable to access camera: ${err.message}. Please check permissions.`);
    }
  };
  
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };
  
  const captureImage = () => {
    if (!canvasRef.current || !videoRef.current) return;
    
    setIsCapturing(true);
    
    try {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        // Get video dimensions
        const videoWidth = videoRef.current.videoWidth || 640;
        const videoHeight = videoRef.current.videoHeight || 480;
        
        // Set canvas dimensions
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        
        // Draw video to canvas
        context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
        
        // Convert to data URL
        const imageSrc = canvasRef.current.toDataURL('image/jpeg', 0.9);
        onCapture(imageSrc);
      }
    } catch (error: any) {
      console.error('Error capturing image:', error);
      setError(`Failed to capture image: ${error.message}`);
    } finally {
      setIsCapturing(false);
    }
  };
  
  return (
    <div className="w-full space-y-4">
      <div className="aspect-video max-h-[300px] bg-black relative rounded-lg overflow-hidden">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black text-white p-4 text-center">
            <div>
              <p>{error}</p>
              <Button 
                onClick={startCamera} 
                variant="secondary" 
                className="mt-4"
              >
                Try Again
              </Button>
            </div>
          </div>
        ) : (
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted
            className="w-full h-full object-contain"
          />
        )}
        
        {!cameraActive && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-white">
            <p>Starting camera...</p>
          </div>
        )}
      </div>
      
      <canvas ref={canvasRef} className="hidden" />
      
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={startCamera}
          disabled={isCapturing}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Restart
        </Button>
        
        <Button
          onClick={captureImage}
          disabled={!cameraActive || isCapturing}
          className="w-1/2"
        >
          <Camera className="mr-2 h-4 w-4" />
          {isCapturing ? 'Capturing...' : 'Capture Photo'}
        </Button>
      </div>
    </div>
  );
}