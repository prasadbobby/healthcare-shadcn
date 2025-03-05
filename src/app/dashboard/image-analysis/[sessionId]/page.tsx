// src/app/dashboard/image-analysis/[sessionId]/page.tsx
import ImageAnalysisContainer from '@/features/image-analysis/components/image-analysis-container';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Healthcare Dashboard - Image Analysis'
};

export default async function ImageAnalysisSessionPage({
  params
}: {
  params: Promise<{ sessionId: string }> | { sessionId: string };
}) {
  // Await the params if it's a Promise
  const resolvedParams = await Promise.resolve(params);
  
  return <ImageAnalysisContainer type="image" sessionId={resolvedParams.sessionId} />;
}