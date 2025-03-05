// src/app/dashboard/image-analysis/page.tsx
import ImageAnalysisContainer from '@/features/image-analysis/components/image-analysis-container';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Healthcare Dashboard - Image Analysis'
};

export default function ImageAnalysisPage() {
  return <ImageAnalysisContainer type="image" />;
}