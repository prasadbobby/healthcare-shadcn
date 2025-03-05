// src/features/image-analysis/components/image-analysis-result.tsx
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ImageAnalysisResultProps {
  result: any;
  image: string | null;
}

export default function ImageAnalysisResult({ result, image }: ImageAnalysisResultProps) {
  if (!result) return null;
  
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          {image && (
            <img 
              src={image} 
              alt="Analyzed image" 
              className="w-full h-auto rounded-lg border"
            />
          )}
        </div>
        
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Analysis Results</h3>
            <div className="flex flex-wrap gap-2 mb-4">
              {result.detectedObjects?.map((item: string, index: number) => (
                <Badge key={index} variant="outline">{item}</Badge>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-md mb-1">Medical Assessment</h4>
            <p className="text-muted-foreground">{result.medicalAssessment}</p>
          </div>
          
          <div>
            <h4 className="font-medium text-md mb-1">Confidence Score</h4>
            <div className="flex items-center">
              <div className="w-full bg-secondary h-2 rounded-full">
                <div 
                  className="bg-primary h-2 rounded-full" 
                  style={{ width: `${result.confidenceScore * 100}%` }}
                ></div>
              </div>
              <span className="ml-2 text-sm">{Math.round(result.confidenceScore * 100)}%</span>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-md mb-1">Recommendations</h4>
            <ul className="list-disc pl-5 text-sm space-y-1 text-muted-foreground">
              {result.recommendations?.map((rec: string, index: number) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>
          
          <div className="flex items-center">
            <span className="font-medium text-md mr-2">Severity:</span>
            <Badge 
              variant={
                result.severity === 'High' ? 'destructive' : 
                result.severity === 'Moderate' ? 'default' : 'outline'
              }
            >
              {result.severity}
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}