// src/app/api/image-analysis/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { question, imageId } = await req.json();
    
    if (!question) {
      return NextResponse.json(
        { error: 'No question provided' },
        { status: 400 }
      );
    }
    
    // In production, you would forward this to your Python backend
    // along with the image ID for context
    
    // Simulate a delay for demo purposes
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Generate a response based on the question
    let response;
    const lowerQuestion = question.toLowerCase();
    
    if (lowerQuestion.includes('abnormality') || lowerQuestion.includes('issue')) {
      response = 'The apparent abnormality in this image is located in the temporal lobe region. It shows altered signal intensity which could indicate several possibilities including a focal lesion, inflammation, or tissue changes. More specific diagnosis would require clinical correlation and potentially additional imaging sequences.';
    } else if (lowerQuestion.includes('diagnosis') || lowerQuestion.includes('condition')) {
      response = 'Based solely on this single image, I cannot provide a definitive diagnosis. The findings could be consistent with several conditions including low-grade glioma, focal cortical dysplasia, or post-inflammatory changes. Clinical context, patient history, and additional imaging are crucial for accurate diagnosis.';
    } else if (lowerQuestion.includes('recommend') || lowerQuestion.includes('next steps')) {
      response = 'I recommend: 1) Clinical correlation with patient symptoms, 2) Additional MRI sequences including contrast-enhanced T1, FLAIR, and potentially diffusion-weighted imaging, 3) Follow-up imaging to assess for any changes over time, 4) Depending on clinical suspicion, consideration for advanced imaging such as MR spectroscopy.';
    } else {
      response = 'To properly address your question about this image, I would need more specific details about what aspect you're interested in. I can provide information about the apparent findings, potential implications, or suggested next steps for clinical assessment.';
    }
    
    return NextResponse.json({
      response
    });
  } catch (error) {
    console.error('Error in image analysis chat API:', error);
    return NextResponse.json(
      { error: 'Failed to process question' },
      { status: 500 }
    );
  }
}