import { NextRequest, NextResponse } from 'next/server';
import { saveUserToStorage } from '@/lib/google-cloud-storage';

export async function POST(req: NextRequest) {
  try {
    const { user } = await req.json();
    
    if (!user) {
      return NextResponse.json(
        { error: 'User data is required' },
        { status: 400 }
      );
    }
    
    // Store user data in Google Cloud Storage
    const result = await saveUserToStorage(user);
    
    if (!result.success) {
      return NextResponse.json(
        { error: 'Failed to store user data' },
        { status: 500 }
      );
    }
    
    return NextResponse.json({ success: true, path: result.path });
  } catch (error) {
    console.error('Error in API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}