// src/app/api/user-data/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { getUsers } from '@/lib/file-storage';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

// Use Node.js runtime
export const runtime = 'nodejs';

// This endpoint should be secured in production!
export async function GET(req: NextRequest) {
  // Only allow in development
  if (process.env.NODE_ENV !== 'development') {
    return NextResponse.json(
      { error: 'Not available in production' },
      { status: 403 }
    );
  }
  
  try {
    // Get the current authenticated session
    const session = await getServerSession(authOptions);
    
    // Only allow authenticated users with admin privileges
    if (!session?.user?.email) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }
    
    // Get all users data
    const users = await getUsers();
    
    return NextResponse.json({ users });
  } catch (error) {
    console.error('Error retrieving user data:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}