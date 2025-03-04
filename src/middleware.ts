// src/middleware.ts
import { NextResponse } from 'next/server';
import { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';

export async function middleware(req: NextRequest) {
  const token = await getToken({ 
    req,
    secret: process.env.NEXTAUTH_SECRET
  });
  
  const isLoggedIn = !!token;
  const { nextUrl } = req;
  const isAuthPage = nextUrl.pathname === '/';
  
  // Redirect authenticated users away from auth page
  if (isAuthPage && isLoggedIn) {
    return NextResponse.redirect(new URL('/dashboard', nextUrl));
  }
  
  // Redirect unauthenticated users to auth page
  if (!isLoggedIn && nextUrl.pathname.startsWith('/dashboard')) {
    return NextResponse.redirect(new URL('/', nextUrl));
  }
  
  return NextResponse.next();
}

// Only apply middleware to these paths
export const config = { 
  matcher: ['/', '/dashboard/:path*']
};