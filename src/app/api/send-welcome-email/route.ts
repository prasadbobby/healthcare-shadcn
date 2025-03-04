// src/app/api/send-welcome-email/route.ts
import { NextRequest, NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

// Use Node.js runtime for this API route
export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  try {
    const { email, name } = await req.json();
    
    if (!email) {
      return NextResponse.json(
        { error: 'Email is required' },
        { status: 400 }
      );
    }
    
    // Set up email transport using Gmail
    const emailServer = process.env.EMAIL_SERVER || '';
    const transporter = nodemailer.createTransport(emailServer);
    
    // Email content
    const mailOptions = {
      from: process.env.EMAIL_FROM,
      to: email,
      subject: 'Welcome to Healthcare Dashboard',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2>Welcome to Healthcare Dashboard, ${name}!</h2>
          <p>Thank you for joining our platform. We're excited to help you with your healthcare analysis needs.</p>
          <p>With our platform, you can:</p>
          <ul>
            <li>Perform clinical case analyses</li>
            <li>Review medical literature</li>
            <li>Analyze symptoms</li>
            <li>Check drug interactions</li>
          </ul>
          <p>Get started by exploring our dashboard and trying out our various analysis tools.</p>
          <div style="margin-top: 30px;">
            <a href="${process.env.NEXTAUTH_URL}/dashboard" style="background-color: #4F46E5; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
              Go to Dashboard
            </a>
          </div>
        </div>
      `
    };
    
    // Send email
    await transporter.sendMail(mailOptions);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error in API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}