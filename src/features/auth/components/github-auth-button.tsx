// src/features/auth/components/github-auth-button.tsx
'use client';

import { signIn } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Icons } from '@/components/icons';

export default function GithubSignInButton() {
  return (
    <Button
      className='w-full'
      variant='outline'
      type='button'
      onClick={() => signIn('github', { callbackUrl: '/dashboard' })}
    >
      <Icons.gitHub className='mr-2 h-4 w-4' />
      Continue with Github
    </Button>
  );
}