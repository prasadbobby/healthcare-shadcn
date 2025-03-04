// src/lib/file-auth-adapter.ts
import type { Adapter } from 'next-auth/adapters';
import { 
  getUsers, 
  getUserByEmail, 
  getUserById, 
  createUser, 
  updateUser,
  logUserActivity
} from './file-storage';

export function FileAdapter(): Adapter {
  return {
    async createUser(user) {
      const newUser = await createUser({
        name: user.name,
        email: user.email,
        emailVerified: user.emailVerified,
        image: user.image
      });
      return {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
        emailVerified: newUser.emailVerified,
        image: newUser.image
      };
    },
    
    async getUser(id) {
      const user = await getUserById(id);
      if (!user) return null;
      
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.emailVerified,
        image: user.image
      };
    },
    
    async getUserByEmail(email) {
      const user = await getUserByEmail(email);
      if (!user) return null;
      
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.emailVerified,
        image: user.image
      };
    },
    
    async getUserByAccount({ provider, providerAccountId }) {
      const users = await getUsers();
      const user = users.find(user => 
        user.providerAccounts?.some(
          account => account.provider === provider && account.providerAccountId === providerAccountId
        )
      );
      
      if (!user) return null;
      
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.emailVerified,
        image: user.image
      };
    },
    
    async updateUser(user) {
      const updated = await updateUser(user.id, user);
      if (!updated) throw new Error("User not found");
      
      return {
        id: updated.id,
        name: updated.name,
        email: updated.email,
        emailVerified: updated.emailVerified,
        image: updated.image
      };
    },
    
    async linkAccount(account) {
      const user = await getUserById(account.userId);
      if (!user) throw new Error("User not found");
      
      const providerAccount = {
        provider: account.provider,
        providerAccountId: account.providerAccountId
      };
      
      const updatedUser = await updateUser(user.id, {
        providerAccounts: [...(user.providerAccounts || []), providerAccount]
      });
      
      return account;
    },
    
    // Basic implementation of required methods
    async deleteUser(userId) {
      // Implementation not needed for basic auth
      return;
    },
    
    async unlinkAccount({ provider, providerAccountId }) {
      // Implementation not needed for basic auth
      return;
    },
    
    async createSession({ sessionToken, userId, expires }) {
      // For JWT strategy, we don't need to implement this
      return { sessionToken, userId, expires };
    },
    
    async getSessionAndUser(sessionToken) {
      // For JWT strategy, we don't need to implement this
      return null;
    },
    
    async updateSession(session) {
      // For JWT strategy, we don't need to implement this
      return session;
    },
    
    async deleteSession(sessionToken) {
      // For JWT strategy, we don't need to implement this
      return;
    },
    
    async createVerificationToken(verificationToken) {
      // Implementation not needed for OAuth providers
      return verificationToken;
    },
    
    async useVerificationToken({ identifier, token }) {
      // Implementation not needed for OAuth providers
      return null;
    }
  };
}