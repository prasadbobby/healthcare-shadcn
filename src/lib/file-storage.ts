// src/lib/file-storage.ts
import fs from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');

// Ensure the data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Initialize users file if it doesn't exist
if (!fs.existsSync(USERS_FILE)) {
  fs.writeFileSync(USERS_FILE, JSON.stringify({ users: [] }), 'utf8');
}

type User = {
  id: string;
  name?: string | null;
  email: string;
  emailVerified?: Date | null;
  image?: string | null;
  createdAt: Date;
  providerAccounts?: {
    provider: string;
    providerAccountId: string;
  }[];
};

export async function getUsers(): Promise<User[]> {
  try {
    const data = fs.readFileSync(USERS_FILE, 'utf8');
    const parsed = JSON.parse(data);
    return parsed.users || [];
  } catch (error) {
    console.error('Error reading users file:', error);
    return [];
  }
}

export async function getUserByEmail(email: string): Promise<User | null> {
  const users = await getUsers();
  return users.find(user => user.email === email) || null;
}

export async function getUserById(id: string): Promise<User | null> {
  const users = await getUsers();
  return users.find(user => user.id === id) || null;
}

export async function createUser(userData: Omit<User, 'id' | 'createdAt'> & { id?: string }): Promise<User> {
  const users = await getUsers();
  
  // Check if user with this email already exists
  const existingUser = users.find(user => user.email === userData.email);
  if (existingUser) {
    return existingUser;
  }
  
  // Create new user
  const newUser: User = {
    id: userData.id || `user_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    name: userData.name,
    email: userData.email,
    emailVerified: userData.emailVerified,
    image: userData.image,
    createdAt: new Date(),
    providerAccounts: userData.providerAccounts || []
  };
  
  // Add to users array and save
  users.push(newUser);
  fs.writeFileSync(USERS_FILE, JSON.stringify({ users }, null, 2), 'utf8');
  
  // Log to a separate login activity file
  logUserActivity('login', newUser);
  
  return newUser;
}

export async function updateUser(id: string, updateData: Partial<User>): Promise<User | null> {
  const users = await getUsers();
  const userIndex = users.findIndex(user => user.id === id);
  
  if (userIndex === -1) return null;
  
  // Update user
  const updatedUser = { ...users[userIndex], ...updateData };
  users[userIndex] = updatedUser;
  
  // Save changes
  fs.writeFileSync(USERS_FILE, JSON.stringify({ users }, null, 2), 'utf8');
  
  return updatedUser;
}

// Function to log user activity
export async function logUserActivity(activity: string, user: { id: string, email: string }) {
  const LOG_FILE = path.join(DATA_DIR, 'user_activity.json');
  
  // Create log file if it doesn't exist
  if (!fs.existsSync(LOG_FILE)) {
    fs.writeFileSync(LOG_FILE, JSON.stringify({ logs: [] }), 'utf8');
  }
  
  // Read existing logs
  const logData = fs.readFileSync(LOG_FILE, 'utf8');
  const logs = JSON.parse(logData).logs || [];
  
  // Add new log entry
  logs.push({
    userId: user.id,
    email: user.email,
    activity,
    timestamp: new Date().toISOString()
  });
  
  // Save updated logs
  fs.writeFileSync(LOG_FILE, JSON.stringify({ logs }, null, 2), 'utf8');
}