import { Storage } from '@google-cloud/storage';

// Initialize Google Cloud Storage
const storage = new Storage({
  projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
});

const bucketName = process.env.GOOGLE_CLOUD_BUCKET_NAME || 'healthcare-app-users';
const bucket = storage.bucket(bucketName);

export async function saveUserToStorage(userData: any) {
  try {
    const userId = userData.id || userData.email.replace(/[^a-zA-Z0-9]/g, '_');
    const filename = `users/${userId}.json`;
    
    // Create file in bucket
    const file = bucket.file(filename);
    
    // Upload user data as JSON
    await file.save(JSON.stringify(userData, null, 2), {
      contentType: 'application/json',
      metadata: {
        createdAt: new Date().toISOString(),
      },
    });
    
    return { success: true, path: filename };
  } catch (error) {
    console.error('Error storing user data:', error);
    return { success: false, error };
  }
}

export async function getUserFromStorage(userId: string) {
  try {
    const filename = `users/${userId}.json`;
    const file = bucket.file(filename);
    
    // Check if file exists
    const [exists] = await file.exists();
    if (!exists) {
      return { success: false, error: 'User not found' };
    }
    
    // Download and parse file content
    const [content] = await file.download();
    const userData = JSON.parse(content.toString());
    
    return { success: true, userData };
  } catch (error) {
    console.error('Error retrieving user data:', error);
    return { success: false, error };
  }
}

export async function updateUserInStorage(userId: string, updates: any) {
  try {
    // First get existing user data
    const { success, userData, error } = await getUserFromStorage(userId);
    
    if (!success || !userData) {
      return { success: false, error: error || 'Failed to retrieve user data' };
    }
    
    // Merge existing data with updates
    const updatedUserData = { ...userData, ...updates, updatedAt: new Date().toISOString() };
    
    // Save back to storage
    const filename = `users/${userId}.json`;
    const file = bucket.file(filename);
    
    await file.save(JSON.stringify(updatedUserData, null, 2), {
      contentType: 'application/json',
    });
    
    return { success: true, userData: updatedUserData };
  } catch (error) {
    console.error('Error updating user data:', error);
    return { success: false, error };
  }
}