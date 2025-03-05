import { NavItem } from 'types';

export type Product = {
  photo_url: string;
  name: string;
  description: string;
  created_at: string;
  price: number;
  id: number;
  category: string;
  updated_at: string;
};

//Info: The following data is used for the sidebar navigation and Cmd K bar.
export const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    url: '/dashboard/overview',
    icon: 'dashboard',
    isActive: false,
    shortcut: ['d', 'd'],
    items: [] // Empty array as there are no child items for Dashboard
  },
  {
    title: 'Generate',
    url: '#', // Placeholder as there is no direct link for the parent
    icon: 'medical',
    isActive: true,
    items: [
      {
        title: 'Clinical Case Analysis',
        url: '/dashboard/chat/clinical',
        icon: 'clinical',
        shortcut: ['c', 'c']
      },
      {
        title: 'Medical Literature Review',
        url: '/dashboard/chat/literature',
        icon: 'literature',
        shortcut: ['l', 'l']
      },
      {
        title: 'Symptom Analysis',
        url: '/dashboard/chat/symptom',
        icon: 'symptom',
        shortcut: ['s', 's']
      },
      {
        title: 'Drug Interaction',
        url: '/dashboard/chat/drug',
        icon: 'drug',
        shortcut: ['d', 'i']
      },
      {
        title: 'Image Analysis',
        url: '/dashboard/image-analysis',
        icon: 'image',
        shortcut: ['i', 'a']
      }
    ]
  },
  {
    title: 'Profile',
    url: '/dashboard/profile',
    icon: 'userPen',
    shortcut: ['p', 'p'],
    isActive: false,
    items: [] // No child items
  }
];
// Mock data for recent healthcare sessions
export interface HealthSession {
  id: number;
  type: string;
  date: string;
  summary: string;
  status: 'completed' | 'in-progress' | 'scheduled';
}

export const recentSessions: HealthSession[] = [
  {
    id: 1,
    type: 'Clinical Case Analysis',
    date: '2025-03-01T10:30:00',
    summary: 'Analysis of patient case with chronic heart failure',
    status: 'completed'
  },
  {
    id: 2,
    type: 'Drug Interaction',
    date: '2025-03-02T14:15:00',
    summary: 'Interaction check for new diabetes medication regimen',
    status: 'completed'
  },
  {
    id: 3,
    type: 'Symptom Analysis',
    date: '2025-03-03T09:00:00',
    summary: 'Evaluation of neurological symptoms',
    status: 'in-progress'
  },
  {
    id: 4,
    type: 'Medical Literature Review',
    date: '2025-03-04T16:45:00',
    summary: "Latest research on Alzheimer's treatment approaches", // Fixed quote
    status: 'scheduled'
  }
];

export interface ChartData {
  month: string;
  clinical: number;
  literature: number;
  symptom: number;
  drug: number;
}

export const healthAnalyticsData: ChartData[] = [
  { month: 'Jan', clinical: 18, literature: 12, symptom: 24, drug: 10 },
  { month: 'Feb', clinical: 25, literature: 15, symptom: 30, drug: 18 },
  { month: 'Mar', clinical: 32, literature: 20, symptom: 28, drug: 22 },
  { month: 'Apr', clinical: 20, literature: 25, symptom: 35, drug: 15 },
  { month: 'May', clinical: 40, literature: 22, symptom: 32, drug: 20 },
  { month: 'Jun', clinical: 35, literature: 30, symptom: 28, drug: 25 }
];