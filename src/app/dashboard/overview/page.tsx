import PageContainer from '@/components/layout/page-container';
import HealthOverview from '@/features/overview/components/health-overview';

export const metadata = {
  title: 'Healthcare Dashboard : Overview'
};

export default function OverviewPage() {
  return (
    <PageContainer>
      <div className='space-y-4'>
        <h2 className='text-3xl font-bold tracking-tight'>Healthcare Dashboard</h2>
        <p className='text-muted-foreground'>
          Monitor and analyze your healthcare AI session activities
        </p>
        <HealthOverview />
      </div>
    </PageContainer>
  );
}