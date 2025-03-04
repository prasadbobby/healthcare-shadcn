'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { healthAnalyticsData, recentSessions } from "@/constants/data";
import { Activity, Microscope, FileText, Pill, TrendingUp } from "lucide-react"; // Removed unused imports
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, Tooltip, XAxis } from "recharts"; // Removed YAxis
import { Avatar, AvatarFallback } from "@/components/ui/avatar"; // Removed unused AvatarImage
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

const chartConfig = {
  clinical: {
    label: 'Clinical Case Analysis',
    color: 'hsl(var(--chart-1))'
  },
  literature: {
    label: 'Medical Literature Review',
    color: 'hsl(var(--chart-2))'
  },
  symptom: {
    label: 'Symptom Analysis',
    color: 'hsl(var(--chart-3))'
  },
  drug: {
    label: 'Drug Interaction',
    color: 'hsl(var(--chart-4))'
  }
} satisfies ChartConfig;

// Data for the pie chart
const sessionTypeData = [
  { name: 'Clinical', value: 42, fill: 'hsl(var(--chart-1))' },
  { name: 'Literature', value: 28, fill: 'hsl(var(--chart-2))' },
  { name: 'Symptom', value: 33, fill: 'hsl(var(--chart-3))' },
  { name: 'Drug', value: 27, fill: 'hsl(var(--chart-4))' }
];

export default function HealthOverview() {
  return (
    <div className="flex flex-1 flex-col space-y-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Sessions
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">130</div>
            <p className="text-xs text-muted-foreground">
              +18.2% from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Clinical Cases
            </CardTitle>
            <Microscope className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">42</div>
            <p className="text-xs text-muted-foreground">
              +12.5% from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Literature Reviews</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">28</div>
            <p className="text-xs text-muted-foreground">
              +6.8% from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Drug Interactions
            </CardTitle>
            <Pill className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">27</div>
            <p className="text-xs text-muted-foreground">
              +21.4% since last week
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Session Analytics</CardTitle>
            <CardDescription>
              Generated analysis sessions over the last 6 months
            </CardDescription>
          </CardHeader>
          <CardContent className="px-2 sm:p-6">
            <ChartContainer
              config={chartConfig}
              className="aspect-auto h-[300px] w-full"
            >
              <AreaChart
                accessibilityLayer
                data={healthAnalyticsData}
                margin={{
                  left: 12,
                  right: 12
                }}
              >
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="month"
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent indicator="dot" />}
                />
                <Area
                  dataKey="clinical"
                  type="monotone"
                  fill="var(--color-clinical)"
                  fillOpacity={0.4}
                  stroke="var(--color-clinical)"
                  stackId="1"
                />
                <Area
                  dataKey="literature"
                  type="monotone"
                  fill="var(--color-literature)"
                  fillOpacity={0.4}
                  stroke="var(--color-literature)"
                  stackId="1"
                />
                <Area
                  dataKey="symptom"
                  type="monotone"
                  fill="var(--color-symptom)"
                  fillOpacity={0.4}
                  stroke="var(--color-symptom)"
                  stackId="1"
                />
                <Area
                  dataKey="drug"
                  type="monotone"
                  fill="var(--color-drug)"
                  fillOpacity={0.4}
                  stroke="var(--color-drug)"
                  stackId="1"
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Recent Sessions</CardTitle>
            <CardDescription>
              Your most recent analysis sessions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {recentSessions.map((session) => (
                <div key={session.id} className="flex items-center">
                  <Avatar className="h-9 w-9">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      {session.type.substring(0, 2)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">{session.type}</p>
                    <p className="text-sm text-muted-foreground">
                      {session.summary}
                    </p>
                  </div>
                  <div className="ml-auto">
                    <Badge
                      variant={
                        session.status === 'completed' 
                          ? 'outline' 
                          : session.status === 'in-progress' 
                            ? 'secondary' 
                            : 'default'
                      }
                      className={cn(
                        "ml-2",
                        session.status === 'completed' && "border-green-500 text-green-500",
                        session.status === 'in-progress' && "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
                        session.status === 'scheduled' && "bg-primary"
                      )}
                    >
                      {session.status === 'completed' && "Completed"}
                      {session.status === 'in-progress' && "In Progress"}
                      {session.status === 'scheduled' && "Scheduled"}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Monthly Breakdown</CardTitle>
            <CardDescription>
              Session counts by type per month
            </CardDescription>
          </CardHeader>
          <CardContent className="px-2 sm:p-6">
            <ChartContainer
              config={chartConfig}
              className="aspect-auto h-[300px] w-full"
            >
              <BarChart
                accessibilityLayer
                data={healthAnalyticsData}
                margin={{
                  left: 12,
                  right: 12
                }}
              >
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="month"
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <ChartTooltip
                  content={<ChartTooltipContent className="w-[150px]" />}
                />
                <Bar dataKey="clinical" fill="var(--color-clinical)" />
                <Bar dataKey="literature" fill="var(--color-literature)" />
                <Bar dataKey="symptom" fill="var(--color-symptom)" />
                <Bar dataKey="drug" fill="var(--color-drug)" />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Session Distribution</CardTitle>
            <CardDescription>
              Breakdown by session type
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 pb-0">
            <ChartContainer
              config={chartConfig}
              className="mx-auto aspect-square max-h-[360px]"
            >
              <PieChart>
                <ChartTooltip content={<ChartTooltipContent hideLabel />} />
                <Pie
                  data={sessionTypeData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  fill="#8884d8"
                >
                  {sessionTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Legend />
              </PieChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}