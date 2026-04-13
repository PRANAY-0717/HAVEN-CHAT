'use client';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { BarChart, Users, ShieldAlert, ShieldCheck } from 'lucide-react';

interface StatsProps {
  messages: any[];
}

export function ChatDashboard({ messages }: StatsProps) {
  const totalMessages = messages.length;
  const toxicMessages = messages.filter((m) => m.is_toxic).length;
  const safeMessages = totalMessages - toxicMessages;
  const toxicityRate = totalMessages > 0 ? (toxicMessages / totalMessages) * 100 : 0;

  const uniqueUsers = new Set(messages.map((m) => m.user_id)).size;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Community Safety</CardTitle>
            <ShieldCheck className="w-4 h-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{100 - Math.round(toxicityRate)}%</div>
            <p className="text-xs text-muted-foreground">Safe interaction rate</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Toxic Content</CardTitle>
            <ShieldAlert className="w-4 h-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{toxicMessages}</div>
            <p className="text-xs text-muted-foreground">Messages blocked</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <BarChart className="w-4 h-4" />
            Toxicity Distribution
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span>Safe ({safeMessages})</span>
              <span>{100 - Math.round(toxicityRate)}%</span>
            </div>
            <Progress value={100 - toxicityRate} className="h-2 bg-red-100 [&>div]:bg-green-500" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span>Toxic ({toxicMessages})</span>
              <span>{Math.round(toxicityRate)}%</span>
            </div>
            <Progress value={toxicityRate} className="h-2 bg-green-100 [&>div]:bg-red-500" />
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
        <div className="flex items-center gap-3">
          <Users className="w-5 h-5 text-primary" />
          <div>
            <div className="text-sm font-bold">{uniqueUsers}</div>
            <div className="text-[10px] text-muted-foreground uppercase">Active Users</div>
          </div>
        </div>
        <Badge variant="secondary">Live Data</Badge>
      </div>
    </div>
  );
}
