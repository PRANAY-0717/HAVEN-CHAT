'use client';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Brain, Sparkles, MessageSquare, AlertCircle } from 'lucide-react';

interface AIInsightsProps {
  userMessages: any[];
}

export function AIInsights({ userMessages }: AIInsightsProps) {
  const toxicCount = userMessages.filter((m) => m.is_toxic).length;
  const totalCount = userMessages.length;
  const toxicRatio = totalCount > 0 ? toxicCount / totalCount : 0;

  let behaviorProfile = {
    title: 'Harmonious Communicator',
    description: 'Your communication is consistently respectful and safe.',
    color: 'bg-green-500',
    suggestions: [
      'Great job! Your tone helps build a healthy community.',
      'Keep using clear and constructive language.',
    ],
  };

  if (toxicRatio > 0.2) {
    behaviorProfile = {
      title: 'Aggressive Communicator',
      description: 'Our system has detected multiple instances of potentially aggressive language.',
      color: 'bg-red-500',
      suggestions: [
        'Try using more neutral phrasing in your responses.',
        'Consider re-reading your messages before sending.',
        'Take a short break if you feel frustrated.',
      ],
    };
  } else if (toxicRatio > 0) {
    behaviorProfile = {
      title: 'Occasional Edge Case',
      description: 'You have had a few messages flagged. Try to be more mindful.',
      color: 'bg-orange-500',
      suggestions: [
        'Be careful with sarcasm, it can sometimes be misclassified.',
        'Try to avoid inflammatory language.',
      ],
    };
  }

  return (
    <div className="space-y-6">
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${behaviorProfile.color} text-white`}>
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <CardTitle className="text-lg">{behaviorProfile.title}</CardTitle>
              <p className="text-xs text-muted-foreground">AI Behavior Profile</p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-foreground/80 leading-relaxed">
            {behaviorProfile.description}
          </p>
        </CardContent>
      </Card>

      <div className="space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
          <Sparkles className="w-3 h-3" />
          Suggestions for you
        </h4>
        <div className="space-y-2">
          {behaviorProfile.suggestions.map((s, i) => (
            <div key={i} className="p-3 bg-muted rounded-lg flex items-start gap-3 border border-muted-foreground/10">
              <div className="p-1 bg-background rounded-full mt-0.5">
                <MessageSquare className="w-3 h-3 text-primary" />
              </div>
              <p className="text-sm">{s}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="p-4 rounded-xl bg-muted/30 border border-muted flex items-center gap-4">
        <div className="flex-1">
          <div className="text-xs text-muted-foreground mb-1 uppercase tracking-tight">Toxicity Risk</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full ${toxicRatio > 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
                style={{ width: `${toxicRatio * 100}%` }}
              />
            </div>
            <span className="text-xs font-bold">{Math.round(toxicRatio * 100)}%</span>
          </div>
        </div>
        <AlertCircle className="w-5 h-5 text-muted-foreground/40" />
      </div>
    </div>
  );
}
