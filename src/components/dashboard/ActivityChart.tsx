'use client';

/**
 * 活动趋势图表组件
 *
 * 使用 Recharts 折线图展示 24 小时内的消息收发趋势。
 * 包含两个数据序列：收到消息（messages）和回复消息（replies），
 * 支持加载骨架屏和错误重试。
 *
 * 通过 useActivity Hook 获取数据，内置错误状态展示和重试按钮。
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Skeleton } from '@/components/ui/skeleton';
import { useActivity } from '@/hooks/useActivity';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function ActivityChart() {
  const { activity, loading, error, refetch } = useActivity();

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>今日活动趋势</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-[300px] space-y-4">
            <AlertCircle className="h-8 w-8 text-destructive" />
            <div className="text-center">
              <p className="text-muted-foreground">加载失败</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
            <Button onClick={refetch} variant="ghost" size="sm">
              <RefreshCw className="mr-2 h-4 w-4" />
              重试
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>今日活动趋势</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <Skeleton className="h-full w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>今日活动趋势</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] min-h-0 min-w-0">
          <ResponsiveContainer
            width="100%"
            height="100%"
            minWidth={0}
            minHeight={0}
            initialDimension={{ width: 800, height: 300 }}
          >
            <LineChart data={activity}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="time" stroke="var(--muted-foreground)" tick={{ fontSize: 12 }} />
              <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 12 }} />
              <Tooltip
                wrapperStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                }}
              />
              <Line
                type="monotone"
                dataKey="messages"
                stroke="var(--chart-1)"
                strokeWidth={2}
                name="收到消息"
              />
              <Line
                type="monotone"
                dataKey="replies"
                stroke="var(--chart-2)"
                strokeWidth={2}
                name="回复消息"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
