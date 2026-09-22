'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowUpRight, ChevronDown, Clock, Database, MessageSquare, RefreshCw, Users } from 'lucide-react';
import { StatCard } from '@/components/dashboard/StatCard';
import { TestChatDialog } from '@/components/dashboard/TestChatDialog';
import { SessionManagerDialog } from '@/components/dashboard/SessionManagerDialog';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { useStats } from '@/hooks/useStats';
import { useLoras } from '@/hooks/useLoras';
import { useServices } from '@/hooks/useServices';
import { useAuth } from '@/contexts/AuthContext';

const ActivityChart = dynamic(
  () => import('@/components/dashboard/ActivityChart').then(module => ({ default: module.ActivityChart })),
  { loading: () => <Skeleton className="h-[300px] rounded-xl" />, ssr: false },
);

export default function DashboardClient() {
  const { user, loading: authLoading } = useAuth();
  const enabled = user?.role === 'admin' && !authLoading;
  const { stats, loading, error, refetch } = useStats(enabled);
  const { loras } = useLoras(enabled);
  const { services, loading: servicesLoading, error: servicesError, refetch: refetchServices } = useServices(enabled);
  const [showOperations, setShowOperations] = useState(false);
  const activeLora = loras.find(lora => lora.status === 'active');

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">工作台</h1>
          <p className="mt-2 text-sm text-muted-foreground">从对话开始，按需要调整人物与知识。</p>
        </div>
        <Button variant="ghost" size="icon" aria-label="刷新运行数据" title="刷新运行数据" disabled={!enabled || loading || servicesLoading} onClick={() => { refetch(); refetchServices(); }}>
          <RefreshCw className={`h-4 w-4 ${loading || servicesLoading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      <section aria-labelledby="workspace-actions" className="space-y-3">
        <h2 id="workspace-actions" className="text-sm font-medium">常用操作</h2>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <TestChatDialog loras={loras} />
          <Button asChild variant="ghost" className="h-auto flex-col rounded-lg border p-4 hover:bg-muted">
            <Link href="/characters"><Users className="mb-2 h-6 w-6 text-primary" /><span>角色与记忆</span></Link>
          </Button>
          <Button asChild variant="ghost" className="h-auto flex-col rounded-lg border p-4 hover:bg-muted">
            <Link href="/knowledge"><Database className="mb-2 h-6 w-6 text-primary" /><span>维护知识库</span></Link>
          </Button>
          <SessionManagerDialog />
        </div>
      </section>

      <section aria-labelledby="workspace-today" className="space-y-3">
        <h2 id="workspace-today" className="text-sm font-medium">今日概况</h2>
        {error ? (
          <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border p-4 text-sm">
            <p className="text-muted-foreground">运行数据暂不可用，不影响打开其他功能。{error}</p>
            <Button variant="outline" size="sm" onClick={refetch}>重新加载</Button>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            {loading || authLoading ? Array.from({ length: 3 }, (_, index) => <Skeleton key={index} className="h-28 rounded-xl" />) : <>
              <StatCard compact title="今日回复" value={stats?.todayReplies?.toString() ?? '—'} icon={MessageSquare} />
              <StatCard compact title="平均响应" value={stats?.avgResponseTime != null ? `${stats.avgResponseTime}s` : '—'} icon={Clock} />
              <StatCard compact title="活跃会话" value={stats?.activeSessions?.toString() ?? '—'} icon={Users} />
            </>}
          </div>
        )}
      </section>

      <section className="border-t pt-5" aria-labelledby="workspace-operations">
        <div className="flex items-center justify-between gap-3">
          <button id="workspace-operations" type="button" aria-expanded={showOperations} aria-controls="operations-content" onClick={() => setShowOperations(value => !value)}
            className="flex min-h-11 items-center gap-2 rounded-md text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${showOperations ? 'rotate-180' : ''}`} />运行详情与趋势
          </button>
          <Link href="/monitor" className="flex items-center gap-1 rounded text-xs text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">完整监控<ArrowUpRight className="h-3.5 w-3.5" /></Link>
        </div>
        <div id="operations-content" hidden={!showOperations}>
          {showOperations && <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <div className="min-w-0 lg:col-span-2"><ActivityChart /></div>
            <Card>
              <CardHeader><CardTitle className="text-base">运行状态</CardTitle></CardHeader>
              <CardContent className="space-y-5 text-sm">
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">当前 LoRA</p>
                  <p>{activeLora?.name || '未加载或未启用 LoRA'}</p>
                  <Link href="/lora" className="inline-flex items-center gap-1 text-xs text-primary">管理模型<ArrowUpRight className="h-3 w-3" /></Link>
                </div>
                <dl className="space-y-2 border-t pt-4">
                  <div className="flex justify-between"><dt className="text-muted-foreground">模型负载</dt><dd>{!error && stats?.modelLoad != null ? `${stats.modelLoad}%` : '—'}</dd></div>
                  <div className="flex justify-between"><dt className="text-muted-foreground">CPU</dt><dd>{!error && stats?.cpuUsage != null ? `${stats.cpuUsage}%` : '—'}</dd></div>
                  <div className="flex justify-between"><dt className="text-muted-foreground">内存</dt><dd>{!error && stats?.memoryUsage ? `${stats.memoryUsage.used} / ${stats.memoryUsage.total} GB` : '—'}</dd></div>
                </dl>
                <div className="space-y-2 border-t pt-4">
                  {servicesLoading ? <Skeleton className="h-12" /> : servicesError ? <p role="alert" className="text-destructive">{servicesError}</p> : services.length ? services.map(service => (
                    <div key={service.name} className="flex justify-between gap-3"><span className="text-muted-foreground">{service.name}</span><span>{service.status === 'running' ? '运行中' : service.status === 'connecting' ? '连接中' : '未就绪'}</span></div>
                  )) : <p className="text-muted-foreground">暂无服务状态</p>}
                </div>
              </CardContent>
            </Card>
          </div>}
        </div>
      </section>
    </div>
  );
}
