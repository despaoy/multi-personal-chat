'use client';

/**
 * 仪表盘客户端组件
 *
 * 智能助手管理平台的首页仪表盘，展示：
 * - 四个统计卡片（今日回复数、平均响应时间、活跃会话、模型负载）
 * - 24 小时活动趋势折线图
 * - 系统状态面板（模型信息、系统资源、服务状态）
 * - 快捷操作入口（测试回复、切换模型、检查服务、管理会话）
 */

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { StatCard } from '@/components/dashboard/StatCard';
import { TestChatDialog } from '@/components/dashboard/TestChatDialog';
import { SessionManagerDialog } from '@/components/dashboard/SessionManagerDialog';
import { MessageSquare, Zap, Clock, Users, BrainCircuit, RefreshCw, AlertCircle, LogIn } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useStats } from '@/hooks/useStats';
import { useLoras } from '@/hooks/useLoras';
import { useServices } from '@/hooks/useServices';
import { api } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

// 懒加载 ActivityChart：recharts 是重依赖（~200KB），仅在仪表盘可见时加载
const ActivityChart = dynamic(
  () => import('@/components/dashboard/ActivityChart').then(m => ({ default: m.ActivityChart })),
  {
    loading: () => (
      <Card>
        <CardHeader><CardTitle>今日活动趋势</CardTitle></CardHeader>
        <CardContent><div className="h-[300px]"><Skeleton className="h-full w-full" /></div></CardContent>
      </Card>
    ),
    ssr: false,
  }
);

export default function DashboardClient() {
  const { user, loading: authLoading } = useAuth();
  const { stats, loading: statsLoading, error: statsError, refetch: refetchStats } = useStats(!!user && !authLoading);
  const { loras } = useLoras(!!user && !authLoading);
  const { services, loading: servicesLoading, error: servicesError, refetch: refetchServices } = useServices(!!user && !authLoading);

  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // 获取当前激活的LoRA
  const activeLora = loras.find(lora => lora.status === 'active');

  if (statsError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <div className="text-center">
          <h3 className="text-lg font-semibold">加载失败</h3>
          <p className="text-muted-foreground">{statsError}</p>
        </div>
        <Button onClick={refetchStats}>
          <RefreshCw className="mr-2 h-4 w-4" />
          重试
        </Button>
      </div>
    );
  }

  if (authLoading || !isMounted) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">仪表盘</h2>
            <p className="text-muted-foreground">
              欢迎回来！查看您的智能助手运行状态和最新活动。
            </p>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-32 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <LogIn className="h-12 w-12 text-muted-foreground" />
        <div className="text-center">
          <h3 className="text-lg font-semibold">请先登录</h3>
          <p className="text-muted-foreground">登录后即可查看仪表盘</p>
        </div>
        <Button onClick={() => window.location.href = '/login'}>
          <LogIn className="mr-2 h-4 w-4" />
          前往登录
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">仪表盘</h2>
          <p className="text-muted-foreground">
            欢迎回来！查看您的智能助手运行状态和最新活动。
          </p>
        </div>
        <Button variant="ghost" size="icon" onClick={() => { refetchStats(); refetchServices(); }} disabled={statsLoading || servicesLoading}>
          <RefreshCw className={`h-4 w-4 ${(statsLoading || servicesLoading) ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {/* 统计卡片 */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statsLoading ? (
          <>
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-32 rounded-lg" />
          </>
        ) : (
          <>
            <StatCard
              title="今日回复数"
              value={stats?.todayReplies?.toString() || '0'}
              icon={MessageSquare}
            />
            <StatCard
              title="平均响应时间"
              value={`${stats?.avgResponseTime || 0}s`}
              icon={Clock}
            />
            <StatCard
              title="活跃会话"
              value={stats?.activeSessions?.toString() || '0'}
              icon={Users}
            />
            <StatCard
              title="模型负载"
              value={`${stats?.modelLoad || 0}%`}
              icon={Zap}
            />
          </>
        )}
      </div>

      {/* 图表和模型状态 */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ActivityChart />
        </div>
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BrainCircuit className="h-5 w-5" />
                系统状态
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {(statsLoading || servicesLoading) ? (
                <>
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-3/4" />
                </>
              ) : (
                <>
                  {/* 模型信息 */}
                  <div className="pb-3 border-b">
                    <h4 className="text-sm font-semibold mb-2 text-muted-foreground">模型信息</h4>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm">基座模型</span>
                      <span className="text-sm font-medium">Qwen3-8B</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">当前LoRA</span>
                      <span className="text-sm font-medium text-primary">
                        {activeLora?.name || '基础模型（无LoRA）'}
                      </span>
                    </div>
                  </div>

                  {/* 系统资源 */}
                  <div className="pb-3 border-b">
                    <h4 className="text-sm font-semibold mb-2 text-muted-foreground">系统资源</h4>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm">CPU使用率</span>
                      <span className="text-sm font-medium">{stats?.cpuUsage || 0}%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">内存使用</span>
                      <span className="text-sm font-medium">
                        {stats?.memoryUsage?.used || 0}GB / {stats?.memoryUsage?.total || 16}GB
                      </span>
                    </div>
                  </div>

                  {/* 服务状态 */}
                  <div>
                    <h4 className="text-sm font-semibold mb-2 text-muted-foreground">服务状态</h4>
                    {servicesError ? (
                      <div className="text-sm text-destructive">{servicesError}</div>
                    ) : (
                      <div className="space-y-2">
                        {services.map((service, index) => (
                          <div key={index} className="flex items-center justify-between">
                            <span className="text-sm">{service.name}</span>
                            <div className="flex items-center gap-2">
                              <div className={`w-2 h-2 rounded-full ${
                                service.status === 'running' ? 'bg-green-500' :
                                service.status === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                              }`} />
                              <span className={`text-xs font-medium ${
                                service.status === 'running' ? 'text-green-600' :
                                service.status === 'connecting' ? 'text-yellow-600' : 'text-red-600'
                              }`}>
                                {service.status === 'running' ? '运行中' :
                                 service.status === 'connecting' ? '连接中' : '已停止'}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 快捷操作 */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>快捷操作</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid grid-cols-2 gap-3">
              {/* 测试回复 - 聊天式对话（已拆分为独立组件，隔离输入状态） */}
              <TestChatDialog loras={loras} />

              {/* 切换模型 */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" className="flex flex-col items-center justify-center rounded-lg border p-4 h-auto hover:bg-muted transition-colors">
                    <BrainCircuit className="h-6 w-6 mb-2 text-primary" />
                    <span className="text-sm font-medium">切换模型</span>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[450px] max-h-[85vh] flex flex-col">
                  <DialogHeader>
                    <DialogTitle>切换LoRA模型</DialogTitle>
                    <DialogDescription>
                      选择要激活的LoRA模型，切换后所有新消息将使用该模型回复
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-2 py-4 overflow-y-auto flex-1 min-h-0">
                    {loras.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        暂无LoRA模型，请先在LoRA管理页面添加
                      </div>
                    ) : (
                      loras.map((lora) => (
                        <div key={lora.id} className="flex items-center justify-between p-3 rounded-lg border">
                          <div>
                            <div className="font-medium text-sm">{lora.name}</div>
                            <div className="text-xs text-muted-foreground">{lora.description || lora.style}</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={lora.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                              {lora.status === 'active' ? '激活中' : '未激活'}
                            </Badge>
                            {lora.status !== 'active' && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={async () => {
                                  try {
                                    await api.toggleLoraStatus(lora.id, lora.status);
                                  } catch {
                                    // ignore
                                  }
                                }}
                              >
                                激活
                              </Button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </DialogContent>
              </Dialog>

              {/* 服务进程由外部 supervisor/screen 管理，前端提供可用的状态入口，
                  不展示一个无法安全实现的“重启”假按钮。 */}
              <Button asChild variant="ghost" className="flex flex-col items-center justify-center rounded-lg border p-4 h-auto hover:bg-muted transition-colors">
                <Link href="/monitor">
                  <Zap className="h-6 w-6 mb-2 text-primary" />
                  <span className="text-sm font-medium">检查服务</span>
                </Link>
              </Button>

              {/* 管理会话（已拆分为独立组件，隔离会话列表状态） */}
              <SessionManagerDialog />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
