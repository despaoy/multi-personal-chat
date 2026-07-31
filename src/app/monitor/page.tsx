'use client';

import { useState, useEffect, useCallback } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Activity, Server, Cpu, HardDrive, Wifi, Zap, RefreshCw, AlertCircle, Clock } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { api, StatsResponse, ServiceStatus } from '@/lib/api';
import { useSettings } from '@/contexts/SettingsContext';

export default function MonitorPage() {
  return (
    <AuthGuard requireAdmin>
      <MonitorContent />
    </AuthGuard>
  );
}

function MonitorContent() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const { t, formatTime } = useSettings();

  const fetchData = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      const [statsData, servicesData] = await Promise.all([
        api.getStats(),
        api.getServices(),
      ]);

      setStats(statsData);
      setServices(servicesData.services);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : t('monitor.loadFailed'));
      console.error('Failed to fetch monitor data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [t]);

  useEffect(() => {

    fetchData();

    // 标签页隐藏时跳过轮询，可见时恢复，减少后台无效请求
    const interval = setInterval(() => {
      if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
      fetchData(true);
    }, 10000);

    // 标签页重新可见时立即刷新一次，保证数据最新
    const onVisible = () => {
      if (document.visibilityState === 'visible') fetchData(true);
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [fetchData]);

  const formatBytes = (gb: number) => `${gb.toFixed(1)}GB`;

  const getProgressColor = (value: number) => {
    if (value >= 90) return '[&>div]:bg-red-500';
    if (value >= 70) return '[&>div]:bg-yellow-500';
    return '[&>div]:bg-primary';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500 animate-pulse';
      case 'connecting': return 'bg-yellow-500';
      default: return 'bg-red-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return { text: t('status.running'), className: 'text-green-600' };
      case 'connecting': return { text: t('status.connecting'), className: 'text-yellow-600' };
      default: return { text: t('status.stopped'), className: 'text-red-600' };
    }
  };

  if (error && !stats) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">{t('monitor.loadFailed')}</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
          <Button onClick={() => fetchData()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            {t('settings.retry')}
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{t('monitor.title')}</h2>
            <p className="text-muted-foreground">{t('monitor.description')}</p>
          </div>
          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="text-xs text-muted-foreground flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {t('monitor.updatedAt')} {formatTime(lastUpdated)}
              </span>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => fetchData(true)}
              disabled={refreshing}
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {loading ? (
            <>
              {[1, 2, 3, 4].map((i) => (
                <Card key={i}>
                  <CardHeader className="flex flex-row items-center justify-between pb-2">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-4 w-4" />
                  </CardHeader>
                  <CardContent>
                    <Skeleton className="h-8 w-24 mb-2" />
                    <Skeleton className="h-2 w-full" />
                  </CardContent>
                </Card>
              ))}
            </>
          ) : (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">{t('monitor.cpuUsage')}</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.cpuUsage ?? 0}%</div>
                  <Progress value={stats?.cpuUsage ?? 0} className={`mt-2 ${getProgressColor(stats?.cpuUsage ?? 0)}`} />
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">{t('monitor.gpuMemory')}</CardTitle>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  {stats?.gpuMemory?.total ? (
                    <>
                      <div className="text-2xl font-bold">
                        {formatBytes(stats.gpuMemory.used)} / {formatBytes(stats.gpuMemory.total)}
                      </div>
                      <Progress
                        value={(stats.gpuMemory.used / stats.gpuMemory.total) * 100}
                        className={`mt-2 ${getProgressColor((stats.gpuMemory.used / stats.gpuMemory.total) * 100)}`}
                      />
                    </>
                  ) : (
                    <div className="text-sm text-muted-foreground">未检测到 GPU</div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">{t('monitor.memoryUsage')}</CardTitle>
                  <Server className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {formatBytes(stats?.memoryUsage?.used ?? 0)} / {formatBytes(stats?.memoryUsage?.total ?? 0)}
                  </div>
                  <Progress
                    value={stats?.memoryUsage?.total ? ((stats.memoryUsage.used / stats.memoryUsage.total) * 100) : 0}
                    className={`mt-2 ${getProgressColor(stats?.memoryUsage?.total ? ((stats.memoryUsage.used / stats.memoryUsage.total) * 100) : 0)}`}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">{t('monitor.diskSpace')}</CardTitle>
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {stats?.diskUsage?.used ?? 0}GB / {stats?.diskUsage?.total ?? 0}GB
                  </div>
                  <Progress
                    value={stats?.diskUsage?.total ? ((stats.diskUsage.used / stats.diskUsage.total) * 100) : 0}
                    className={`mt-2 ${getProgressColor(stats?.diskUsage?.total ? ((stats.diskUsage.used / stats.diskUsage.total) * 100) : 0)}`}
                  />
                </CardContent>
              </Card>
            </>
          )}
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                {t('monitor.services')}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {loading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-4 w-12" />
                  </div>
                ))
              ) : (
                services.map((service, index) => {
                  const statusInfo = getStatusText(service.status);
                  return (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`h-2 w-2 rounded-full ${getStatusColor(service.status)}`} />
                        <span>{service.name}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        {service.uptime && service.uptime !== '-' && (
                          <span className="text-xs text-muted-foreground">{service.uptime}</span>
                        )}
                        <span className={`text-sm ${statusInfo.className}`}>{statusInfo.text}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wifi className="h-5 w-5" />
                {t('monitor.overview')}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {loading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-4 w-16" />
                  </div>
                ))
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('monitor.todayReplies')}</span>
                    <span className="text-sm font-medium">{stats?.todayReplies ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('monitor.avgResponseTime')}</span>
                    <span className="text-sm font-medium">{stats?.avgResponseTime ?? 0}s</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('monitor.activeSessions')}</span>
                    <span className="text-sm font-medium">{stats?.activeSessions ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('monitor.modelLoad')}</span>
                    <span className="text-sm font-medium">{stats?.modelLoad ?? 0}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('monitor.cpuUsage')}</span>
                    <span className="text-sm font-medium">{stats?.cpuUsage ?? 0}%</span>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AppLayout>
  );
}
