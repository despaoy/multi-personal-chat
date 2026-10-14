'use client';

import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { AlertCircle, CheckCircle2, RefreshCw, ShieldCheck, XCircle } from 'lucide-react';
import { useAdapterChecks } from '@/hooks/useRouter';
import { toast } from 'sonner';

export default function AdapterChecksPage() {
  return <AuthGuard requireAdmin><AdapterChecksContent /></AuthGuard>;
}

function AdapterChecksContent() {
  const { adapters, loading, error, checking, refetch, checkAdapter } = useAdapterChecks();
  const handleCheck = async (name: string) => {
    try {
      await checkAdapter(name);
      toast.success('适配器检查结果已保存');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '适配器检查失败');
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <ShieldCheck className="h-6 w-6" />适配器检查
            </h2>
            <p className="text-muted-foreground">检查适配器是否适合当前模型，角色在聊天时明确选择。</p>
          </div>
          <Button variant="ghost" size="icon" aria-label="刷新适配器列表" onClick={() => void refetch()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>适配器列表</CardTitle>
            <CardDescription>只有完成检查并保存的结果才显示兼容性状态。</CardDescription>
          </CardHeader>
          <CardContent>
            {error ? (
              <div role="alert" className="flex flex-col items-center gap-3 py-8">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <p className="text-destructive">{error}</p>
                <Button onClick={() => void refetch()} disabled={loading}>重试</Button>
              </div>
            ) : loading ? <Skeleton className="h-48" /> : adapters.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">暂无适配器</p>
            ) : (
              <Table>
                <TableHeader><TableRow><TableHead>名称</TableHead><TableHead>兼容性</TableHead><TableHead>操作</TableHead></TableRow></TableHeader>
                <TableBody>
                  {adapters.map((adapter) => (
                    <TableRow key={adapter.name}>
                      <TableCell className="font-mono text-xs">{adapter.name}</TableCell>
                      <TableCell>
                        {adapter.compatibility === null ? <Badge variant="outline">未检查</Badge> : adapter.compatibility.compatible ? (
                          <Badge className="bg-green-100 text-green-800"><CheckCircle2 className="mr-1 h-3 w-3" />兼容</Badge>
                        ) : <Badge className="bg-red-100 text-red-800"><XCircle className="mr-1 h-3 w-3" />不兼容</Badge>}
                      </TableCell>
                      <TableCell>
                        <Button variant="ghost" size="sm" onClick={() => void handleCheck(adapter.name)} disabled={checking !== null}>
                          <ShieldCheck className="mr-1 h-3 w-3" />{checking === adapter.name ? '检查中...' : '检查'}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
