'use client';

import { useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { BrainCircuit, Trash2, RefreshCw, AlertCircle, FolderSearch } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { useLoras } from '@/hooks/useLoras';
import { api } from '@/lib/api';
import { toast } from 'sonner';

export default function LoraPage() {
  return (
    <AuthGuard requireAdmin>
      <LoraContent />
    </AuthGuard>
  );
}

function LoraContent() {
  const [activeTab, setActiveTab] = useState('all');
  const [scanning, setScanning] = useState(false);
  const { loras, loading, error, refetch, toggleLoraStatus, deleteLora } = useLoras();

  const handleScanLoras = async () => {
    setScanning(true);
    try {
      const result = await api.scanLoras();
      if (result.new_count > 0 || result.updated_count > 0) {
        await refetch();
      }
      if (result.success) {
        toast.success(result.message);
      } else {
        toast.warning(result.message);
      }
    } catch {
      toast.error('扫描LoRA失败');
    } finally {
      setScanning(false);
    }
  };

  // 过滤LoRA模型：仅显示用户自己训练的无风格LoRA
  const filteredLoras = loras.filter(lora => {
    if (activeTab === 'all') return true;
    if (activeTab === 'active') return lora.status === 'active';
    if (activeTab === 'inactive') return lora.status === 'inactive';
    return true;
  });

  const handleToggleStatus = async (id: string) => {
    try {
      await toggleLoraStatus(id);
      toast.success('LoRA已切换（同一时间只能启用一个）');
    } catch (err) {
      console.error('Failed to toggle lora status:', err);
      toast.error('切换LoRA状态失败');
    }
  };

  const handleDeleteLora = async (id: string) => {
    if (!confirm('确定要删除这个LoRA模型吗？此操作不可恢复。')) {
      return;
    }
    
    try {
      await deleteLora(id);
      toast.success('LoRA模型已删除');
    } catch (err) {
      console.error('Failed to delete lora:', err);
      toast.error('删除LoRA模型失败');
    }
  };

  if (error) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">加载失败</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
          <Button onClick={refetch}>
            <RefreshCw className="mr-2 h-4 w-4" />
            重试
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
            <h2 className="text-2xl font-bold tracking-tight">LoRA 管理</h2>
            <p className="text-muted-foreground">管理您的LoRA模型 · 同一时间只能启用一个LoRA进行推理</p>
          </div>
          <Button variant="ghost" size="icon" onClick={refetch} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
          <Button variant="outline" size="sm" onClick={handleScanLoras} disabled={scanning}>
            <FolderSearch className="mr-2 h-4 w-4" />
            {scanning ? '扫描中...' : '扫描新LoRA'}
          </Button>
        </div>

        <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="all">全部</TabsTrigger>
            <TabsTrigger value="active">已启用</TabsTrigger>
            <TabsTrigger value="inactive">已停用</TabsTrigger>
          </TabsList>
          {/* 所有Tab共享同一内容区域，通过filteredLoras过滤 */}
          {["all", "active", "inactive"].map(tab => (
            <TabsContent key={tab} value={tab} className="mt-4">
              {loading ? (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {[1, 2, 3].map((i) => (
                    <Card key={i}>
                      <CardHeader>
                        <Skeleton className="h-6 w-3/4" />
                        <Skeleton className="h-4 w-full mt-2" />
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-2 w-full" />
                        <Skeleton className="h-4 w-full" />
                      </CardContent>
                      <CardFooter className="border-t pt-4">
                        <Skeleton className="h-8 w-full" />
                      </CardFooter>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {filteredLoras.length === 0 ? (
                    <div className="col-span-full text-center py-12">
                      <p className="text-muted-foreground">暂无 LoRA 模型</p>
                    </div>
                  ) : (
                    filteredLoras.map((lora) => (
                      <Card key={lora.id} className={lora.status === 'active' ? 'border-primary' : ''}>
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div>
                              <CardTitle className="flex items-center gap-2">
                                {lora.name}
                                {lora.status === 'active' && (
                                  <Badge variant="default">当前使用</Badge>
                                )}
                              </CardTitle>
                              <CardDescription>{lora.description}</CardDescription>
                            </div>
                            <BrainCircuit className="h-5 w-5 text-muted-foreground" />
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">风格类型</span>
                            <Badge variant="secondary">
                              {lora.style || '无风格'}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">模型大小</span>
                            <span>{lora.size}</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">训练进度</span>
                              <span>
                                {lora.totalSteps > 0
                                  ? lora.trainedSteps >= lora.totalSteps
                                    ? '训练完成'
                                    : `${lora.trainedSteps}/${lora.totalSteps}`
                                  : '已完成'}
                              </span>
                            </div>
                            <Progress
                              value={lora.totalSteps > 0 ? Math.min((lora.trainedSteps / lora.totalSteps) * 100, 100) : 100}
                              className="h-2"
                            />
                          </div>
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">创建时间</span>
                            <span>{lora.createdAt ? new Date(lora.createdAt + 'T00:00:00').toLocaleDateString('zh-CN') : '未知'}</span>
                          </div>
                        </CardContent>
                        <CardFooter className="border-t pt-4">
                          <div className="flex w-full items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Switch
                                checked={lora.status === 'active'}
                                onCheckedChange={() => handleToggleStatus(lora.id)}
                              />
                              <span className="text-sm">
                                {lora.status === 'active' ? '已启用' : '已停用'}
                              </span>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDeleteLora(lora.id)}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </CardFooter>
                      </Card>
                    ))
                  )}
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </AppLayout>
  );
}
