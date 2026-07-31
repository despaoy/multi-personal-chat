'use client';

import {
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Pause,
  Plus,
  XCircle,
  Zap,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import type { Dataset, TrainingTask } from '@/lib/api';
import {
  TrainingParamsEditor,
  type TrainingParamsEditorProps,
} from './TrainingParamsEditor';

const statusColors: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  training: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-100 text-gray-800',
};

const statusIcons: Record<string, typeof Clock> = {
  pending: Clock,
  training: Zap,
  completed: CheckCircle2,
  failed: XCircle,
  cancelled: XCircle,
};

interface TrainingTasksPanelProps {
  datasets: Dataset[];
  tasks: TrainingTask[];
  loading: boolean;
  dialogOpen: boolean;
  submitting: boolean;
  onDialogOpenChange: (open: boolean) => void;
  onStartTraining: TrainingParamsEditorProps['onSubmit'];
  onCancelTraining: (taskId: string) => Promise<void>;
}

export function TrainingTasksPanel({
  datasets,
  tasks,
  loading,
  dialogOpen,
  submitting,
  onDialogOpenChange,
  onStartTraining,
  onCancelTraining,
}: TrainingTasksPanelProps) {
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Dialog open={dialogOpen} onOpenChange={onDialogOpenChange}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              新建训练
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[760px] max-h-[92vh] flex flex-col">
            <DialogHeader>
              <DialogTitle>启动 LoRA 训练</DialogTitle>
              <DialogDescription>
                配置训练参数并启动训练任务；高级设置可一键应用显存预设。
              </DialogDescription>
            </DialogHeader>
            <div className="overflow-y-auto flex-1 min-h-0 pr-1">
              <TrainingParamsEditor
                datasets={datasets.map((dataset) => ({
                  name: dataset.name,
                  sampleCount: dataset.stats?.total ?? 0,
                }))}
                submitting={submitting}
                onSubmit={onStartTraining}
              />
            </div>
            <DialogFooter className="shrink-0">
              <Button variant="secondary" onClick={() => onDialogOpenChange(false)}>
                关闭
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((index) => (
            <Card key={index}>
              <CardHeader>
                <Skeleton className="h-6 w-1/2" />
              </CardHeader>
              <CardContent className="space-y-4">
                <Skeleton className="h-2 w-full" />
                <Skeleton className="h-4 w-1/3" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : tasks.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <Zap className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">暂无训练任务</h3>
              <p className="text-muted-foreground mb-4">
                启动您的第一个 LoRA 训练任务
              </p>
              <Button onClick={() => onDialogOpenChange(true)}>
                <Plus className="mr-2 h-4 w-4" />
                新建训练
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {tasks.map((task) => {
            const StatusIcon = statusIcons[task.status] || Clock;
            return (
              <Card key={task.task_id}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BrainCircuit className="h-5 w-5 text-muted-foreground" />
                    {task.lora_name}
                    <Badge className={statusColors[task.status]}>
                      <StatusIcon className="h-3 w-3 mr-1" />
                      {task.status}
                    </Badge>
                  </CardTitle>
                  <CardDescription>任务ID: {task.task_id}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">训练进度</span>
                      <span>{task.progress}%</span>
                    </div>
                    <Progress value={task.progress} className="h-2" />
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">创建时间</span>
                    <span>{new Date(task.created_at).toLocaleString()}</span>
                  </div>
                  {task.error_message && (
                    <div className="bg-red-50 text-red-800 p-3 rounded-lg text-sm">
                      <AlertCircle className="h-4 w-4 inline mr-2" />
                      {task.error_message}
                    </div>
                  )}
                </CardContent>
                {task.status === 'training' && (
                  <CardFooter className="border-t pt-4">
                    <Button
                      variant="destructive"
                      className="w-full"
                      onClick={() => void onCancelTraining(task.task_id)}
                    >
                      <Pause className="mr-2 h-4 w-4" />
                      取消训练
                    </Button>
                  </CardFooter>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
