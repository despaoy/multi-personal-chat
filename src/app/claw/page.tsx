'use client';

import { useState, useEffect, useCallback } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { Plus, Edit3, Trash2, Play, Terminal, Wrench, Shield, RefreshCw, Code } from 'lucide-react';

interface ClawTool {
  name: string;
  description: string;
  code: string;
  enabled: boolean;
  builtin: boolean;
  created_at?: string;
  updated_at?: string;
}

export default function ClawPage() {
  return (
    <AuthGuard requireAdmin>
      <ClawContent />
    </AuthGuard>
  );
}

function ClawContent() {
  const [tools, setTools] = useState<ClawTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ output: string; error: string; result: string; success: boolean } | null>(null);
  const [testRunning, setTestRunning] = useState(false);

  // 编辑表单
  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [editCode, setEditCode] = useState('');
  const [editEnabled, setEditEnabled] = useState(true);
  const [editingExisting, setEditingExisting] = useState(false);

  const loadTools = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listClawTools();
      setTools(data.tools);
    } catch {
      toast.error('加载工具列表失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadTools(); }, [loadTools]);

  const openNewTool = () => {
    setEditName('');
    setEditDesc('');
    setEditCode('');
    setEditEnabled(true);
    setEditingExisting(false);
    setEditDialogOpen(true);
  };

  const openEditTool = (tool: ClawTool) => {
    setEditName(tool.name);
    setEditDesc(tool.description);
    setEditCode(tool.code);
    setEditEnabled(tool.enabled);
    setEditingExisting(true);
    setEditDialogOpen(true);
  };

  const handleSave = async () => {
    if (!editName.trim()) { toast.error('请输入工具名称'); return; }
    try {
      await api.saveClawTool({ name: editName.trim(), description: editDesc.trim(), code: editCode, enabled: editEnabled });
      toast.success(editingExisting ? '工具已更新' : '工具已创建');
      setEditDialogOpen(false);
      loadTools();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '保存失败');
    }
  };

  const handleDelete = async (name: string) => {
    try {
      await api.deleteClawTool(name);
      toast.success('工具已删除');
      setDeleteConfirm(null);
      loadTools();
    } catch {
      toast.error('删除失败');
    }
  };

  const handleTest = async () => {
    setTestRunning(true);
    setTestResult(null);
    try {
      const result = await api.executeClawTool(editCode);
      setTestResult(result);
    } catch (err) {
      setTestResult({ success: false, output: '', error: String(err), result: '' });
    } finally {
      setTestRunning(false);
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Claw 工具管理</h2>
            <p className="text-muted-foreground">管理 QQ 机器人 /claw 模式下的自定义工具</p>
          </div>
          <Button onClick={openNewTool}>
            <Plus className="mr-2 h-4 w-4" />
            新建工具
          </Button>
        </div>

        {/* 说明卡片 */}
        <Card className="bg-muted/30">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <Terminal className="h-5 w-5 text-muted-foreground mt-0.5" />
              <div className="text-sm text-muted-foreground space-y-1">
                <p>在 QQ 私聊中发送 <code className="px-1.5 py-0.5 rounded bg-muted text-xs font-mono">/claw</code> 进入 Claw 操作模式。</p>
                <p>Claw 模式下，你可以用自然语言描述需求，胡桃会自动选择合适的工具执行。</p>
                <p>下方可以编写自定义 Python 工具，工具会通过 <code className="px-1.5 py-0.5 rounded bg-muted text-xs font-mono">exec()</code> 动态加载执行。</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 工具列表 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wrench className="h-5 w-5" />
              工具列表
              <span className="text-sm font-normal text-muted-foreground ml-2">
                {tools.length} 个工具
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1,2,3].map(i => <Skeleton key={i} className="h-20 w-full" />)}
              </div>
            ) : tools.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">暂无工具</div>
            ) : (
              <div className="space-y-3">
                {tools.map(tool => (
                  <div
                    key={tool.name}
                    className="flex items-start justify-between p-4 border rounded-lg hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex-1 min-w-0 mr-4">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono font-medium">{tool.name}</span>
                        {tool.builtin ? (
                          <Badge variant="secondary" className="text-xs">
                            <Shield className="h-3 w-3 mr-1" />内置
                          </Badge>
                        ) : tool.enabled ? (
                          <Badge variant="default" className="text-xs bg-green-600">启用</Badge>
                        ) : (
                          <Badge variant="secondary" className="text-xs">禁用</Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">{tool.description}</p>
                      {tool.code && (
                        <p className="text-xs text-muted-foreground mt-1 font-mono truncate">
                          <Code className="h-3 w-3 inline mr-1" />
                          {tool.code.split('\n')[0]}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {!tool.builtin && (
                        <>
                          <Button variant="ghost" size="icon" onClick={() => openEditTool(tool)} title="编辑">
                            <Edit3 className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" onClick={() => setDeleteConfirm(tool.name)} title="删除">
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* 编辑/新建工具对话框 */}
        <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
          <DialogContent className="sm:max-w-[700px] max-h-[90vh] flex flex-col">
            <DialogHeader>
              <DialogTitle>{editingExisting ? '编辑工具' : '新建工具'}</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 overflow-y-auto flex-1 min-h-0 py-2">
              <div className="space-y-2">
                <Label htmlFor="tool-name">工具名称（英文，唯一标识）</Label>
                <Input
                  id="tool-name"
                  placeholder="例如：clean_temp_files"
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  disabled={editingExisting}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tool-desc">功能描述</Label>
                <Input
                  id="tool-desc"
                  placeholder="描述工具的功能，LLM 根据此描述选择工具"
                  value={editDesc}
                  onChange={e => setEditDesc(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="tool-code">Python 代码</Label>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleTest}
                      disabled={testRunning || !editCode.trim()}
                    >
                      {testRunning ? (
                        <RefreshCw className="mr-1 h-3 w-3 animate-spin" />
                      ) : (
                        <Play className="mr-1 h-3 w-3" />
                      )}
                      测试运行
                    </Button>
                  </div>
                </div>
                <Textarea
                  id="tool-code"
                  placeholder={`# 编写 Python 函数体，可通过 args 字典获取参数\n# 例如：filename = args.get("filename", "default.txt")\n\nimport os\npath = args.get("path", "/tmp")\nfiles = os.listdir(path)\nreturn "\\n".join(files)`}
                  className="min-h-[250px] font-mono text-sm"
                  value={editCode}
                  onChange={e => setEditCode(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  代码作为函数体执行，参数通过 <code className="px-1 rounded bg-muted text-xs">args</code> 字典传入，
                  用 <code className="px-1 rounded bg-muted text-xs">return</code> 返回结果。
                </p>
              </div>
              <div className="flex items-center gap-3">
                <Switch checked={editEnabled} onCheckedChange={setEditEnabled} id="tool-enabled" />
                <Label htmlFor="tool-enabled" className="cursor-pointer">启用此工具</Label>
              </div>
            </div>

            {/* 测试结果 */}
            {testResult && (
              <>
                <Separator />
                <div className="space-y-2">
                  <Label className="flex items-center gap-2">
                    <Terminal className="h-4 w-4" />
                    测试结果
                    <Badge variant={testResult.success ? 'default' : 'destructive'} className="text-xs ml-1">
                      {testResult.success ? '成功' : '失败'}
                    </Badge>
                  </Label>
                  <ScrollArea className="max-h-[200px]">
                    <pre className="text-xs bg-muted rounded-md p-3 whitespace-pre-wrap break-all">
                      {testResult.output && <span className="text-foreground">{testResult.output}</span>}
                      {testResult.error && <span className="text-destructive">{testResult.error}</span>}
                      {!testResult.output && !testResult.error && (
                        <span className="text-muted-foreground">(无输出)</span>
                      )}
                    </pre>
                  </ScrollArea>
                </div>
              </>
            )}

            <DialogFooter>
              <Button variant="outline" onClick={() => setEditDialogOpen(false)}>取消</Button>
              <Button onClick={handleSave}>
                {editingExisting ? '更新' : '创建'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 删除确认 */}
        <Dialog open={!!deleteConfirm} onOpenChange={() => setDeleteConfirm(null)}>
          <DialogContent className="sm:max-w-[400px]">
            <DialogHeader><DialogTitle>确认删除</DialogTitle></DialogHeader>
            <p className="text-muted-foreground">
              确定要删除工具 <code className="px-1 rounded bg-muted text-sm font-mono">{deleteConfirm}</code> 吗？
            </p>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteConfirm(null)}>取消</Button>
              <Button variant="destructive" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>删除</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AppLayout>
  );
}
