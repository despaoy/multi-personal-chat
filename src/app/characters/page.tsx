'use client';

import { useCallback, useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api, type CharacterMemoryRecord, type CharacterMemoryScope, type CharacterRelationshipRecord } from '@/lib/api';
import { RefreshCw, Trash2, AlertCircle, Users, Heart, Pencil } from 'lucide-react';
import { toast } from 'sonner';

const STAGE_LABELS: Record<string, string> = {
  stranger: '陌生',
  acquaintance: '认识',
  familiar: '熟悉',
  close: '亲近',
};

const MEMORY_TYPE_LABELS: Record<string, string> = {
  user_fact: '用户事实',
  shared_event: '共同经历',
  promise: '承诺约定',
  conversation_summary: '对话摘要',
};

const DEFAULT_SCOPE: CharacterMemoryScope = {
  platform: 'qq',
  adapter: 'nonebot',
  sender_id: '',
  conversation_type: 'private',
  conversation_id: '',
};

export default function CharactersPage() {
  return (
    <AuthGuard requireAdmin>
      <AppLayout>
        <CharactersContent />
      </AppLayout>
    </AuthGuard>
  );
}

function CharactersContent() {
  const [characters, setCharacters] = useState<Array<{ character_id: string; display_name: string; version: string }>>([]);
  const [selectedCharacter, setSelectedCharacter] = useState('');
  const [scope, setScope] = useState<CharacterMemoryScope>(DEFAULT_SCOPE);
  const [scopeApplied, setScopeApplied] = useState<CharacterMemoryScope | null>(null);
  const [relationship, setRelationship] = useState<CharacterRelationshipRecord | null>(null);
  const [memories, setMemories] = useState<CharacterMemoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [stageEdit, setStageEdit] = useState('stranger');
  const [savingStage, setSavingStage] = useState(false);
  const [editingMemoryId, setEditingMemoryId] = useState<number | null>(null);
  const [memoryContentDraft, setMemoryContentDraft] = useState('');
  const [memoryImportanceDraft, setMemoryImportanceDraft] = useState('0.5');
  const [savingMemory, setSavingMemory] = useState(false);

  const loadCharacters = useCallback(async () => {
    try {
      const response = await api.listCharacters();
      setCharacters(response.characters);
      if (response.characters.length > 0 && !selectedCharacter) {
        setSelectedCharacter(response.characters[0].character_id);
      }
    } catch (error) {
      toast.error('加载人物画像失败');
      console.error(error);
    }
  }, [selectedCharacter]);

  useEffect(() => {
    loadCharacters().finally(() => setLoading(false));
  }, [loadCharacters]);

  const loadScopeData = useCallback(async () => {
    if (!selectedCharacter || !scopeApplied) return;
    // 先清空上一个范围的数据：新范围加载失败时绝不能继续展示
    // 旧用户的关系与记忆（数据串号风险）
    setRelationship(null);
    setStageEdit('stranger');
    setMemories([]);
    // 同步放弃编辑中的草稿，避免保存到新范围的记忆上
    setEditingMemoryId(null);
    setMemoryContentDraft('');
    setMemoryImportanceDraft('0.5');
    try {
      const [relRes, memRes] = await Promise.all([
        api.getCharacterRelationship(selectedCharacter, scopeApplied),
        api.listCharacterMemories(selectedCharacter, scopeApplied),
      ]);
      setRelationship(relRes.relationship);
      setStageEdit(relRes.relationship?.relationship_stage || 'stranger');
      setMemories(memRes.memories);
    } catch (error) {
      toast.error('加载角色数据失败');
      console.error(error);
    }
  }, [selectedCharacter, scopeApplied]);

  useEffect(() => {
    loadScopeData();
  }, [loadScopeData]);

  const handleApplyScope = () => {
    if (!scope.sender_id.trim()) {
      toast.warning('请填写用户 ID（sender_id）');
      return;
    }
    const normalized: CharacterMemoryScope = {
      ...scope,
      sender_id: scope.sender_id.trim(),
      conversation_id:
        scope.conversation_type === 'private'
          ? scope.sender_id.trim()
          : scope.conversation_id.trim(),
    };
    if (scope.conversation_type !== 'private' && !normalized.conversation_id) {
      toast.warning('群聊/频道需要填写会话（群/频道）ID');
      return;
    }
    setScopeApplied(normalized);
  };

  const handleSaveStage = async () => {
    if (!selectedCharacter || !scopeApplied) return;
    setSavingStage(true);
    try {
      await api.updateCharacterRelationship(selectedCharacter, scopeApplied, {
        stage: stageEdit,
        preferred_address: relationship?.preferred_address || '',
        summary: relationship?.summary || '',
      });
      toast.success('关系阶段已更新');
      await loadScopeData();
    } catch (error) {
      toast.error('更新关系失败');
      console.error(error);
    } finally {
      setSavingStage(false);
    }
  };

  const handleStartEditMemory = (memory: CharacterMemoryRecord) => {
    setEditingMemoryId(memory.id);
    setMemoryContentDraft(memory.content);
    setMemoryImportanceDraft(memory.importance.toFixed(1));
  };

  const handleCancelEditMemory = () => {
    setEditingMemoryId(null);
    setMemoryContentDraft('');
    setMemoryImportanceDraft('0.5');
  };

  const handleSaveMemory = async () => {
    if (!selectedCharacter || !scopeApplied || editingMemoryId === null) return;
    const content = memoryContentDraft.trim();
    const importance = Number.parseFloat(memoryImportanceDraft);
    if (!content) {
      toast.warning('记忆内容不能为空');
      return;
    }
    if (content.length > 500) {
      toast.warning('记忆内容最多 500 字符');
      return;
    }
    if (Number.isNaN(importance) || importance < 0 || importance > 1) {
      toast.warning('重要度需在 0 到 1 之间');
      return;
    }
    setSavingMemory(true);
    try {
      await api.updateCharacterMemory(selectedCharacter, editingMemoryId, scopeApplied, {
        content,
        importance,
      });
      toast.success('记忆已更新');
      handleCancelEditMemory();
      await loadScopeData();
    } catch (error) {
      toast.error('更新记忆失败');
      console.error(error);
    } finally {
      setSavingMemory(false);
    }
  };

  const handleDeleteMemory = async (memoryId: number) => {
    if (!selectedCharacter || !scopeApplied) return;
    if (!confirm('确定要删除这条记忆吗？')) return;
    try {
      await api.deleteCharacterMemory(selectedCharacter, memoryId, scopeApplied);
      toast.success('记忆已删除');
      await loadScopeData();
    } catch (error) {
      toast.error('删除记忆失败');
      console.error(error);
    }
  };

  const handleClearMemories = async () => {
    if (!selectedCharacter || !scopeApplied) return;
    if (!confirm('确定要清空该用户在此角色下的全部记忆吗？此操作不可恢复。')) return;
    try {
      const result = await api.clearCharacterMemories(selectedCharacter, scopeApplied);
      toast.success(result.message);
      await loadScopeData();
    } catch (error) {
      toast.error('清空记忆失败');
      console.error(error);
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">角色与记忆管理</h1>
        <p className="text-muted-foreground mt-1">
          查看和管理角色长期记忆与关系状态。记忆按「平台 + 适配器 + 会话 + 用户」严格隔离。
        </p>
      </div>

      {characters.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 space-y-3">
            <Users className="h-12 w-12 text-muted-foreground" />
            <div className="text-center">
              <h3 className="text-lg font-semibold">暂无已注册人物画像</h3>
              <p className="text-sm text-muted-foreground">
                请在 backend/data/character_profiles/ 下部署画像 JSON 并在 LoRA 注册表中映射
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>查询范围</CardTitle>
              <CardDescription>
                选择角色并填写用户范围。私聊按用户隔离；群聊按「群 + 用户」隔离。
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="character">角色</Label>
                  <Select value={selectedCharacter} onValueChange={setSelectedCharacter}>
                    <SelectTrigger id="character">
                      <SelectValue placeholder="选择角色" />
                    </SelectTrigger>
                    <SelectContent>
                      {characters.map((c) => (
                        <SelectItem key={c.character_id} value={c.character_id}>
                          {c.display_name}（{c.character_id}）
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="conversation_type">会话类型</Label>
                  <Select
                    value={scope.conversation_type}
                    onValueChange={(value) =>
                      setScope((prev) => ({ ...prev, conversation_type: value }))
                    }
                  >
                    <SelectTrigger id="conversation_type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="private">私聊</SelectItem>
                      <SelectItem value="group">群聊</SelectItem>
                      <SelectItem value="channel">频道</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="platform">平台</Label>
                  <Input
                    id="platform"
                    value={scope.platform}
                    onChange={(e) => setScope((prev) => ({ ...prev, platform: e.target.value }))}
                    placeholder="qq"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="adapter">适配器</Label>
                  <Input
                    id="adapter"
                    value={scope.adapter}
                    onChange={(e) => setScope((prev) => ({ ...prev, adapter: e.target.value }))}
                    placeholder="nonebot"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="sender_id">用户 ID</Label>
                  <Input
                    id="sender_id"
                    value={scope.sender_id}
                    onChange={(e) => setScope((prev) => ({ ...prev, sender_id: e.target.value }))}
                    placeholder="发送者 QQ 号"
                  />
                </div>
                {scope.conversation_type !== 'private' && (
                  <div className="space-y-2">
                    <Label htmlFor="conversation_id">会话（群/频道）ID</Label>
                    <Input
                      id="conversation_id"
                      value={scope.conversation_id}
                      onChange={(e) =>
                        setScope((prev) => ({ ...prev, conversation_id: e.target.value }))
                      }
                      placeholder="群号或频道 ID"
                    />
                  </div>
                )}
              </div>
              <Button onClick={handleApplyScope}>
                <RefreshCw className="mr-2 h-4 w-4" />
                查询
              </Button>
            </CardContent>
          </Card>

          {scopeApplied && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Heart className="h-5 w-5" />
                    关系状态
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {relationship ? (
                    <div className="space-y-4">
                      <div className="flex flex-wrap items-center gap-4 text-sm">
                        <Badge variant="secondary">
                          {STAGE_LABELS[relationship.relationship_stage] || relationship.relationship_stage}
                        </Badge>
                        <span className="text-muted-foreground">
                          交互轮数：{relationship.interaction_count}
                        </span>
                        {relationship.preferred_address && (
                          <span className="text-muted-foreground">
                            称呼偏好：{relationship.preferred_address}
                          </span>
                        )}
                      </div>
                      {relationship.summary && (
                        <p className="text-sm text-muted-foreground">{relationship.summary}</p>
                      )}
                      <div className="flex items-center gap-3">
                        <Select value={stageEdit} onValueChange={setStageEdit}>
                          <SelectTrigger className="w-40">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(STAGE_LABELS).map(([value, label]) => (
                              <SelectItem key={value} value={value}>
                                {label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button onClick={handleSaveStage} disabled={savingStage}>
                          {savingStage ? '保存中…' : '手动覆盖关系阶段'}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <AlertCircle className="h-4 w-4" />
                      该范围尚无关系记录（用户未与此角色对话过）
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0">
                  <div>
                    <CardTitle>长期记忆（{memories.length} 条）</CardTitle>
                    <CardDescription>
                      仅展示最近 100 条。记忆在生成时按相关度 + 重要性 + 新近度选取前 5 条注入参考区。
                    </CardDescription>
                  </div>
                  {memories.length > 0 && (
                    <Button variant="destructive" size="sm" onClick={handleClearMemories}>
                      <Trash2 className="mr-2 h-4 w-4" />
                      清空全部
                    </Button>
                  )}
                </CardHeader>
                <CardContent>
                  {memories.length === 0 ? (
                    <p className="text-sm text-muted-foreground py-8 text-center">
                      暂无记忆记录
                    </p>
                  ) : (
                    <ul className="space-y-2">
                      {memories.map((memory) => (
                        <li
                          key={memory.id}
                          className="rounded-lg border p-3"
                        >
                          {editingMemoryId === memory.id ? (
                            <div className="space-y-3">
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge variant="outline">
                                  {MEMORY_TYPE_LABELS[memory.memory_type] || memory.memory_type}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {memory.memory_key}
                                </span>
                              </div>
                              <div className="space-y-1">
                                <Label htmlFor={`memory-content-${memory.id}`}>记忆内容</Label>
                                <textarea
                                  id={`memory-content-${memory.id}`}
                                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                  value={memoryContentDraft}
                                  onChange={(e) => setMemoryContentDraft(e.target.value)}
                                  maxLength={500}
                                />
                              </div>
                              <div className="flex flex-wrap items-end gap-3">
                                <div className="w-36 space-y-1">
                                  <Label htmlFor={`memory-importance-${memory.id}`}>
                                    重要度（0-1）
                                  </Label>
                                  <Input
                                    id={`memory-importance-${memory.id}`}
                                    type="number"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={memoryImportanceDraft}
                                    onChange={(e) => setMemoryImportanceDraft(e.target.value)}
                                  />
                                </div>
                                <Button
                                  size="sm"
                                  onClick={handleSaveMemory}
                                  disabled={savingMemory}
                                >
                                  {savingMemory ? '保存中…' : '保存'}
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={handleCancelEditMemory}
                                  disabled={savingMemory}
                                >
                                  取消
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <div className="flex items-start justify-between gap-4">
                              <div className="min-w-0 space-y-1">
                                <div className="flex flex-wrap items-center gap-2">
                                  <Badge variant="outline">
                                    {MEMORY_TYPE_LABELS[memory.memory_type] || memory.memory_type}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {memory.memory_key}
                                  </span>
                                </div>
                                <p className="text-sm break-words">{memory.content}</p>
                                <p className="text-xs text-muted-foreground">
                                  重要度 {memory.importance.toFixed(1)} · 更新于{' '}
                                  {formatTimestamp(memory.updated_at)}
                                </p>
                              </div>
                              <div className="flex shrink-0 gap-1">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleStartEditMemory(memory)}
                                  disabled={editingMemoryId !== null}
                                >
                                  <Pencil className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleDeleteMemory(memory.id)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}
