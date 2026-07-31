'use client';

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
import { Input } from '@/components/ui/input';
import {
  type DialogueConversation,
  type SavedDialogueItem,
} from '@/lib/api';
import { Eye, FileText, RefreshCw, Save, Search, Trash2 } from 'lucide-react';

interface SavedDialoguesPanelProps {
  items: SavedDialogueItem[];
  loading: boolean;
  searchQuery: string;
  previewId: number | null;
  previewDialogues: DialogueConversation[];
  previewLoading: boolean;
  onSearchQueryChange: (value: string) => void;
  onRefresh: () => void;
  onPreview: (id: number) => void;
  onCreateDataset: (id: number) => void;
  onDeleteSaved: (id: number) => void;
  onClosePreview: () => void;
  onDeletePreviewDialogue: (savedId: number, dialogueIndex: number) => void;
}

export function SavedDialoguesPanel({
  items,
  loading,
  searchQuery,
  previewId,
  previewDialogues,
  previewLoading,
  onSearchQueryChange,
  onRefresh,
  onPreview,
  onCreateDataset,
  onDeleteSaved,
  onClosePreview,
  onDeletePreviewDialogue,
}: SavedDialoguesPanelProps) {
  const normalizedQuery = searchQuery.trim().toLowerCase();
  const filteredItems = normalizedQuery
    ? items.filter(
        (item) =>
          item.name.toLowerCase().includes(normalizedQuery) ||
          item.character_desc.toLowerCase().includes(normalizedQuery),
      )
    : items;

  return (
    <>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Save className="h-5 w-5" />
                已保存的对话数据
              </CardTitle>
              <CardDescription>
                所有已生成的对话数据集，可预览、导出或创建训练数据
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="搜索已保存对话..."
                  value={searchQuery}
                  onChange={(event) => onSearchQueryChange(event.target.value)}
                  className="pl-8 h-9"
                />
              </div>
              <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
                刷新
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {items.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Save className="h-12 w-12 mx-auto mb-3 opacity-30" />
              <p className="text-lg">暂无保存的对话数据</p>
              <p className="text-sm mt-1">生成对话后点击保存，或开启自动保存模式</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {filteredItems.map((item) => (
                <Card
                  key={item.id}
                  className={`cursor-pointer transition-colors ${
                    previewId === item.id ? 'ring-2 ring-primary' : 'hover:bg-accent/50'
                  }`}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm truncate">{item.name}</CardTitle>
                    <CardDescription className="text-xs">
                      {item.character_desc?.slice(0, 50) || ''}
                      {(item.character_desc?.length || 0) > 50 ? '...' : ''}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pb-2 text-xs space-y-1">
                    <div className="flex justify-between text-muted-foreground">
                      <span>{item.dialogue_count} 组对话</span>
                      <span>{item.style || '默认风格'}</span>
                    </div>
                    <div className="text-muted-foreground/70">
                      保存于: {new Date(item.created_at).toLocaleString('zh-CN')}
                    </div>
                  </CardContent>
                  <CardFooter className="pt-0 gap-1 flex-wrap">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => onPreview(item.id)}
                    >
                      <Eye className="h-3 w-3 mr-1" />
                      预览
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => onCreateDataset(item.id)}
                    >
                      <FileText className="h-3 w-3 mr-1" />
                      创建数据集
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs text-destructive hover:text-destructive"
                      onClick={() => onDeleteSaved(item.id)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {previewId !== null && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">对话预览</CardTitle>
              <Button variant="ghost" size="sm" onClick={onClosePreview}>
                关闭
              </Button>
            </div>
          </CardHeader>
          <CardContent className="max-h-[500px] overflow-y-auto space-y-3">
            {previewLoading ? (
              <div className="text-center py-8 text-muted-foreground">加载中...</div>
            ) : previewDialogues.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">无对话数据</div>
            ) : (
              previewDialogues.map((dialogue, dialogueIndex) => (
                <div
                  key={dialogueIndex}
                  className="border rounded-lg p-3 text-sm relative group"
                >
                  <button
                    type="button"
                    onClick={() => onDeletePreviewDialogue(previewId, dialogueIndex)}
                    className="absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
                    title="删除此对话"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                  <div className="flex items-center gap-2 mb-2 pr-6">
                    <Badge variant="outline" className="text-xs">
                      {dialogue.scene || `对话 ${dialogueIndex + 1}`}
                    </Badge>
                    <Badge variant="secondary" className="text-xs">
                      {Math.ceil((dialogue.conversations?.length || 0) / 2)} 轮
                    </Badge>
                    {dialogue.tags?.map((tag, tagIndex) => (
                      <Badge key={tagIndex} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <div className="space-y-2 max-h-[200px] overflow-y-auto">
                    {dialogue.conversations?.map((conversation, conversationIndex) => (
                      <div key={conversationIndex} className="flex gap-2">
                        <span
                          className={`shrink-0 text-xs font-medium mt-0.5 ${
                            conversation.from === 'human'
                              ? 'text-blue-600'
                              : 'text-green-600'
                          }`}
                        >
                          {conversation.from === 'human' ? 'Q' : 'A'}:
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {conversation.value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      )}
    </>
  );
}
