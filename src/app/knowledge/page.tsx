'use client';

import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger,
  DialogFooter
} from '@/components/ui/dialog';
import {
  Database,
  Upload,
  Plus,
  Search,
  FileText,
  FileSpreadsheet,
  Trash2,
  Edit,
  X,
  Loader2,
  FolderOpen,
  Folder,
  Archive,
  BookOpen,
  ScanSearch,
} from 'lucide-react';
import { useKnowledge } from '@/hooks/useKnowledge';
import { useState, useRef } from 'react';
import { toast } from 'sonner';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { type KnowledgeDocument, type ScanDirectory } from '@/lib/api';

export default function KnowledgePage() {
  return (
    <AuthGuard requireAdmin>
      <KnowledgeContent />
    </AuthGuard>
  );
}

function KnowledgeContent() {
  const { 
    bases,
    activeBaseId,
    selectBase,
    createBase,
    deleteBase,
    folders,
    activeFolderId,
    selectFolder,
    createFolder,
    deleteFolder,
    documents, 
    stats, 
    loading, 
    error, 
    searchResults, 
    searching,
    createDocument, 
    updateDocument, 
    deleteDocument, 
    searchKnowledge,
    clearSearchResults,
    uploadZip,
    scanResults,
    scanning,
    scanDirs,
    importDir,
  } = useKnowledge();

  const [searchQuery, setSearchQuery] = useState('');
  const [isCreateBaseDialogOpen, setIsCreateBaseDialogOpen] = useState(false);
  const [isCreateFolderDialogOpen, setIsCreateFolderDialogOpen] = useState(false);
  const [isCreateDocDialogOpen, setIsCreateDocDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingDoc, setEditingDoc] = useState<KnowledgeDocument | null>(null);
  const [newBaseName, setNewBaseName] = useState('');
  const [newFolderName, setNewFolderName] = useState('');
  const [newDoc, setNewDoc] = useState({ title: '', content: '' });
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [isScanDialogOpen, setIsScanDialogOpen] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeBase = bases.find(b => b.id === activeBaseId);
  const activeFolder = folders.find(f => f.id === activeFolderId);

  // ========== 知识库操作 ==========
  const handleCreateBase = async () => {
    if (!newBaseName.trim()) { toast.error('请输入知识库名称'); return; }
    try {
      setSubmitting(true);
      await createBase(newBaseName.trim());
      toast.success('知识库创建成功');
      setNewBaseName('');
      setIsCreateBaseDialogOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '创建失败');
    } finally { setSubmitting(false); }
  };

  const handleDeleteBase = async (kbId: number) => {
    if (!confirm('确定删除此知识库及其所有文件夹和文档？')) return;
    try {
      await deleteBase(kbId);
      toast.success('知识库已删除');
    } catch { toast.error('删除失败'); }
  };

  // ========== 文件夹操作 ==========
  const handleCreateFolder = async () => {
    if (!activeBaseId || !newFolderName.trim()) { toast.error('请输入文件夹名称'); return; }
    try {
      setSubmitting(true);
      await createFolder(activeBaseId, newFolderName.trim());
      toast.success('文件夹创建成功');
      setNewFolderName('');
      setIsCreateFolderDialogOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '创建失败');
    } finally { setSubmitting(false); }
  };

  const handleDeleteFolder = async (folderId: number) => {
    if (!confirm('确定删除此文件夹？文档将移至未分类')) return;
    try {
      await deleteFolder(folderId);
      toast.success('文件夹已删除');
    } catch { toast.error('删除失败'); }
  };

  // ========== ZIP上传 ==========
  const handleZipUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !activeBaseId) return;
    if (!file.name.endsWith('.zip')) { toast.error('请上传ZIP文件'); return; }
    try {
      setUploading(true);
      const result = await uploadZip(activeBaseId, file);
      toast.success(result.message);
      if (result.errors?.length) {
        result.errors.forEach(err => toast.warning(err));
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '上传失败');
    } finally { 
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ========== 文档操作 ==========
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) await searchKnowledge(searchQuery);
    else clearSearchResults();
  };

  const handleCreateDocument = async () => {
    if (!newDoc.title.trim() || !newDoc.content.trim()) { toast.error('请填写标题和内容'); return; }
    try {
      setSubmitting(true);
      await createDocument(newDoc);
      toast.success('文档创建成功');
      setNewDoc({ title: '', content: '' });
      setIsCreateDocDialogOpen(false);
    } catch { toast.error('创建文档失败'); }
    finally { setSubmitting(false); }
  };

  const handleEditDocument = async () => {
    if (!editingDoc || !editingDoc.title.trim() || !editingDoc.content.trim()) { toast.error('请填写标题和内容'); return; }
    try {
      setSubmitting(true);
      await updateDocument(editingDoc.id, { title: editingDoc.title, content: editingDoc.content });
      toast.success('文档更新成功');
      setIsEditDialogOpen(false);
      setEditingDoc(null);
    } catch { toast.error('更新文档失败'); }
    finally { setSubmitting(false); }
  };

  const handleDeleteDocument = async (docId: number) => {
    if (!confirm('确定删除此文档？')) return;
    try { await deleteDocument(docId); toast.success('文档删除成功'); }
    catch { toast.error('删除文档失败'); }
  };

  const openEditDialog = (doc: KnowledgeDocument) => {
    setEditingDoc({ ...doc });
    setIsEditDialogOpen(true);
  };

  // ========== 扫描操作 ==========
  const handleOpenScanDialog = async () => {
    setIsScanDialogOpen(true);
    await scanDirs();
  };

  const handleImportDir = async (dirName: string) => {
    try {
      setImporting(dirName);
      const result = await importDir(dirName);
      toast.success(result.message);
      if (result.errors?.length) {
        result.errors.forEach(err => toast.warning(err));
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '导入失败');
    } finally {
      setImporting(null);
    }
  };

  const renderScanTree = (items: ScanDirectory[], depth: number = 0): React.ReactNode => {
    return items.map((item) => {
      if (item.type === 'folder') {
        return (
          <div key={item.name} style={{ paddingLeft: `${depth * 16}px` }}>
            <div className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-accent">
              <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
              <span className="text-sm font-medium">{item.name}</span>
              {item.fileCount !== undefined && (
                <Badge variant="outline" className="text-xs ml-auto">{item.fileCount} 文件</Badge>
              )}
            </div>
            {item.children && renderScanTree(item.children, depth + 1)}
          </div>
        );
      }
      return (
        <div key={item.name} style={{ paddingLeft: `${depth * 16}px` }}>
          <div className="flex items-center gap-2 py-1 px-2 rounded hover:bg-accent">
            <FileText className="h-4 w-4 text-primary shrink-0" />
            <span className="text-sm">{item.name}</span>
            {item.size !== undefined && (
              <span className="text-xs text-muted-foreground ml-auto">
                {(item.size / 1024).toFixed(1)}KB
              </span>
            )}
          </div>
        </div>
      );
    });
  };

  return (
    <AppLayout>
      <div className="flex min-w-0 gap-6 h-[calc(100vh-8rem)]">
        {/* 左侧：知识库+文件夹树 */}
        <div className="w-72 shrink-0 flex flex-col gap-4">
          {/* 知识库列表 */}
          <Card className="flex-1 flex flex-col">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">知识库</CardTitle>
                <Dialog open={isCreateBaseDialogOpen} onOpenChange={setIsCreateBaseDialogOpen}>
                  <DialogTrigger asChild>
                    <Button size="icon" variant="ghost" className="h-7 w-7">
                      <Plus className="h-4 w-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                      <DialogTitle>创建知识库</DialogTitle>
                      <DialogDescription>知识库是文档的顶层容器，如&quot;原神知识&quot;</DialogDescription>
                    </DialogHeader>
                    <div className="py-4">
                      <Input placeholder="知识库名称" value={newBaseName} onChange={e => setNewBaseName(e.target.value)} />
                    </div>
                    <DialogFooter>
                      <Button variant="secondary" onClick={() => setIsCreateBaseDialogOpen(false)}>取消</Button>
                      <Button onClick={handleCreateBase} disabled={submitting}>创建</Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent className="flex-1 pt-0">
              <ScrollArea className="h-full">
                <div className="space-y-1">
                  <Button
                    variant={activeBaseId === null ? 'secondary' : 'ghost'}
                    size="sm"
                    className="w-full justify-start gap-2"
                    onClick={() => selectBase(null)}
                  >
                    <Database className="h-4 w-4" />
                    全部文档
                  </Button>
                  {bases.map(base => (
                    <div key={base.id} className="flex items-center gap-1">
                      <Button
                        variant={activeBaseId === base.id ? 'secondary' : 'ghost'}
                        size="sm"
                        className="flex-1 justify-start gap-2"
                        onClick={() => selectBase(base.id)}
                      >
                        <BookOpen className="h-4 w-4" />
                        <span className="truncate">{base.name}</span>
                        <Badge variant="outline" className="ml-auto text-xs">{base.documentCount}</Badge>
                      </Button>
                      <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={() => handleDeleteBase(base.id)}>
                        <Trash2 className="h-3 w-3 text-muted-foreground" />
                      </Button>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* 文件夹列表（选中知识库时显示） */}
          {activeBaseId && (
            <Card className="flex-1 flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">文件夹</CardTitle>
                  <div className="flex gap-1">
                    <Dialog open={isCreateFolderDialogOpen} onOpenChange={setIsCreateFolderDialogOpen}>
                      <DialogTrigger asChild>
                        <Button size="icon" variant="ghost" className="h-7 w-7">
                          <Plus className="h-4 w-4" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="sm:max-w-md">
                        <DialogHeader>
                          <DialogTitle>创建文件夹</DialogTitle>
                          <DialogDescription>在 {activeBase?.name} 下创建文件夹</DialogDescription>
                        </DialogHeader>
                        <div className="py-4">
                          <Input placeholder="文件夹名称" value={newFolderName} onChange={e => setNewFolderName(e.target.value)} />
                        </div>
                        <DialogFooter>
                          <Button variant="secondary" onClick={() => setIsCreateFolderDialogOpen(false)}>取消</Button>
                          <Button onClick={handleCreateFolder} disabled={submitting}>创建</Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                    <>
                      <input ref={fileInputRef} type="file" accept=".zip" className="hidden" onChange={handleZipUpload} />
                      <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                        {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Archive className="h-4 w-4" />}
                      </Button>
                    </>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 pt-0">
                <ScrollArea className="h-full">
                  <div className="space-y-1">
                    <Button
                      variant={activeFolderId === null ? 'secondary' : 'ghost'}
                      size="sm"
                      className="w-full justify-start gap-2"
                      onClick={() => selectFolder(null)}
                    >
                      <FolderOpen className="h-4 w-4" />
                      全部
                    </Button>
                    {folders.map(folder => (
                      <div key={folder.id} className="flex items-center gap-1">
                        <Button
                          variant={activeFolderId === folder.id ? 'secondary' : 'ghost'}
                          size="sm"
                          className="flex-1 justify-start gap-2"
                          onClick={() => selectFolder(folder.id)}
                        >
                          <Folder className="h-4 w-4" />
                          <span className="truncate">{folder.name}</span>
                          <Badge variant="outline" className="ml-auto text-xs">{folder.documentCount}</Badge>
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={() => handleDeleteFolder(folder.id)}>
                          <Trash2 className="h-3 w-3 text-muted-foreground" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 右侧：文档内容区 */}
        <div className="min-w-0 flex-1 space-y-6 overflow-auto">
          {/* 标题栏 */}
          <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                {activeBase ? activeBase.name : '知识库'}
                {activeFolder && <span className="text-muted-foreground"> / {activeFolder.name}</span>}
              </h2>
              <p className="text-muted-foreground text-sm">
                {activeBaseId ? '管理和检索知识库文档' : '选择左侧知识库或查看全部文档'}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={handleOpenScanDialog}>
                <ScanSearch className="mr-2 h-4 w-4" />
                扫描文件夹
              </Button>
              {activeBaseId && (
                <>
                  <input ref={fileInputRef} type="file" accept=".zip" className="hidden" onChange={handleZipUpload} />
                  <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                    {uploading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Archive className="mr-2 h-4 w-4" />}
                    上传ZIP
                  </Button>
                </>
              )}
              <Dialog open={isCreateDocDialogOpen} onOpenChange={setIsCreateDocDialogOpen}>
                <DialogTrigger asChild>
                  <Button><Plus className="mr-2 h-4 w-4" />添加文档</Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-2xl max-h-[85vh] flex flex-col">
                  <DialogHeader>
                    <DialogTitle>添加新文档</DialogTitle>
                    <DialogDescription>
                      {activeBase ? `将文档添加到 ${activeBase.name}${activeFolder ? ` / ${activeFolder.name}` : ''}` : '创建知识库文档'}
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4 overflow-y-auto flex-1 min-h-0">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">标题</label>
                      <Input placeholder="输入文档标题" value={newDoc.title} onChange={e => setNewDoc({ ...newDoc, title: e.target.value })} />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">内容</label>
                      <Textarea placeholder="输入文档内容" className="min-h-[200px] max-h-[50vh]" value={newDoc.content} onChange={e => setNewDoc({ ...newDoc, content: e.target.value })} />
                    </div>
                  </div>
                  <DialogFooter className="shrink-0">
                    <Button variant="secondary" onClick={() => setIsCreateDocDialogOpen(false)}>取消</Button>
                    <Button onClick={handleCreateDocument} disabled={submitting}>
                      {submitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />创建中...</> : '创建文档'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>

          {/* 搜索框 */}
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="搜索知识库..." className="pl-10" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
              {searchQuery && (
                <button type="button" onClick={() => { setSearchQuery(''); clearSearchResults(); }} className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <X className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                </button>
              )}
            </div>
            <Button type="submit" disabled={searching}>
              {searching ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />搜索中...</> : '搜索'}
            </Button>
          </form>

          {/* 搜索结果 */}
          {searchResults.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>搜索结果</CardTitle>
                <CardDescription>找到 {searchResults.length} 个相关片段</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {searchResults.map((result, index) => (
                  <Card key={index} className="border-l-4 border-l-primary">
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium">{result.documentTitle}</h4>
                        <span className="text-sm text-muted-foreground">相关度: {(result.score * 100).toFixed(0)}%</span>
                      </div>
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">{result.content}</p>
                    </CardContent>
                  </Card>
                ))}
              </CardContent>
            </Card>
          )}

          {/* 统计卡片 */}
          {!searchResults.length && (
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">文档总数</CardTitle>
                  <FileText className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.totalDocuments || 0}</div>
                  <p className="text-xs text-muted-foreground">已上传的文档</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">向量条目</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.totalChunks || 0}</div>
                  <p className="text-xs text-muted-foreground">约 {Math.round((stats?.totalCharacters || 0) / 1000)} 千字</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">总字符数</CardTitle>
                  <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{((stats?.totalCharacters || 0) / 1000).toFixed(1)}K</div>
                  <p className="text-xs text-muted-foreground">知识库总字数</p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* 文档列表 */}
          {!searchResults.length && (
            <Card>
              <CardHeader>
                <CardTitle>
                  {activeFolder ? `${activeFolder.name}文件夹` : activeBase ? `${activeBase.name} - 全部文档` : '全部文档'}
                </CardTitle>
                <CardDescription>
                  {documents.length > 0 ? `共 ${documents.length} 篇文档` : '暂无文档'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center py-12"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
                ) : error ? (
                  <div className="text-center py-12 text-muted-foreground"><p>加载失败: {error}</p></div>
                ) : documents.length === 0 ? (
                  <div className="border-2 border-dashed rounded-lg p-8 text-center">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      {activeBaseId ? '上传ZIP文件批量导入，或手动添加文档' : '选择知识库后添加文档'}
                    </p>
                    {activeBaseId && (
                      <p className="text-xs text-muted-foreground">
                        ZIP结构：文件夹名/文件名.txt，文件夹名自动成为分类
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {documents.map(doc => (
                      <Card key={doc.id} className="cursor-pointer hover:bg-accent transition-colors">
                        <CardContent className="pt-4 pb-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <FileText className="h-4 w-4 text-primary shrink-0" />
                                <h3 className="font-medium truncate">{doc.title}</h3>
                                {doc.category && doc.category !== '未分类' && (
                                  <Badge variant="outline" className="text-xs shrink-0">{doc.category}</Badge>
                                )}
                              </div>
                              <p className="text-sm text-muted-foreground line-clamp-2 mb-2">{doc.content}</p>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <span>{doc.chunkCount} 个片段</span>
                                <span>更新于 {formatDistanceToNow(new Date(doc.updatedAt), { addSuffix: true, locale: zhCN })}</span>
                              </div>
                            </div>
                            <div className="flex gap-1 ml-2 shrink-0">
                              <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => openEditDialog(doc)}>
                                <Edit className="h-4 w-4" />
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleDeleteDocument(doc.id)}>
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* 扫描文件夹对话框 */}
      <Dialog open={isScanDialogOpen} onOpenChange={setIsScanDialogOpen}>
        <DialogContent className="sm:max-w-2xl max-h-[85vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>扫描知识库文件夹</DialogTitle>
            <DialogDescription>
              扫描 backend/knowledge_bases/ 目录下的文件夹，预览后一键导入
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 overflow-y-auto flex-1 min-h-0">
            {scanning ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                <span className="ml-3 text-muted-foreground">扫描中...</span>
              </div>
            ) : scanResults.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <FolderOpen className="h-8 w-8 mx-auto mb-2" />
                <p>未发现可用的知识库文件夹</p>
                <p className="text-xs mt-1">请将文件夹放入 backend/knowledge_bases/ 目录</p>
              </div>
            ) : (
              <div className="space-y-4">
                {scanResults.map((dir) => (
                  <Card key={dir.name}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <BookOpen className="h-5 w-5 text-primary" />
                          <CardTitle className="text-base">{dir.name}</CardTitle>
                          <Badge variant="outline" className="text-xs">
                            {dir.fileCount} 文件 · {(dir.totalSize || 0) / 1024 < 1024 
                              ? `${((dir.totalSize || 0) / 1024).toFixed(1)}KB` 
                              : `${((dir.totalSize || 0) / 1024 / 1024).toFixed(1)}MB`}
                          </Badge>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => handleImportDir(dir.name)}
                          disabled={importing === dir.name}
                        >
                          {importing === dir.name ? (
                            <><Loader2 className="mr-1 h-3 w-3 animate-spin" />导入中...</>
                          ) : '导入'}
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="border rounded-md p-2">
                        {dir.children && renderScanTree(dir.children, 0)}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
          <DialogFooter className="shrink-0">
            <Button variant="secondary" onClick={() => setIsScanDialogOpen(false)}>关闭</Button>
            <Button variant="outline" onClick={scanDirs} disabled={scanning}>
              {scanning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ScanSearch className="mr-2 h-4 w-4" />}
              重新扫描
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 编辑对话框 */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="sm:max-w-2xl max-h-[85vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>编辑文档</DialogTitle>
            <DialogDescription>修改文档的标题和内容</DialogDescription>
          </DialogHeader>
          {editingDoc && (
            <div className="space-y-4 py-4 overflow-y-auto flex-1 min-h-0">
              <div className="space-y-2">
                <label className="text-sm font-medium">标题</label>
                <Input value={editingDoc.title} onChange={e => setEditingDoc({ ...editingDoc, title: e.target.value })} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">内容</label>
                <Textarea className="min-h-[200px] max-h-[50vh]" value={editingDoc.content} onChange={e => setEditingDoc({ ...editingDoc, content: e.target.value })} />
              </div>
            </div>
          )}
          <DialogFooter className="shrink-0">
            <Button variant="secondary" onClick={() => setIsEditDialogOpen(false)}>取消</Button>
            <Button onClick={handleEditDocument} disabled={submitting}>
              {submitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />更新中...</> : '更新文档'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  );
}
