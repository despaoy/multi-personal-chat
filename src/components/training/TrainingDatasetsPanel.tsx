'use client';

import { Database, Download, FileText, Loader2, Plus, Search } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import type { api, Dataset } from '@/lib/api';

interface TrainingDatasetsPanelProps {
  showScanSection: boolean;
  scanFolderPath: string;
  setScanFolderPath: (value: string) => void;
  scanLoading: boolean;
  scanResults: Awaited<ReturnType<typeof api.scanDatasets>>['datasets'];
  importLoading: string | null;
  handleScanFolder: () => void;
  handleImportAll: () => void;
  handleImportDataset: (path: string, name: string) => void;
  datasetsLoading: boolean;
  datasets: Dataset[];
  setCreateDatasetOpen: (open: boolean) => void;
  handleExportDataset: (name: string) => void;
}

export function TrainingDatasetsPanel({
  showScanSection,
  scanFolderPath,
  setScanFolderPath,
  scanLoading,
  scanResults,
  importLoading,
  handleScanFolder,
  handleImportAll,
  handleImportDataset,
  datasetsLoading,
  datasets,
  setCreateDatasetOpen,
  handleExportDataset,
}: TrainingDatasetsPanelProps) {
  return (
    <>
      {/* 扫描文件夹面板 */}
      {showScanSection && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              扫描文件夹导入数据集
            </CardTitle>
            <CardDescription>
              将所有数据集放在同一文件夹下，每个子文件夹为一个数据集（子文件夹名 = 数据集名称）
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="输入文件夹路径，留空使用默认目录"
                value={scanFolderPath}
                onChange={(e) => setScanFolderPath(e.target.value)}
                className="flex-1"
              />
              <Button onClick={handleScanFolder} disabled={scanLoading}>
                {scanLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    扫描中...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    开始扫描
                  </>
                )}
              </Button>
            </div>

            {scanResults.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    发现 {scanResults.length} 个数据集
                  </span>
                  <Button size="sm" variant="outline" onClick={handleImportAll}>
                    <Download className="mr-2 h-3 w-3" />
                    一键导入全部
                  </Button>
                </div>
                <div className="space-y-2">
                  {scanResults.map((ds) => (
                    <div
                      key={ds.name}
                      className="flex items-center justify-between p-3 border rounded-lg bg-muted/30"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
                        <div className="min-w-0">
                          <p className="font-medium truncate">{ds.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {ds.file_count} 个文件 · {ds.files.slice(0, 3).join(', ')}{ds.files.length > 3 ? '...' : ''}
                          </p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => handleImportDataset(ds.path, ds.name)}
                        disabled={importLoading === ds.name}
                      >
                        {importLoading === ds.name ? (
                          <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                        ) : (
                          <Download className="mr-1 h-3 w-3" />
                        )}
                        导入
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {datasetsLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-full mt-2" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : datasets.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">暂无数据集</h3>
              <p className="text-muted-foreground mb-4">
                创建您的第一个训练数据集
              </p>
              <Button onClick={() => setCreateDatasetOpen(true)}>
                <Plus className="mr-2 h-4 w-4" />
                新建数据集
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <Card key={dataset.name}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      {dataset.name}
                    </CardTitle>
                    <CardDescription>训练数据集</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">总样本数</span>
                  <Badge variant="secondary">{dataset.stats.total || 0}</Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">训练集</span>
                  <span>{dataset.stats.train || 0}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">验证集</span>
                  <span>{dataset.stats.eval || 0}</span>
                </div>
              </CardContent>
              <CardFooter className="border-t pt-4">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => handleExportDataset(dataset.name)}
                >
                  <Download className="mr-2 h-4 w-4" />
                  导出数据集
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
