/** 知识库文档 */
export interface KnowledgeDocument {
  id: number;
  title: string;
  content: string;
  category: string;
  knowledge_base_id: number | null;
  folder_id: number | null;
  sourceType: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
  chunkCount: number;
  createdAt: string;
  updatedAt: string;
}

/** 知识库（顶层容器） */
export interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
  documentCount: number;
  folderCount: number;
  created_at: string;
  updated_at: string;
}

/** 知识库文件夹 */
export interface KnowledgeFolder {
  id: number;
  knowledge_base_id: number;
  name: string;
  description: string;
  documentCount: number;
  created_at: string;
  updated_at: string;
}

export type KnowledgeTextEncoding = 'unicode' | 'gbk';

/** 知识导入结果：只把全部完成标记为成功。 */
export interface KnowledgeImportResult {
  status: 'complete' | 'partial' | 'failed' | 'unconfirmed';
  success: boolean;
  message: string;
  createdFolders: string[];
  createdDocs: number;
  errors: string[];
}

/** 扫描目录树节点 */
export interface ScanDirectory {
  name: string;
  type: 'folder' | 'file';
  children?: ScanDirectory[];
  fileCount?: number;
  totalSize?: number;
  size?: number;
  extension?: string;
}

/** 知识库搜索结果项 - 包含相关性评分 */
export interface KnowledgeSearchResult {
  documentId: number;
  documentTitle: string;
  /** 后端多数分支不返回 chunkId，仅部分向量检索路径可能携带 */
  chunkId?: number;
  chunkIndex: number;
  content: string;
  score: number;
}

/** @deprecated 使用 KnowledgeRetrievalMode。保留该类型用于兼容旧 API 客户端。 */
type KnowledgeSearchType = "rag_pipeline" | "empty";

/** 稳定的知识检索模式。 */
type KnowledgeRetrievalMode = "evidence" | "empty";

/** 知识库统计数据 */
export interface KnowledgeStats {
  totalDocuments: number;
  totalChunks: number;
  totalCharacters: number;
}

/** 知识库文档列表响应 */
export interface KnowledgeDocumentsResponse {
  success: boolean;
  documents: KnowledgeDocument[];
  stats: KnowledgeStats;
}

/** 创建知识库文档请求参数 */
export interface KnowledgeCreateRequest {
  title: string;
  content: string;
  category?: string;
  knowledge_base_id?: number | null;
  folder_id?: number | null;
  sourceType?: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
}

/** 更新知识库文档请求参数 */
export interface KnowledgeUpdateRequest {
  title?: string;
  content?: string;
  category?: string;
  knowledge_base_id?: number | null;
  folder_id?: number | null;
  sourceType?: string;
  sourceUrl?: string;
  fileType?: string;
  fileSize?: number;
}

/** 知识库搜索请求参数 */
export interface KnowledgeSearchRequest {
  query: string;
  topK?: number;
  /** 按知识库名称过滤检索结果，用于意图分类器路由后的精准检索 */
  knowledgeBaseName?: string;
}

/** 知识库搜索响应 */
export interface KnowledgeSearchResponse {
  success: boolean;
  query: string;
  results: KnowledgeSearchResult[];
  /** @deprecated 使用 retrievalMode；将在下一个 API major 版本移除。 */
  searchType: KnowledgeSearchType;
  /** 完成检索时为 evidence，无匹配时为 empty；系统故障由接口报错。 */
  retrievalMode: KnowledgeRetrievalMode;
}

/** 知识库统计响应 */
export interface KnowledgeStatsResponse {
  success: boolean;
  stats: KnowledgeStats;
}
