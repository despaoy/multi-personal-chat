'use client';

/**
 * 知识库管理 Hook
 *
 * 提供知识库、文件夹、文档的完整 CRUD 操作和语义搜索功能。
 * 支持2级层级：知识库 → 文件夹 → 文档
 */

import { useState, useEffect, useCallback } from 'react';
import { toast } from 'sonner';
import {
  api,
  KnowledgeDocument,
  KnowledgeBase,
  KnowledgeFolder,
  KnowledgeStats,
  KnowledgeSearchResult,
  KnowledgeCreateRequest,
  KnowledgeUpdateRequest,
  ScanDirectory
} from '@/lib/api';

export function useKnowledge(enabled = true) {
  // 知识库
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [activeBaseId, setActiveBaseId] = useState<number | null>(null);

  // 文件夹
  const [folders, setFolders] = useState<KnowledgeFolder[]>([]);
  const [activeFolderId, setActiveFolderId] = useState<number | null>(null);

  // 文档
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);
  
  // 搜索
  const [searchResults, setSearchResults] = useState<KnowledgeSearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  // 扫描
  const [scanResults, setScanResults] = useState<ScanDirectory[]>([]);
  const [scanning, setScanning] = useState(false);

  // ========== 知识库操作 ==========
  const fetchBases = useCallback(async () => {
    try {
      const data = await api.getKnowledgeBases();
      setBases(data.bases);
    } catch (err) {
      console.error('Failed to fetch knowledge bases:', err);
    }
  }, []);

  const createBase = async (name: string, description: string = '') => {
    try {
      const data = await api.createKnowledgeBase(name, description);
      await fetchBases();
      return data.base;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '创建失败';
      setError(msg);
      toast.error(msg);
      throw err;
    }
  };

  const deleteBase = async (kbId: number) => {
    try {
      await api.deleteKnowledgeBase(kbId);
      if (activeBaseId === kbId) {
        setActiveBaseId(null);
        setFolders([]);
        setDocuments([]);
      }
      await fetchBases();
    } catch (err) {
      const msg = err instanceof Error ? err.message : '删除失败';
      setError(msg);
      toast.error(msg);
      throw err;
    }
  };

  const selectBase = (kbId: number | null) => {
    setActiveBaseId(kbId);
    setActiveFolderId(null);
  };

  // ========== 文件夹操作 ==========
  const fetchFolders = useCallback(async (kbId: number) => {
    try {
      const data = await api.getKnowledgeFolders(kbId);
      setFolders(data.folders);
    } catch (err) {
      console.error('Failed to fetch folders:', err);
    }
  }, []);

  const createFolder = async (kbId: number, name: string) => {
    try {
      const data = await api.createKnowledgeFolder(kbId, name);
      await fetchFolders(kbId);
      return data.folder;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '创建文件夹失败';
      setError(msg);
      toast.error(msg);
      throw err;
    }
  };

  const deleteFolder = async (folderId: number) => {
    try {
      await api.deleteKnowledgeFolder(folderId);
      if (activeBaseId) await fetchFolders(activeBaseId);
      if (activeFolderId === folderId) setActiveFolderId(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : '删除文件夹失败';
      setError(msg);
      toast.error(msg);
      throw err;
    }
  };

  const selectFolder = (folderId: number | null) => {
    setActiveFolderId(folderId);
  };

  // ========== 文档操作 ==========
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getKnowledgeDocuments(
        undefined, undefined, undefined,
        activeBaseId || undefined,
        activeFolderId || undefined
      );
      setDocuments(data.documents);
      setStats(data.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取知识库文档失败');
      console.error('Failed to fetch knowledge documents:', err);
    } finally {
      setLoading(false);
    }
  }, [activeBaseId, activeFolderId]);

  const searchKnowledge = async (query: string, topK: number = 5) => {
    try {
      setSearching(true);
      setError(null);
      const data = await api.searchKnowledge({ query, topK });
      setSearchResults(data.results);
      return data.results;
    } catch (err) {
      setError(err instanceof Error ? err.message : '搜索知识库失败');
      console.error('Failed to search knowledge:', err);
      return [];
    } finally {
      setSearching(false);
    }
  };

  const createDocument = async (request: KnowledgeCreateRequest) => {
    try {
      setError(null);
      const data = await api.createKnowledgeDocument({
        ...request,
        knowledge_base_id: activeBaseId,
        folder_id: activeFolderId,
      });
      await fetchDocuments();
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建文档失败');
      console.error('Failed to create document:', err);
      throw err;
    }
  };

  const updateDocument = async (docId: number, request: KnowledgeUpdateRequest) => {
    try {
      setError(null);
      const data = await api.updateKnowledgeDocument(docId, request);
      await fetchDocuments();
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新文档失败');
      console.error('Failed to update document:', err);
      throw err;
    }
  };

  const deleteDocument = async (docId: number) => {
    try {
      setError(null);
      const data = await api.deleteKnowledgeDocument(docId);
      await fetchDocuments();
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除文档失败');
      console.error('Failed to delete document:', err);
      throw err;
    }
  };

  const fetchStats = async () => {
    try {
      const data = await api.getKnowledgeStats();
      setStats(data.stats);
    } catch (err) {
      console.error('Failed to fetch knowledge stats:', err);
    }
  };

  const uploadZip = async (kbId: number, file: File) => {
    try {
      const result = await api.uploadKnowledgeZip(kbId, file);
      await fetchFolders(kbId);
      if (activeBaseId === kbId) await fetchDocuments();
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '上传ZIP失败';
      setError(msg);
      throw err;
    }
  };

  // ========== 扫描操作 ==========
  const scanDirs = async () => {
    try {
      setScanning(true);
      const data = await api.scanKnowledgeDirs();
      setScanResults(data.directories);
      return data.directories;
    } catch (err) {
      console.error('Failed to scan dirs:', err);
      return [];
    } finally {
      setScanning(false);
    }
  };

  const importDir = async (directoryName: string, kbId?: number) => {
    try {
      const result = await api.importScannedDir(directoryName, kbId);
      await fetchBases();
      if (activeBaseId) {
        await fetchFolders(activeBaseId);
        await fetchDocuments();
      }
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '导入目录失败';
      setError(msg);
      throw err;
    }
  };

  // ========== 副作用 ==========
  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    fetchBases();
  }, [fetchBases, enabled]);

  useEffect(() => {
    if (!enabled) return;
    if (activeBaseId) {
      fetchFolders(activeBaseId);
    } else {
      setFolders([]);
    }
  }, [activeBaseId, fetchFolders, enabled]);

  useEffect(() => {
    if (!enabled) return;
    fetchDocuments();
  }, [fetchDocuments, enabled]);

  return {
    // 知识库
    bases,
    activeBaseId,
    selectBase,
    createBase,
    deleteBase,
    fetchBases,
    // 文件夹
    folders,
    activeFolderId,
    selectFolder,
    createFolder,
    deleteFolder,
    fetchFolders,
    // 文档
    documents,
    stats,
    loading,
    error,
    fetchDocuments,
    createDocument,
    updateDocument,
    deleteDocument,
    fetchStats,
    // 搜索
    searchResults,
    searching,
    searchKnowledge,
    clearSearchResults: () => setSearchResults([]),
    // ZIP上传
    uploadZip,
    // 扫描
    scanResults,
    scanning,
    scanDirs,
    importDir,
  };
}
