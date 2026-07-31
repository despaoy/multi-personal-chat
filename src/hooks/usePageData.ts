'use client';

import { useState, useEffect, useCallback, useRef, type Dispatch, type SetStateAction } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api';

const PAGE_DATA_STORAGE_PREFIX = 'qq_assistant_data_v2';

function getPageDataStorageKey(pageKey: string, userId: number | null): string {
  const owner = userId === null ? 'anonymous' : String(userId);
  return `${PAGE_DATA_STORAGE_PREFIX}_${owner}_${encodeURIComponent(pageKey)}`;
}

function parseJsonRecord(raw: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Invalid cached/server data falls back to the supplied defaults.
  }
  return null;
}

async function loadServerPageData(pageKey: string): Promise<Record<string, unknown> | null> {
  const response = await api.getUserData(pageKey);
  if (!response.success || !response.data || !('data_json' in response.data)) {
    return null;
  }

  const dataJson = response.data.data_json;
  return typeof dataJson === 'string' ? parseJsonRecord(dataJson) : null;
}


/**
 * 页面表单数据持久化 Hook
 * 自动保存和恢复页面表单数据，刷新后不丢失
 *
 * @param pageKey 页面唯一标识（如 'training', 'settings'）
 * @param defaultData 默认数据
 * @returns [data, setData, saveData] 数据、设置函数、手动保存函数
 */
export function usePageData<T extends Record<string, unknown>>(
  pageKey: string,
  defaultData: T
): [T, Dispatch<SetStateAction<T>>, () => Promise<void>] {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const identity = `${userId === null ? 'anonymous' : userId}:${pageKey}`;
  const storageKey = getPageDataStorageKey(pageKey, userId);
  const defaultDataSignature = JSON.stringify(defaultData);

  const [data, setDataState] = useState<T>(defaultData);
  const [hydratedIdentity, setHydratedIdentity] = useState<string | null>(null);
  const [saveRetryCount, setSaveRetryCount] = useState(0);
  const defaultDataRef = useRef(defaultData);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadVersionRef = useRef(0);
  const dirtyIdentityRef = useRef<string | null>(null);
  const persistedSnapshotRef = useRef<string | null>(null);

  useEffect(() => {
    defaultDataRef.current = defaultData;
  }, [defaultData]);

  const setData = useCallback<Dispatch<SetStateAction<T>>>((nextData) => {
    dirtyIdentityRef.current = identity;
    setSaveRetryCount(0);
    setDataState(nextData);
  }, [identity]);

  const persistData = useCallback(async (value: T) => {
    const serialized = JSON.stringify(value);
    localStorage.setItem(storageKey, serialized);

    if (userId !== null) {
      await api.saveUserData(pageKey, serialized);
    }
    persistedSnapshotRef.current = serialized;
  }, [pageKey, storageKey, userId]);

  // 初始化：从 localStorage 或后端加载数据
  useEffect(() => {
    const loadVersion = ++loadVersionRef.current;
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }

    setHydratedIdentity(null);
    setSaveRetryCount(0);
    dirtyIdentityRef.current = null;
    persistedSnapshotRef.current = null;

    const defaults = defaultDataRef.current;
    let cachedData = defaults;
    try {
      const localData = localStorage.getItem(storageKey);
      const parsed = localData ? parseJsonRecord(localData) : null;
      if (parsed) {
        cachedData = { ...defaults, ...parsed } as T;
      }
    } catch (err) {
      console.error('Failed to load local page data:', err);
    }
    setDataState(cachedData);

    const hydrate = async () => {
      let resolvedData = cachedData;

      if (userId !== null) {
        try {
          const serverData = await loadServerPageData(pageKey);
          if (serverData) {
            resolvedData = { ...defaults, ...serverData } as T;
          }
        } catch (err) {
          console.error('Failed to load server page data:', err);
        }
      }

      if (loadVersionRef.current !== loadVersion) {
        return;
      }

      try {
        persistedSnapshotRef.current = JSON.stringify(resolvedData);
      } catch (err) {
        console.error('Failed to serialize hydrated page data:', err);
        persistedSnapshotRef.current = null;
      }

      // Preserve edits made while hydration was in flight.
      if (dirtyIdentityRef.current !== identity) {
        setDataState(resolvedData);
      }
      setHydratedIdentity(identity);
    };

    void hydrate();

    return () => {
      if (loadVersionRef.current === loadVersion) {
        loadVersionRef.current += 1;
      }
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
    };
  }, [defaultDataSignature, identity, pageKey, storageKey, userId]);

  // 手动保存：先 localStorage 再后端
  const saveData = useCallback(async () => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }

    // An explicit save wins over an in-flight hydration request.
    if (hydratedIdentity !== identity) {
      loadVersionRef.current += 1;
      setHydratedIdentity(identity);
    }

    try {
      await persistData(data);
      setSaveRetryCount(0);
    } catch (err) {
      console.error('Failed to save page data:', err);
      setSaveRetryCount((count) => count + 1);
    }
  }, [data, hydratedIdentity, identity, persistData]);

  // 数据变化时自动延迟保存（1秒防抖）
  useEffect(() => {
    if (hydratedIdentity !== identity) {
      return;
    }

    let serialized: string;
    try {
      serialized = JSON.stringify(data);
    } catch (err) {
      console.error('Failed to serialize page data:', err);
      return;
    }

    // Hydration establishes the baseline; only user changes are auto-saved.
    if (serialized === persistedSnapshotRef.current) {
      return;
    }

    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }

    const retryDelay = saveRetryCount === 0
      ? 1000
      : Math.min(30000, 1000 * 2 ** Math.min(saveRetryCount, 5));
    const valueToPersist = data;
    saveTimerRef.current = setTimeout(() => {
      saveTimerRef.current = null;
      void persistData(valueToPersist)
        .then(() => setSaveRetryCount(0))
        .catch((err) => {
          console.error('Auto-save failed:', err);
          setSaveRetryCount((count) => count + 1);
        });
    }, retryDelay);

    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
    };
  }, [data, hydratedIdentity, identity, persistData, saveRetryCount]);

  return [data, setData, saveData];
}
