'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { api, SystemConfig } from '@/lib/api';
import { Locale, t as translate } from '@/lib/i18n';

interface SettingsContextType {
  locale: Locale;
  timezone: string;
  config: SystemConfig;
  loading: boolean;
  t: (key: string, fallback?: string) => string;
  formatTime: (date: Date, options?: Intl.DateTimeFormatOptions) => string;
  syncSettings: (newConfig: SystemConfig) => void;
  refreshSettings: () => Promise<void>;
}

const SettingsContext = createContext<SettingsContextType | null>(null);

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<SystemConfig>({});
  const [loading, setLoading] = useState(true);

  const locale = (config.language as Locale) || 'zh-CN';
  const timezone = (config.timezone as string) || 'Asia/Shanghai';

  const t = useCallback((key: string, fallback?: string) => {
    return translate(locale, key, fallback);
  }, [locale]);

  const formatTime = useCallback((date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      timeZone: timezone,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    };
    return date.toLocaleTimeString(
      locale === 'en' ? 'en-US' : locale === 'zh-TW' ? 'zh-TW' : 'zh-CN',
      options || defaultOptions
    );
  }, [locale, timezone]);

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getConfig();
      setConfig(data.config);
    } catch (err) {
      console.error('Failed to fetch settings:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Sync a configuration that the caller has already persisted.
  const syncSettings = useCallback((newConfig: SystemConfig) => {
    setConfig(newConfig);
  }, []);

  const refreshSettings = useCallback(async () => {
    await fetchConfig();
  }, [fetchConfig]);

  useEffect(() => {

    fetchConfig();
  }, [fetchConfig]);

  return (
    <SettingsContext.Provider value={{ locale, timezone, config, loading, t, formatTime, syncSettings, refreshSettings }}>
      {children}
    </SettingsContext.Provider>
  );
}
