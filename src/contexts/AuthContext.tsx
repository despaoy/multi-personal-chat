'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { api, User } from '@/lib/api';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  // 不在 useState 初始化时从 localStorage 恢复 user，避免 SSR/CSR hydration 不匹配
  // 也不在 loading 期间让 user 有值，防止子组件在 token 验证前发起 API 请求
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // The server is the only source of authentication state, never localStorage.
  useEffect(() => {
    let cancelled = false;
    api.getCurrentUser()
      .then(({ user }) => {
        if (!cancelled) setUser(user);
      })
      .catch(() => {
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const { user } = await api.login({ username, password });
    setUser(user);
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const { user } = await api.register({ username, password });
    setUser(user);
  }, []);

  const logout = useCallback(async () => {
    try {
      // 等待响应完成，确保 HttpOnly Cookie 在页面跳转前已经清除。
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } finally {
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
