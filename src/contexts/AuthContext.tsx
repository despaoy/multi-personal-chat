'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { User } from '@/lib/api';

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

  // 异步验证 token 是否仍然有效（Cookie 由浏览器自动携带）
  useEffect(() => {
    fetch('/api/auth/me', {
      credentials: 'include', // 携带 httpOnly Cookie
    }).then(async res => {
      if (res.ok) {
        const data = await res.json();
        if (data.user) {
          // M3 fix: 捕获后端返回的 role 字段，供侧栏按角色过滤导航项
          const userData: User = {
            id: data.user.id,
            username: data.user.username,
            created_at: data.user.created_at || '',
            role: data.user.role === 'admin' ? 'admin' : 'user',
          };
          localStorage.setItem('qq_assistant_user', JSON.stringify(userData));
          setUser(userData);
        }
      } else {
        localStorage.removeItem('qq_assistant_user');
        setUser(null);
      }
    }).catch(() => {
      // 安全：网络失败时不可从 localStorage 恢复 user。
      // localStorage 是客户端可篡改的存储，把它当作鉴权依据会让任何人在断网/伪造时绕过登录。
      // 离线即视为未认证，由 AuthGuard 引导用户重新登录。
      localStorage.removeItem('qq_assistant_user');
      setUser(null);
    }).finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include', // 接收 httpOnly Cookie
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      let errorMessage = '登录失败';
      try {
        const err = await response.json();
        errorMessage = err.detail || errorMessage;
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    const data = await response.json();
    // Only store non-sensitive user info (never store token)
    // M3 fix: 捕获 role，供侧栏按角色过滤
    const userData: User = {
      id: data.user.id,
      username: data.user.username,
      created_at: data.user.created_at || '',
      role: data.user.role === 'admin' ? 'admin' : 'user',
    };
    localStorage.setItem('qq_assistant_user', JSON.stringify(userData));
    setUser(userData);
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include', // 接收 httpOnly Cookie
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      let errorMessage = '注册失败';
      try {
        const err = await response.json();
        errorMessage = err.detail || errorMessage;
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    const data = await response.json();
    // Only store non-sensitive user info (never store token)
    // M3 fix: 捕获 role，供侧栏按角色过滤
    const userData: User = {
      id: data.user.id,
      username: data.user.username,
      created_at: data.user.created_at || '',
      role: data.user.role === 'admin' ? 'admin' : 'user',
    };
    localStorage.setItem('qq_assistant_user', JSON.stringify(userData));
    setUser(userData);
  }, []);

  const logout = useCallback(async () => {
    try {
      // 等待响应完成，确保 HttpOnly Cookie 在页面跳转前已经清除。
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } finally {
      localStorage.removeItem('qq_assistant_user');
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
