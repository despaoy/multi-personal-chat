'use client';

import { useAuth } from '@/contexts/AuthContext';
import { AppLayout } from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { LogIn, ShieldAlert } from 'lucide-react';

/**
 * 认证守卫组件
 * - loading期间显示骨架屏，不渲染子组件（避免用localStorage缓存的无效user触发API调用）
 * - 未登录时显示登录提示
 * - 已登录且验证通过时渲染子组件
 * - requireAdmin=true 时额外检查 role === 'admin'，非 admin 用户显示无权访问提示
 *   M1 fix: 此前 adminOnly 页面仅靠 Sidebar 过滤可见性，用户直接输入 URL 仍可访问页面 UI
 */
export function AuthGuard({
  children,
  requireAdmin = false,
}: {
  children: React.ReactNode;
  requireAdmin?: boolean;
}) {
  const { user, loading } = useAuth();

  // 正在验证token，不渲染子组件避免无效API调用
  if (loading) {
    return (
      <AppLayout>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-4 w-64 mt-2" />
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-32" />
            ))}
          </div>
        </div>
      </AppLayout>
    );
  }

  if (!user) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <LogIn className="h-12 w-12 text-muted-foreground" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">请先登录</h3>
            <p className="text-muted-foreground">登录后即可查看此页面内容</p>
          </div>
          <Button onClick={() => window.location.href = '/login'}>
            <LogIn className="mr-2 h-4 w-4" />
            前往登录
          </Button>
        </div>
      </AppLayout>
    );
  }

  // M1 fix: adminOnly 页面的纵深防御
  if (requireAdmin && user.role !== 'admin') {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <ShieldAlert className="h-12 w-12 text-muted-foreground" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">无权访问</h3>
            <p className="text-muted-foreground">此页面仅管理员可访问</p>
          </div>
        </div>
      </AppLayout>
    );
  }

  return <>{children}</>;
}
