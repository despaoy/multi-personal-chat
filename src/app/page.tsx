/**
 * 仪表盘首页
 *
 * 为应用入口页面，使用 AppLayout 布局包裹仪表盘客户端组件。
 * 定义页面 SEO 元数据（标题和描述），将实际的仪表盘交互逻辑委托给 DashboardClient 客户端组件。
 *
 * @module Dashboard
 * @see {@link ./DashboardClient.tsx} 仪表盘客户端组件
 */

import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import DashboardClient from './DashboardClient';

/** 页面 SEO 元数据 */
export const metadata = {
  title: '仪表盘 | MultiPersonal Chat System',
  description: 'MultiPersonal Chat System管理平台',
};

/**
 * 仪表盘页面组件（服务器组件）
 *
 * 负责：
 * - 导出页面 metadata（仅服务器组件可用）
 * - 组装 AuthGuard + AppLayout 与 DashboardClient
 *
 * M2 fix: 仪表盘作为应用首页，此前是唯一未受 AuthGuard 保护的页面，
 * 未登录用户可直接访问并触发后续 API 调用。现统一用 AuthGuard 包裹，
 * 并配合 src/proxy.ts 做集中式路由守卫，形成双层防护。
 */
export default function Dashboard() {
  return (
    <AuthGuard requireAdmin>
      <AppLayout>
        <DashboardClient />
      </AppLayout>
    </AuthGuard>
  );
}
