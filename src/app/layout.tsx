/**
 * 根布局组件
 *
 * 全站最外层布局，配置：
 * - 主题切换（next-themes ThemeProvider）
 * - 全局设置（SettingsProvider - 语言/时区）
 * - 全局 Toast 通知容器
 *
 * 包含 SEO 元数据（title、description、keywords）。
 */

import type { Metadata } from 'next';
import { ThemeProvider } from 'next-themes';
import './globals.css';
import { Toaster } from '@/components/ui/sonner';
import { SettingsProvider } from '@/contexts/SettingsContext';
import { AuthProvider } from '@/contexts/AuthContext';

export const metadata: Metadata = {
  title: {
    default: 'MultiPersonal Chat System | 管理平台',
    template: '%s | MultiPersonal Chat System',
  },
  description:
    '基于本地大语言模型的MultiPersonal Chat System管理平台，支持个性化风格定制和历史记录管理。',
  keywords: [
    'QQ机器人',
    '智能助手',
    '大语言模型',
    'LoRA微调',
    '个性化聊天',
  ],
  authors: [{ name: 'MultiPersonal Chat System团队' }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className={`antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="light">
          <AuthProvider>
            <SettingsProvider>
              {children}
              <Toaster />
            </SettingsProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
