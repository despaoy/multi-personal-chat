'use client';

import { Home, MessageSquare, Settings, BrainCircuit, Activity, Database, Bot, Zap, User, LogIn, Terminal, Brain, Cable, ClipboardCheck, FlaskConical, Shuffle, Scale, Users } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { useState, useEffect, memo } from 'react';
import { useSettings } from '@/contexts/SettingsContext';
import { useAuth } from '@/contexts/AuthContext';

// 时钟独立组件 - 避免每秒重渲染导致导航 Link 的 RSC 预取被中断
const StatusBar = memo(function StatusBar() {
  const { t, formatTime } = useSettings();
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => setCurrentTime(formatTime(new Date()));
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, [formatTime]);

  return (
    <div className="border-t p-4">
      <div className="rounded-lg bg-muted/50 p-4">
        <div className="flex items-center gap-2 text-sm font-medium">
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
          <span>{t('sidebar.status')}</span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {t('sidebar.lastCheck')}: {currentTime || '--:--:--'}
        </p>
      </div>
    </div>
  );
});

// 导航项独立组件 - 结合 memo 避免时钟引起的级联重渲染
const NavItem = memo(function NavItem({
  href, icon: Icon, name, isActive,
}: {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  name: string;
  isActive: boolean;
}) {
  return (
    <Link
      href={href}
      prefetch={false}
      className={cn(
        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
        isActive
          ? 'bg-sidebar-accent text-sidebar-accent-foreground'
          : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
      )}
    >
      <Icon className="h-5 w-5" />
      {name}
    </Link>
  );
});

// 认证状态区域 - 独立组件，仅在客户端挂载后渲染，避免hydration不匹配
const AuthSection = memo(function AuthSection() {
  const { user } = useAuth();
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  if (!mounted) return null;

  return user ? (
    <div className="mt-auto px-3 py-2 text-xs text-muted-foreground flex items-center gap-2">
      <User className="h-3 w-3" />
      <span>{user.username}</span>
    </div>
  ) : (
    <Link
      href="/login"
      prefetch={false}
      className="mt-auto flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
    >
      <LogIn className="h-4 w-4" />
      <span>登录</span>
    </Link>
  );
});

export function Sidebar() {
  const pathname = usePathname();
  const { t } = useSettings();
  const { user } = useAuth();

  // M3 fix: 为涉及模型/配置/实验管理的敏感页面标记 adminOnly，
  // 非管理员用户在侧栏看不到这些入口。注意：后端已对这些端点强制 admin 校验，
  // 此处仅为 UX 层过滤 + 纵深防御，并非唯一鉴权手段。
  const isAdmin = user?.role === 'admin';

  const navigation: Array<{
    name: string;
    href: string;
    icon: React.ComponentType<{ className?: string }>;
    adminOnly?: boolean;
  }> = [
    { name: t('nav.dashboard'), href: '/', icon: Home, adminOnly: true },
    { name: t('nav.history'), href: '/history', icon: MessageSquare, adminOnly: true },
    { name: t('nav.training'), href: '/training', icon: Zap, adminOnly: true },
    { name: t('nav.lora'), href: '/lora', icon: BrainCircuit, adminOnly: true },
    { name: '角色与记忆', href: '/characters', icon: Users, adminOnly: true },
    { name: t('nav.intentTraining'), href: '/intent-training', icon: Brain, adminOnly: true },
    { name: t('nav.monitor'), href: '/monitor', icon: Activity, adminOnly: true },
    { name: '平台连接', href: '/integrations', icon: Cable, adminOnly: true },
    { name: t('nav.knowledge'), href: '/knowledge', icon: Database, adminOnly: true },
    { name: t('nav.evaluation'), href: '/evaluation', icon: ClipboardCheck, adminOnly: true },
    { name: t('nav.experiments'), href: '/experiments', icon: FlaskConical, adminOnly: true },
    { name: t('nav.router'), href: '/router', icon: Shuffle, adminOnly: true },
    { name: t('nav.preferences'), href: '/preferences', icon: Scale, adminOnly: true },
    { name: t('nav.claw'), href: '/claw', icon: Terminal, adminOnly: true },
    { name: t('nav.settings'), href: '/settings', icon: Settings, adminOnly: true },
  ];

  // 非管理员过滤掉 adminOnly 项
  const visibleNavigation = isAdmin ? navigation : navigation.filter((item) => !item.adminOnly);

  return (
    <div className="flex h-full w-64 flex-col border-r bg-sidebar">
      <div className="flex h-16 items-center border-b px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold">{t('sidebar.title')}</span>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {visibleNavigation.map((item) => (
          <NavItem
            key={item.href}
            href={item.href}
            icon={item.icon}
            name={item.name}
            isActive={pathname === item.href}
          />
        ))}
        <AuthSection />
      </nav>
      <StatusBar />
    </div>
  );
}
