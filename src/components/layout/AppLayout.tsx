'use client';

import { Sidebar } from './Sidebar';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useSettings } from '@/contexts/SettingsContext';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { ChevronRight, LogOut, Menu } from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { isNavigationActive, navigationGroups, visibleNavigation } from '@/lib/navigation';

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { t } = useSettings();
  const { user, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [navigationOpen, setNavigationOpen] = useState(false);
  const currentPage = visibleNavigation(user?.role === 'admin').find(item => isNavigationActive(pathname, item.href));
  const currentGroup = navigationGroups.find(group => group.id === currentPage?.group);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  const handleLogout = async () => {
    await logout();
    router.replace('/login');
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-background focus:p-3">{t('navigation.skip', '跳到页面内容')}</a>
      <div className="hidden w-56 shrink-0 border-r md:block"><Sidebar /></div>
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-16 shrink-0 items-center justify-between gap-3 border-b px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-2">
            <Dialog open={navigationOpen} onOpenChange={setNavigationOpen}>
              <DialogTrigger asChild><Button variant="ghost" size="icon" className="shrink-0 md:hidden" aria-label={t('navigation.open', '打开导航')}><Menu className="h-5 w-5" /></Button></DialogTrigger>
              <DialogContent className="inset-y-0 left-0 top-0 h-dvh w-[min(20rem,90vw)] max-w-none translate-x-0 translate-y-0 gap-0 rounded-none border-y-0 border-l-0 p-0 sm:max-w-none">
                <DialogTitle className="sr-only">{t('navigation.main', '主导航')}</DialogTitle>
                <DialogDescription className="sr-only">{t('navigation.hint', '常用功能在上方，其余按需展开。')}</DialogDescription>
                <Sidebar onNavigate={() => setNavigationOpen(false)} />
              </DialogContent>
            </Dialog>
            {currentGroup && <span className="hidden items-center gap-2 text-sm text-muted-foreground sm:flex">{t(currentGroup.translationKey, currentGroup.label)}<ChevronRight className="h-3.5 w-3.5" aria-hidden="true" /></span>}
            <span className="truncate text-sm font-medium">{currentPage ? t(currentPage.translationKey, currentPage.label) : t('header.title')}</span>
          </div>
          <div className="flex shrink-0 items-center gap-1 sm:gap-3">
            {mounted && user && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="hidden max-w-32 truncate sm:inline">{user.username}</span>
                <Button variant="ghost" size="icon" onClick={handleLogout} title={t('navigation.logout', '退出登录')} aria-label={t('navigation.logout', '退出登录')}>
                  <LogOut className="h-4 w-4" />
                </Button>
              </div>
            )}
            <ThemeToggle />
          </div>
        </header>
        <main id="main-content" tabIndex={-1} className="min-h-0 flex-1 overflow-auto p-4 outline-none sm:p-6 lg:p-8">
          {children}
        </main>
      </div>
    </div>
  );
}
