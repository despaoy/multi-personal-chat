'use client';

import { Sidebar } from './Sidebar';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useSettings } from '@/contexts/SettingsContext';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { LogOut, User } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { t } = useSettings();
  const { user, logout } = useAuth();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  const handleLogout = async () => {
    await logout();
    router.replace('/login');
  };

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="flex h-16 items-center justify-between border-b px-6">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-semibold">{t('header.title')}</h1>
          </div>
          <div className="flex items-center gap-4">
            {mounted && user && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <User className="h-4 w-4" />
                <span>{user.username}</span>
                <Button variant="ghost" size="icon" onClick={handleLogout} title="退出登录">
                  <LogOut className="h-4 w-4" />
                </Button>
              </div>
            )}
            <ThemeToggle />
          </div>
        </header>
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
