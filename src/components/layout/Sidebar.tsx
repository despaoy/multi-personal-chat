'use client';

import { useId, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Bot, ChevronDown, Search, X } from 'lucide-react';
import { useSettings } from '@/contexts/SettingsContext';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';
import { navigationGroups, visibleNavigation, isNavigationActive, type NavigationItem } from '@/lib/navigation';
import { Input } from '@/components/ui/input';

function NavigationLink({ item, pathname, onNavigate }: { item: NavigationItem; pathname: string; onNavigate?: () => void }) {
  const { t } = useSettings();
  const active = isNavigationActive(pathname, item.href);
  return (
    <Link href={item.href} prefetch={false} onClick={onNavigate} aria-current={active ? 'page' : undefined}
      className={cn('flex min-h-10 items-center rounded-lg px-3 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active ? 'bg-primary/10 font-medium text-primary' : 'text-muted-foreground hover:bg-sidebar-accent hover:text-foreground')}>
      {t(item.translationKey, item.label)}
    </Link>
  );
}

function NavigationGroup({ label, items, pathname, searching, onNavigate }: {
  label: string; items: NavigationItem[]; pathname: string; searching: boolean; onNavigate?: () => void;
}) {
  const [expanded, setExpanded] = useState(items.some(item => isNavigationActive(pathname, item.href)));
  const open = searching || expanded;
  const id = useId();
  return (
    <section>
      <button type="button" onClick={() => setExpanded(value => !value)} aria-expanded={open} aria-controls={id}
        className="flex min-h-11 w-full items-center justify-between rounded-lg px-3 text-sm font-medium hover:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        {label}<ChevronDown aria-hidden="true" className={cn('h-4 w-4 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>
      <div id={id} hidden={!open} className="ml-3 space-y-0.5 border-l pl-2">
        {items.map(item => <NavigationLink key={item.href} item={item} pathname={pathname} onNavigate={onNavigate} />)}
      </div>
    </section>
  );
}

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  const { t } = useSettings();
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const items = visibleNavigation(user?.role === 'admin', query, t);
  return (
    <aside className="flex h-full w-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex h-16 shrink-0 items-center gap-3 px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary text-primary-foreground"><Bot className="h-5 w-5" aria-hidden="true" /></div>
        <span className="text-sm font-semibold tracking-tight">{t('sidebar.title')}</span>
      </div>
      <div className="px-3 pb-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <Input value={query} onChange={event => setQuery(event.target.value)}
            placeholder={t('navigation.search', '查找功能…')} aria-label={t('navigation.search', '查找功能…')}
            className="h-10 border-transparent bg-muted/60 pl-9 pr-9 shadow-none focus-visible:border-input" />
          {query && <button type="button" aria-label={t('navigation.clear', '清空搜索')} onClick={() => setQuery('')} className="absolute right-1 top-1 rounded-md p-2 hover:bg-accent"><X className="h-4 w-4" /></button>}
        </div>
      </div>
      <nav aria-label={t('navigation.main', '主导航')} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 pb-6">
        {navigationGroups.map(group => {
          const entries = items.filter(item => item.group === group.id);
          if (!entries.length) return null;
          if (group.id === 'workspace') return <section key={group.id} className="space-y-1">
            <p className="px-3 pb-1 text-xs text-muted-foreground">{t(group.translationKey, group.label)}</p>
            {entries.map(item => <NavigationLink key={item.href} item={item} pathname={pathname} onNavigate={onNavigate} />)}
          </section>;
          return <NavigationGroup key={`${group.id}:${pathname}`} label={t(group.translationKey, group.label)} items={entries} pathname={pathname} searching={!!query.trim()} onNavigate={onNavigate} />;
        })}
        {!items.length && <p role="status" className="px-3 py-6 text-sm text-muted-foreground">{t('navigation.noResults', '没有匹配的功能，试试“记忆”或“训练”。')}</p>}
      </nav>
      <div className="border-t px-6 py-4 text-xs text-muted-foreground">{t('navigation.hint', '常用功能在上方，其余按需展开。')}</div>
    </aside>
  );
}
