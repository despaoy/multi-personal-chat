/**
 * 统计卡片组件
 *
 * 用于仪表盘展示关键指标，包含图标、标题、数值和可选的趋势指示。
 *
 * @module StatCard
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LucideIcon } from 'lucide-react';

/**
 * StatCard 组件属性
 * @property {string} title - 指标标题（如"今日回复数"）
 * @property {string | number} value - 指标数值
 * @property {string} [change] - 变化幅度文本（可选）
 * @property {'up' | 'down'} [trend] - 趋势方向：up 上升 / down 下降
 * @property {LucideIcon} icon - Lucide 图标组件
 */
interface StatCardProps {
  title: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down';
  icon: LucideIcon;
  compact?: boolean;
}

export function StatCard({ title, value, change, trend, icon: Icon, compact = false }: StatCardProps) {
  return (
    <Card className={compact ? 'min-w-0 gap-3 py-4 sm:gap-6 sm:py-6' : undefined}>
      <CardHeader className={`flex flex-row items-center justify-between pb-2 ${compact ? 'px-3 sm:px-6' : ''}`}>
        <CardTitle className={`${compact ? 'text-xs sm:text-sm' : 'text-sm'} font-medium text-muted-foreground`}>{title}</CardTitle>
        <Icon className={`h-4 w-4 text-muted-foreground ${compact ? 'hidden sm:block' : ''}`} aria-hidden="true" />
      </CardHeader>
      <CardContent className={compact ? 'px-3 sm:px-6' : undefined}>
        <div className={`${compact ? 'break-all text-xl sm:text-2xl' : 'text-2xl'} font-bold`}>{value}</div>
        {change && (
          <p className={`text-xs ${trend === 'up' ? 'text-green-500' : 'text-red-500'}`}>
            {trend === 'up' ? '↑' : '↓'} {change}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
