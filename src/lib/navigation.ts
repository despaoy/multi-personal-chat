/** Shared navigation model: presentation only; route/API authorization stays unchanged. */
export type NavigationGroupId = 'workspace' | 'models' | 'research' | 'system';
export type NavigationItem = {
  href: string;
  label: string;
  translationKey: string;
  group: NavigationGroupId;
  adminOnly: boolean;
  keywords: string;
};

export const navigationGroups: { id: NavigationGroupId; label: string; translationKey: string }[] = [
  { id: 'workspace', label: '日常工作', translationKey: 'navigation.workspace' },
  { id: 'models', label: '模型与训练', translationKey: 'navigation.models' },
  { id: 'research', label: '评测与实验', translationKey: 'navigation.research' },
  { id: 'system', label: '系统管理', translationKey: 'navigation.system' },
];

export const navigationItems: NavigationItem[] = [
  { href: '/', label: '工作台', translationKey: 'navigation.home', group: 'workspace', adminOnly: true, keywords: '首页 仪表盘 dashboard home' },
  { href: '/characters', label: '角色与记忆', translationKey: 'navigation.characters', group: 'workspace', adminOnly: true, keywords: '人物 关系 长期记忆 memory persona' },
  { href: '/history', label: '对话记录', translationKey: 'nav.history', group: 'workspace', adminOnly: true, keywords: '聊天 消息 会话 chat history' },
  { href: '/knowledge', label: '知识库', translationKey: 'nav.knowledge', group: 'workspace', adminOnly: true, keywords: 'RAG 文档 检索 knowledge' },
  { href: '/lora', label: 'LoRA 管理', translationKey: 'nav.lora', group: 'models', adminOnly: true, keywords: '模型 适配器 model' },
  { href: '/training', label: '模型训练', translationKey: 'nav.training', group: 'models', adminOnly: true, keywords: '微调 任务 training' },
  { href: '/intent-training', label: '意图训练', translationKey: 'nav.intentTraining', group: 'models', adminOnly: true, keywords: '分类 intent' },
  { href: '/router', label: '模型路由', translationKey: 'nav.router', group: 'models', adminOnly: true, keywords: '分发 router' },
  { href: '/evaluation', label: '效果评测', translationKey: 'nav.evaluation', group: 'research', adminOnly: true, keywords: '测试 评分 evaluation' },
  { href: '/experiments', label: '实验管理', translationKey: 'nav.experiments', group: 'research', adminOnly: true, keywords: '对照 experiments' },
  { href: '/preferences', label: '偏好学习', translationKey: 'nav.preferences', group: 'research', adminOnly: true, keywords: '反馈 对齐 DPO preferences' },
  { href: '/narrative', label: '假想分支', translationKey: 'navigation.narrative', group: 'research', adminOnly: false, keywords: '剧情 分支 实验室 narrative' },
  { href: '/monitor', label: '运行监控', translationKey: 'nav.monitor', group: 'system', adminOnly: true, keywords: '服务 状态 性能 monitor' },
  { href: '/integrations', label: '平台连接', translationKey: 'navigation.integrations', group: 'system', adminOnly: true, keywords: '机器人 QQ 微信 接入 integrations' },
  { href: '/claw', label: 'Claw 控制台', translationKey: 'nav.claw', group: 'system', adminOnly: true, keywords: '终端 terminal claw' },
  { href: '/settings', label: '系统设置', translationKey: 'nav.settings', group: 'system', adminOnly: true, keywords: '配置 settings' },
];

export function isNavigationActive(pathname: string, href: string): boolean {
  return pathname === href || (href !== '/' && pathname.startsWith(`${href}/`));
}

export function visibleNavigation(isAdmin: boolean, query = '', translate = (_key: string, fallback: string) => fallback): NavigationItem[] {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return navigationItems.filter(item => (isAdmin || !item.adminOnly) && terms.every(term =>
    `${item.label} ${translate(item.translationKey, item.label)} ${item.keywords} ${item.href}`.toLocaleLowerCase().includes(term)
  ));
}
