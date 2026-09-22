/**
 * 国际化翻译字典
 *
 * 支持简体中文(zh-CN)、繁体中文(zh-TW)、英文(en)
 * 每个键对应一个翻译字符串
 */

export type Locale = 'zh-CN' | 'zh-TW' | 'en';

const translations: Record<Locale, Record<string, string>> = {
  'zh-CN': {
    'navigation.workspace': '日常工作',
    'navigation.models': '模型与训练',
    'navigation.research': '评测与实验',
    'navigation.system': '系统管理',
    'navigation.home': '工作台',
    'navigation.characters': '角色与记忆',
    'navigation.narrative': '假想分支',
    'navigation.integrations': '平台连接',
    'navigation.search': '查找功能…',
    'navigation.clear': '清空搜索',
    'navigation.main': '主导航',
    'navigation.noResults': '没有匹配的功能，试试“记忆”或“训练”。',
    'navigation.hint': '常用功能在上方，其余按需展开。',
    'navigation.open': '打开导航',
    'navigation.skip': '跳到页面内容',
    'navigation.logout': '退出登录',
    // 侧边栏
    'sidebar.title': 'QQ 智能助手',
    'nav.dashboard': '仪表盘',
    'nav.history': '历史记录',
    'nav.training': 'LoRA训练',
    'nav.lora': 'LoRA管理',
    'nav.intentTraining': '意图训练',
    'nav.monitor': '系统监控',
    'nav.knowledge': '知识库',
    'nav.evaluation': '评估',
    'nav.experiments': '实验',
    'nav.router': '路由',
    'nav.preferences': '偏好',
    'nav.claw': 'Claw工具',
    'nav.settings': '设置',
    'sidebar.status': '系统运行中',
    'sidebar.lastCheck': '最后检查',

    // 顶部栏
    'header.title': '智能助手管理平台',

    // 设置页
    'settings.title': '设置',
    'settings.description': '配置您的智能助手和系统参数',
    'settings.tab.general': '通用设置',
    'settings.tab.bot': '机器人设置',
    'settings.tab.model': '模型设置',
    'settings.tab.notifications': '通知设置',
    'settings.tab.security': '安全设置',

    // 通用设置
    'settings.general.title': '基本信息',
    'settings.general.description': '配置系统的基本信息',
    'settings.general.systemName': '系统名称',
    'settings.general.language': '语言',
    'settings.general.language.zhCN': '简体中文',
    'settings.general.language.zhTW': '繁体中文',
    'settings.general.language.en': 'English',
    'settings.general.timezone': '时区',
    'settings.general.timezone.asiaShanghai': '中国标准时间 (UTC+8)',
    'settings.general.timezone.asiaTokyo': '日本标准时间 (UTC+9)',
    'settings.general.timezone.americaNewYork': '东部时间 (UTC-5)',

    // 机器人设置
    'settings.bot.title': '机器人配置',
    'settings.bot.description': '配置QQ机器人的行为参数',
    'settings.bot.autoReply': '自动回复',
    'settings.bot.autoReplyDesc': '启用后会自动回复@消息',
    'settings.bot.groupReply': '群聊回复',
    'settings.bot.groupReplyDesc': '在群聊中回复@消息',
    'settings.bot.privateReply': '私聊回复',
    'settings.bot.privateReplyDesc': '在私聊中回复消息',
    'settings.bot.replyDelay': '基础延迟 (秒)',
    'settings.bot.replyDelayDesc': '实际延迟 = 基础延迟 + 动态打字时间 + 随机抖动',
    'settings.bot.defaultTemplate': '默认回复模板',
    'settings.bot.defaultTemplatePlaceholder': '设置默认的回复模板...',

    // 模型设置
    'settings.model.title': '模型配置',
    'settings.model.description': '配置大语言模型参数',
    'settings.model.baseModel': '基座模型',
    'settings.model.temperature': '温度 (Temperature)',
    'settings.model.temperatureDesc': '越高越随机，越低越确定',
    'settings.model.maxLength': '最大生成长度',
    'settings.model.contextWindow': '上下文窗口',
    'settings.model.useKnowledge': '使用知识库检索',
    'settings.model.useKnowledgeDesc': '回复前先检索相关知识',
    'settings.model.provider': '模型提供商',
    'settings.model.providerDesc': '选择模型服务提供商',
    'settings.model.apiBaseUrl': 'API 地址',
    'settings.model.apiBaseUrlDesc': 'OpenAI 兼容 API 的基础地址',
    'settings.model.apiKey': 'API Key',
    'settings.model.apiKeyDesc': 'API 密钥',
    'settings.model.apiModel': 'API 模型名',
    'settings.model.apiModelDesc': '远程调用的模型名称',

    // 通知设置
    'settings.notifications.title': '通知配置',
    'settings.notifications.description': '配置系统通知和告警',
    'settings.notifications.errorAlert': '错误告警',
    'settings.notifications.errorAlertDesc': '系统出错时发送通知',
    'settings.notifications.dailyStats': '每日统计',
    'settings.notifications.dailyStatsDesc': '每日发送运行统计报告',
    'settings.notifications.anomalyDetection': '异常检测',
    'settings.notifications.anomalyDetectionDesc': '检测到异常行为时告警',

    // 安全设置
    'settings.security.title': '安全配置',
    'settings.security.description': '配置内容安全和权限管理',
    'settings.security.contentFilter': '敏感词过滤',
    'settings.security.contentFilterDesc': '过滤输入和输出中的敏感内容',
    'settings.security.contentReview': '内容审核',
    'settings.security.contentReviewDesc': '对生成内容进行安全审核',
    'settings.security.adminQQ': '管理员QQ号',
    'settings.security.adminQQPlaceholder': '输入管理员QQ号，每行一个',
    'settings.security.adminQQDesc': '管理员可以通过私聊控制机器人',

    // 设置页底部
    'settings.loaded': '已加载',
    'settings.items': '项配置',
    'settings.save': '保存设置',
    'settings.saving': '保存中...',
    'settings.saved': '设置已保存',
    'settings.saveFailed': '保存设置失败',
    'settings.loadFailed': '获取配置失败',
    'settings.retry': '重试',
    'settings.loadError': '加载失败',

    // 监控页
    'monitor.title': '系统监控',
    'monitor.description': '实时监控系统运行状态和资源使用情况',
    'monitor.cpuUsage': 'CPU 使用率',
    'monitor.gpuMemory': 'GPU 内存',
    'monitor.memoryUsage': '内存使用',
    'monitor.diskSpace': '磁盘空间',
    'monitor.services': '服务状态',
    'monitor.overview': '运行概况',
    'monitor.todayReplies': '今日回复数',
    'monitor.avgResponseTime': '平均响应时间',
    'monitor.activeSessions': '活跃会话',
    'monitor.modelLoad': '模型负载',
    'monitor.updatedAt': '更新于',
    'monitor.loadFailed': '加载失败',

    // 服务状态
    'status.running': '运行中',
    'status.connecting': '连接中',
    'status.stopped': '已停止',
  },

  'zh-TW': {
    'navigation.workspace': '日常工作',
    'navigation.models': '模型與訓練',
    'navigation.research': '評測與實驗',
    'navigation.system': '系統管理',
    'navigation.home': '工作台',
    'navigation.characters': '角色與記憶',
    'navigation.narrative': '假想分支',
    'navigation.integrations': '平台連接',
    'navigation.search': '尋找功能…',
    'navigation.clear': '清除搜尋',
    'navigation.main': '主導覽',
    'navigation.noResults': '沒有符合的功能，試試「記憶」或「訓練」。',
    'navigation.hint': '常用功能在上方，其餘按需展開。',
    'navigation.open': '開啟導覽',
    'navigation.skip': '跳到頁面內容',
    'navigation.logout': '登出',
    // 側邊欄
    'sidebar.title': 'QQ 智慧助手',
    'nav.dashboard': '儀表盤',
    'nav.history': '歷史記錄',
    'nav.training': 'LoRA訓練',
    'nav.lora': 'LoRA管理',
    'nav.intentTraining': '意圖訓練',
    'nav.monitor': '系統監控',
    'nav.knowledge': '知識庫',
    'nav.evaluation': '評估',
    'nav.experiments': '實驗',
    'nav.router': '路由',
    'nav.preferences': '偏好',
    'nav.claw': 'Claw工具',
    'nav.settings': '設定',
    'sidebar.status': '系統運行中',
    'sidebar.lastCheck': '最後檢查',

    // 頂部欄
    'header.title': '智慧助手管理平台',

    // 設定頁
    'settings.title': '設定',
    'settings.description': '配置您的智慧助手和系統參數',
    'settings.tab.general': '通用設定',
    'settings.tab.bot': '機器人設定',
    'settings.tab.model': '模型設定',
    'settings.tab.notifications': '通知設定',
    'settings.tab.security': '安全設定',

    // 通用設定
    'settings.general.title': '基本資訊',
    'settings.general.description': '配置系統的基本資訊',
    'settings.general.systemName': '系統名稱',
    'settings.general.language': '語言',
    'settings.general.language.zhCN': '简体中文',
    'settings.general.language.zhTW': '繁體中文',
    'settings.general.language.en': 'English',
    'settings.general.timezone': '時區',
    'settings.general.timezone.asiaShanghai': '中國標準時間 (UTC+8)',
    'settings.general.timezone.asiaTokyo': '日本標準時間 (UTC+9)',
    'settings.general.timezone.americaNewYork': '東部時間 (UTC-5)',

    // 機器人設定
    'settings.bot.title': '機器人配置',
    'settings.bot.description': '配置QQ機器人的行為參數',
    'settings.bot.autoReply': '自動回覆',
    'settings.bot.autoReplyDesc': '啟用後會自動回覆@訊息',
    'settings.bot.groupReply': '群聊回覆',
    'settings.bot.groupReplyDesc': '在群聊中回覆@訊息',
    'settings.bot.privateReply': '私聊回覆',
    'settings.bot.privateReplyDesc': '在私聊中回覆訊息',
    'settings.bot.replyDelay': '基礎延遲 (秒)',
    'settings.bot.replyDelayDesc': '實際延遲 = 基礎延遲 + 動態打字時間 + 隨機抖動',
    'settings.bot.defaultTemplate': '預設回覆範本',
    'settings.bot.defaultTemplatePlaceholder': '設定預設的回覆範本...',

    // 模型設定
    'settings.model.title': '模型配置',
    'settings.model.description': '配置大語言模型參數',
    'settings.model.baseModel': '基座模型',
    'settings.model.temperature': '溫度 (Temperature)',
    'settings.model.temperatureDesc': '越高越隨機，越低越確定',
    'settings.model.maxLength': '最大生成长度',
    'settings.model.contextWindow': '上下文視窗',
    'settings.model.useKnowledge': '使用知識庫檢索',
    'settings.model.useKnowledgeDesc': '回覆前先檢索相關知識',
    'settings.model.provider': '模型提供商',
    'settings.model.providerDesc': '選擇模型服務提供商',
    'settings.model.apiBaseUrl': 'API 地址',
    'settings.model.apiBaseUrlDesc': 'OpenAI 相容 API 的基礎地址',
    'settings.model.apiKey': 'API Key',
    'settings.model.apiKeyDesc': 'API 密鑰',
    'settings.model.apiModel': 'API 模型名',
    'settings.model.apiModelDesc': '遠端調用的模型名稱',

    // 通知設定
    'settings.notifications.title': '通知配置',
    'settings.notifications.description': '配置系統通知和警報',
    'settings.notifications.errorAlert': '錯誤警報',
    'settings.notifications.errorAlertDesc': '系統出錯時發送通知',
    'settings.notifications.dailyStats': '每日統計',
    'settings.notifications.dailyStatsDesc': '每日發送運行統計報告',
    'settings.notifications.anomalyDetection': '異常檢測',
    'settings.notifications.anomalyDetectionDesc': '檢測到異常行為時警報',

    // 安全設定
    'settings.security.title': '安全配置',
    'settings.security.description': '配置內容安全和權限管理',
    'settings.security.contentFilter': '敏感詞過濾',
    'settings.security.contentFilterDesc': '過濾輸入和輸出中的敏感內容',
    'settings.security.contentReview': '內容審核',
    'settings.security.contentReviewDesc': '對生成內容進行安全審核',
    'settings.security.adminQQ': '管理員QQ號',
    'settings.security.adminQQPlaceholder': '輸入管理員QQ號，每行一個',
    'settings.security.adminQQDesc': '管理員可以透過私聊控制機器人',

    // 設定頁底部
    'settings.loaded': '已載入',
    'settings.items': '項配置',
    'settings.save': '儲存設定',
    'settings.saving': '儲存中...',
    'settings.saved': '設定已儲存',
    'settings.saveFailed': '儲存設定失敗',
    'settings.loadFailed': '獲取配置失敗',
    'settings.retry': '重試',
    'settings.loadError': '載入失敗',

    // 監控頁
    'monitor.title': '系統監控',
    'monitor.description': '即時監控系統運行狀態和資源使用情況',
    'monitor.cpuUsage': 'CPU 使用率',
    'monitor.gpuMemory': 'GPU 記憶體',
    'monitor.memoryUsage': '記憶體使用',
    'monitor.diskSpace': '磁碟空間',
    'monitor.services': '服務狀態',
    'monitor.overview': '運行概況',
    'monitor.todayReplies': '今日回覆數',
    'monitor.avgResponseTime': '平均回應時間',
    'monitor.activeSessions': '活躍會話',
    'monitor.modelLoad': '模型負載',
    'monitor.updatedAt': '更新於',
    'monitor.loadFailed': '載入失敗',

    // 服務狀態
    'status.running': '運行中',
    'status.connecting': '連接中',
    'status.stopped': '已停止',
  },

  'en': {
    'navigation.workspace': 'Workspace',
    'navigation.models': 'Models & training',
    'navigation.research': 'Evaluation & experiments',
    'navigation.system': 'Administration',
    'navigation.home': 'Overview',
    'navigation.characters': 'Characters & memory',
    'navigation.narrative': 'Narrative branches',
    'navigation.integrations': 'Integrations',
    'navigation.search': 'Find a feature…',
    'navigation.clear': 'Clear search',
    'navigation.main': 'Main navigation',
    'navigation.noResults': 'No matching features. Try “memory” or “training”.',
    'navigation.hint': 'Everyday tools first. Expand other groups as needed.',
    'navigation.open': 'Open navigation',
    'navigation.skip': 'Skip to content',
    'navigation.logout': 'Sign out',
    // Sidebar
    'sidebar.title': 'QQ Smart Assistant',
    'nav.dashboard': 'Dashboard',
    'nav.history': 'History',
    'nav.training': 'LoRA Training',
    'nav.lora': 'LoRA Models',
    'nav.intentTraining': 'Intent Training',
    'nav.monitor': 'Monitor',
    'nav.knowledge': 'Knowledge',
    'nav.evaluation': 'Evaluation',
    'nav.experiments': 'Experiments',
    'nav.router': 'Router',
    'nav.preferences': 'Preferences',
    'nav.claw': 'Claw Tools',
    'nav.settings': 'Settings',
    'sidebar.status': 'System Running',
    'sidebar.lastCheck': 'Last Check',

    // Header
    'header.title': 'Smart Assistant Platform',

    // Settings page
    'settings.title': 'Settings',
    'settings.description': 'Configure your smart assistant and system parameters',
    'settings.tab.general': 'General',
    'settings.tab.bot': 'Bot',
    'settings.tab.model': 'Model',
    'settings.tab.notifications': 'Notifications',
    'settings.tab.security': 'Security',

    // General settings
    'settings.general.title': 'Basic Information',
    'settings.general.description': 'Configure basic system information',
    'settings.general.systemName': 'System Name',
    'settings.general.language': 'Language',
    'settings.general.language.zhCN': 'Simplified Chinese',
    'settings.general.language.zhTW': 'Traditional Chinese',
    'settings.general.language.en': 'English',
    'settings.general.timezone': 'Timezone',
    'settings.general.timezone.asiaShanghai': 'China Standard Time (UTC+8)',
    'settings.general.timezone.asiaTokyo': 'Japan Standard Time (UTC+9)',
    'settings.general.timezone.americaNewYork': 'Eastern Time (UTC-5)',

    // Bot settings
    'settings.bot.title': 'Bot Configuration',
    'settings.bot.description': 'Configure QQ bot behavior parameters',
    'settings.bot.autoReply': 'Auto Reply',
    'settings.bot.autoReplyDesc': 'Automatically reply to @mentions when enabled',
    'settings.bot.groupReply': 'Group Reply',
    'settings.bot.groupReplyDesc': 'Reply to @mentions in group chats',
    'settings.bot.privateReply': 'Private Reply',
    'settings.bot.privateReplyDesc': 'Reply to private messages',
    'settings.bot.replyDelay': 'Base Delay (seconds)',
    'settings.bot.replyDelayDesc': 'Actual delay = base + dynamic typing time + random jitter',
    'settings.bot.defaultTemplate': 'Default Reply Template',
    'settings.bot.defaultTemplatePlaceholder': 'Set default reply template...',

    // Model settings
    'settings.model.title': 'Model Configuration',
    'settings.model.description': 'Configure LLM parameters',
    'settings.model.baseModel': 'Base Model',
    'settings.model.temperature': 'Temperature',
    'settings.model.temperatureDesc': 'Higher = more random, Lower = more deterministic',
    'settings.model.maxLength': 'Max Generation Length',
    'settings.model.contextWindow': 'Context Window',
    'settings.model.useKnowledge': 'Use Knowledge Base',
    'settings.model.useKnowledgeDesc': 'Search relevant knowledge before replying',
    'settings.model.provider': 'Model Provider',
    'settings.model.providerDesc': 'Select model service provider',
    'settings.model.apiBaseUrl': 'API Base URL',
    'settings.model.apiBaseUrlDesc': 'OpenAI-compatible API base URL',
    'settings.model.apiKey': 'API Key',
    'settings.model.apiKeyDesc': 'API key for authentication',
    'settings.model.apiModel': 'API Model Name',
    'settings.model.apiModelDesc': 'Remote model name to call',

    // Notification settings
    'settings.notifications.title': 'Notification Configuration',
    'settings.notifications.description': 'Configure system notifications and alerts',
    'settings.notifications.errorAlert': 'Error Alerts',
    'settings.notifications.errorAlertDesc': 'Send notifications on system errors',
    'settings.notifications.dailyStats': 'Daily Statistics',
    'settings.notifications.dailyStatsDesc': 'Send daily statistics report',
    'settings.notifications.anomalyDetection': 'Anomaly Detection',
    'settings.notifications.anomalyDetectionDesc': 'Alert when anomalous behavior detected',

    // Security settings
    'settings.security.title': 'Security Configuration',
    'settings.security.description': 'Configure content security and access control',
    'settings.security.contentFilter': 'Content Filter',
    'settings.security.contentFilterDesc': 'Filter sensitive content in input and output',
    'settings.security.contentReview': 'Content Review',
    'settings.security.contentReviewDesc': 'Review generated content for safety',
    'settings.security.adminQQ': 'Admin QQ Numbers',
    'settings.security.adminQQPlaceholder': 'Enter admin QQ numbers, one per line',
    'settings.security.adminQQDesc': 'Admins can control the bot via private chat',

    // Settings page footer
    'settings.loaded': 'Loaded',
    'settings.items': 'config items',
    'settings.save': 'Save Settings',
    'settings.saving': 'Saving...',
    'settings.saved': 'Settings saved',
    'settings.saveFailed': 'Failed to save settings',
    'settings.loadFailed': 'Failed to load config',
    'settings.retry': 'Retry',
    'settings.loadError': 'Load Failed',

    // Monitor page
    'monitor.title': 'System Monitor',
    'monitor.description': 'Real-time monitoring of system status and resource usage',
    'monitor.cpuUsage': 'CPU Usage',
    'monitor.gpuMemory': 'GPU Memory',
    'monitor.memoryUsage': 'Memory Usage',
    'monitor.diskSpace': 'Disk Space',
    'monitor.services': 'Service Status',
    'monitor.overview': 'Overview',
    'monitor.todayReplies': 'Today Replies',
    'monitor.avgResponseTime': 'Avg Response Time',
    'monitor.activeSessions': 'Active Sessions',
    'monitor.modelLoad': 'Model Load',
    'monitor.updatedAt': 'Updated at',
    'monitor.loadFailed': 'Load Failed',

    // Service status
    'status.running': 'Running',
    'status.connecting': 'Connecting',
    'status.stopped': 'Stopped',
  },
};

/**
 * 获取翻译文本
 * @param locale - 语言代码
 * @param key - 翻译键
 * @param fallback - 回退文本（默认为key本身）
 */
export function t(locale: Locale, key: string, fallback?: string): string {
  return translations[locale]?.[key] ?? fallback ?? key;
}

/**
 * 获取所有支持的语言列表
 */
export function getSupportedLocales(): { value: Locale; label: string }[] {
  return [
    { value: 'zh-CN', label: '简体中文' },
    { value: 'zh-TW', label: '繁體中文' },
    { value: 'en', label: 'English' },
  ];
}

/**
 * 获取所有支持的时区列表
 */
export function getSupportedTimezones(): { value: string; labelKey: string }[] {
  return [
    { value: 'Asia/Shanghai', labelKey: 'settings.general.timezone.asiaShanghai' },
    { value: 'Asia/Tokyo', labelKey: 'settings.general.timezone.asiaTokyo' },
    { value: 'America/New_York', labelKey: 'settings.general.timezone.americaNewYork' },
  ];
}
