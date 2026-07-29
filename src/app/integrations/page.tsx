'use client';

import { useCallback, useEffect, useMemo, useState, type ComponentType } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { api, ServiceStatus, SystemConfig } from '@/lib/api';
import { AlertTriangle, Bot, Cable, CheckCircle2, CircleOff, ExternalLink, MessageCircle, RadioTower, RefreshCw, Send, ShieldCheck, Smartphone, WifiOff } from 'lucide-react';
import { toast } from 'sonner';

type PlatformStatus = {
  enabled: boolean;
  status: 'connected' | 'idle' | 'degraded' | 'disabled' | 'running' | 'stopped' | string;
  lastEvent?: string;
};

type AstrBotGatewayStatus = {
  name: string;
  status: 'running' | 'degraded' | 'stopped' | string;
  running: boolean;
  expected: boolean;
  port: number;
};

type StatsMetricsResponse = {
  astrBotGateway?: AstrBotGatewayStatus;
  platformStatus?: Record<string, PlatformStatus>;
};

type PlatformCard = {
  key: string;
  name: string;
  adapter: string;
  configKey: string;
  description: string;
  icon: ComponentType<{ className?: string }>;
};

const platforms: PlatformCard[] = [
  { key: 'qq', name: 'QQ / OneBot', adapter: 'NapCat / OneBot v11', configKey: 'astrbotQQEnabled', description: 'QQ 群聊与私聊，建议由 NapCat 接入 AstrBot。', icon: Bot },
  { key: 'telegram', name: 'Telegram', adapter: 'Telegram Bot API', configKey: 'astrbotTelegramEnabled', description: '通过 Bot Token 接入，适合最先做外网实连验证。', icon: Send },
  { key: 'wecom', name: '企业微信', adapter: 'WeCom App / Bot', configKey: 'astrbotWecomEnabled', description: '生产优先推荐，稳定性和合规性最好。', icon: MessageCircle },
  { key: 'wechat_official', name: '微信公众号', adapter: 'Official Account', configKey: 'astrbotWechatOfficialEnabled', description: '适合公众号消息回调和客服场景。', icon: RadioTower },
  { key: 'wechat_personal', name: '个人微信', adapter: 'GeWeChat / WechatPadPro', configKey: 'astrbotWechatPersonalEnabled', description: '实验能力。账号登录、扫码和凭据保存在 AstrBot 适配器内，本项目只接收标准化消息。', icon: Smartphone },
];

const statusLabel: Record<string, string> = {
  running: '运行中',
  connected: '已连接',
  idle: '等待消息',
  degraded: '异常',
  stopped: '未启动',
  disabled: '未启用',
  connecting: '连接中',
};

const personalWechatSteps = [
  '在本页打开“个人微信”开关，并选择你准备在 AstrBot 中使用的适配器。',
  '打开 AstrBot 面板，在插件市场或平台适配器区域安装/启用个人微信适配器，例如 GeWeChat 或 WechatPadPro。',
  '在 AstrBot 中创建个人微信机器人实例，按适配器要求填写服务地址、token、appId 等信息并扫码登录。',
  '确认 qqchat_gateway 插件已启用，并设置 QQCHAT_BACKEND_URL 指向本项目后端，ASTRBOT_INTEGRATION_TOKEN 与后端一致。',
  '用个人微信给机器人发一条私聊测试消息，回到本页刷新，个人微信状态应从“等待消息”变为“已连接”。',
];

function statusClass(status: string) {
  switch (status) {
    case 'running':
    case 'connected':
      return 'bg-green-500';
    case 'idle':
    case 'connecting':
      return 'bg-yellow-500';
    case 'disabled':
      return 'bg-muted-foreground';
    default:
      return 'bg-red-500';
  }
}

function statusBadgeVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'running' || status === 'connected') return 'default';
  if (status === 'degraded' || status === 'stopped') return 'destructive';
  if (status === 'disabled') return 'outline';
  return 'secondary';
}

function boolFromConfig(config: SystemConfig, key: string, fallback: boolean) {
  const value = config[key];
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') return value.toLowerCase() === 'true';
  return fallback;
}

function strFromConfig(config: SystemConfig, key: string, fallback = '') {
  const value = config[key];
  return value == null ? fallback : String(value);
}

export default function IntegrationsPage() {
  return (
    <AuthGuard requireAdmin>
      <IntegrationsContent />
    </AuthGuard>
  );
}

function IntegrationsContent() {
  const [config, setConfig] = useState<SystemConfig>({});
  const [gateway, setGateway] = useState<AstrBotGatewayStatus | null>(null);
  const [platformStatus, setPlatformStatus] = useState<Record<string, PlatformStatus>>({});
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [savingKey, setSavingKey] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const callbackPath = '/api/integrations/astrbot/messages';
  const backendUrl = strFromConfig(config, 'qqchatBackendUrl', 'http://127.0.0.1:8000');
  const callbackUrl = `${backendUrl.replace(/\/$/, '')}${callbackPath}`;
  // M2 fix: 此前硬编码 http://127.0.0.1:6185，非本机部署场景下"打开 AstrBot 面板"按钮失效。
  // 现从 config 读取 astrbotPanelUrl，默认值保持 127.0.0.1:6185 以兼容现有部署。
  const astrbotPanelUrl = strFromConfig(config, 'astrbotPanelUrl', 'http://127.0.0.1:6185');
  const personalWechatEnabled = boolFromConfig(config, 'astrbotWechatPersonalEnabled', false);
  const personalWechatAdapter = strFromConfig(config, 'astrbotWechatPersonalAdapter', 'gewechat');
  const personalWechatEndpoint = strFromConfig(config, 'astrbotWechatPersonalEndpoint', '');
  const personalWechatNotes = strFromConfig(config, 'astrbotWechatPersonalNotes', '');

  const load = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      const [configData, metricsData, servicesData] = await Promise.all([
        api.getConfig(),
        // M4 fix: 改用 api.getMetrics 走统一 request 助手，
        // 401（token 过期）时会自动清理用户状态并跳转 /login，
        // 而非裸 fetch 那样只抛“获取平台指标失败”让用户停在原地。
        api.getMetrics<StatsMetricsResponse>(),
        api.getServices(),
      ]);
      setConfig(configData.config);
      setGateway(metricsData.astrBotGateway || null);
      setPlatformStatus(metricsData.platformStatus || {});
      setServices(servicesData.services || []);
      setLastUpdated(new Date());
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '加载平台连接状态失败');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const astrbotService = useMemo(
    () => services.find((service) => service.name.toLowerCase().includes('astrbot')) as (ServiceStatus & { port?: number }) | undefined,
    [services]
  );

  const saveConfigPatch = async (patch: SystemConfig, successMessage = '配置已保存') => {
    try {
      const next = { ...config, ...patch };
      const result = await api.updateConfig(next);
      if (result.success) {
        setConfig(result.config);
        toast.success(successMessage);
        await load(true);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '保存失败');
    }
  };

  const updatePlatform = async (key: string, enabled: boolean) => {
    try {
      setSavingKey(key);
      await saveConfigPatch({ [key]: enabled }, '平台开关已保存');
    } finally {
      setSavingKey(null);
    }
  };

  const savePersonalWechatConfig = async () => {
    setSavingKey('wechat_personal_config');
    try {
      await saveConfigPatch({
        astrbotWechatPersonalEnabled: personalWechatEnabled,
        astrbotWechatPersonalAdapter: personalWechatAdapter,
        astrbotWechatPersonalEndpoint: personalWechatEndpoint,
        astrbotWechatPersonalNotes: personalWechatNotes,
      }, '个人微信配置已保存');
    } finally {
      setSavingKey(null);
    }
  };

  const gatewayStatus = gateway?.status || astrbotService?.status || 'stopped';
  const personalStatus = platformStatus.wechat_personal?.status || (personalWechatEnabled ? 'idle' : 'disabled');

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">平台连接</h2>
            <p className="text-muted-foreground">集中查看 AstrBot 网关、QQ、微信、Telegram 等平台连接状态。</p>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdated && <span className="text-xs text-muted-foreground">更新于 {lastUpdated.toLocaleTimeString('zh-CN')}</span>}
            <Button variant="outline" size="sm" onClick={() => load(true)} disabled={refreshing}>
              <RefreshCw className={`mr-2 h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              刷新
            </Button>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Cable className="h-5 w-5" />
                  AstrBot 网关
                </CardTitle>
                <CardDescription>服务器上的多平台消息接入层。</CardDescription>
              </div>
              {loading ? <Skeleton className="h-6 w-20" /> : <Badge variant={statusBadgeVariant(gatewayStatus)}>{statusLabel[gatewayStatus] || gatewayStatus}</Badge>}
            </CardHeader>
            <CardContent className="space-y-5">
              {loading ? (
                <div className="space-y-3">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : (
                <>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-md border p-3">
                      <div className="text-xs text-muted-foreground">状态</div>
                      <div className="mt-2 flex items-center gap-2 font-medium">
                        <span className={`h-2.5 w-2.5 rounded-full ${statusClass(gatewayStatus)}`} />
                        {statusLabel[gatewayStatus] || gatewayStatus}
                      </div>
                    </div>
                    <div className="rounded-md border p-3">
                      <div className="text-xs text-muted-foreground">管理面板</div>
                      <div className="mt-2 font-medium">:{gateway?.port || astrbotService?.port || 6185}</div>
                    </div>
                    <div className="rounded-md border p-3">
                      <div className="text-xs text-muted-foreground">后端鉴权</div>
                      <div className="mt-2 flex items-center gap-2 font-medium">
                        <ShieldCheck className="h-4 w-4 text-green-600" /> Token Header
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>后端回调地址</Label>
                    <Input value={callbackUrl} readOnly />
                    <p className="text-xs text-muted-foreground">AstrBot 插件把标准化消息 POST 到该地址，并携带 X-Integration-Token。</p>
                  </div>

                  <div className="rounded-md border bg-muted/30 p-4 text-sm leading-6">
                    <div className="font-medium">实连顺序建议</div>
                    <p className="text-muted-foreground">先打开 AstrBot 面板创建机器人，再接 Telegram 或 QQ OneBot。微信建议优先企业微信/公众号；个人微信请先确认账号与适配器风险。</p>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>快速入口</CardTitle>
              <CardDescription>用于登录 AstrBot 与检查当前服务链路。</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <a href={astrbotPanelUrl} target="_blank" rel="noreferrer">
                <Button className="w-full justify-between" variant="outline">
                  打开 AstrBot 面板
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </a>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-md border p-3"><div className="text-muted-foreground">前端</div><div className="mt-1 font-medium">5000</div></div>
                <div className="rounded-md border p-3"><div className="text-muted-foreground">后端</div><div className="mt-1 font-medium">8000</div></div>
                <div className="rounded-md border p-3"><div className="text-muted-foreground">vLLM</div><div className="mt-1 font-medium">8001</div></div>
                <div className="rounded-md border p-3"><div className="text-muted-foreground">AstrBot</div><div className="mt-1 font-medium">6185</div></div>
              </div>
              <div className="rounded-md border p-3 text-xs text-muted-foreground">
                QQ/微信登录二维码会在 AstrBot 或对应适配器页面完成，本页只做状态聚合和回调配置展示。
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="border-yellow-200 bg-yellow-50/40 dark:border-yellow-900 dark:bg-yellow-950/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Smartphone className="h-5 w-5" />
              个人微信连接
              <Badge variant={statusBadgeVariant(personalStatus)}>{statusLabel[personalStatus] || personalStatus}</Badge>
            </CardTitle>
            <CardDescription>通过 AstrBot 的个人微信适配器接入。本项目不保存微信登录凭据，只接收 AstrBot 转发后的标准化消息。</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex items-start gap-3 rounded-md border border-yellow-300 bg-background/80 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600" />
              <div className="space-y-1">
                <div className="font-medium">个人微信属于实验接入</div>
                <p className="text-muted-foreground">它通常依赖第三方适配器或桥接服务，稳定性和账号风控不可完全保证。生产环境优先使用企业微信或微信公众号。</p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>适配器类型</Label>
                <Select value={personalWechatAdapter} onValueChange={(value) => setConfig((prev) => ({ ...prev, astrbotWechatPersonalAdapter: value }))}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择个人微信适配器" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gewechat">GeWeChat</SelectItem>
                    <SelectItem value="wechatpadpro">WechatPadPro</SelectItem>
                    <SelectItem value="other">其他 AstrBot 个人微信适配器</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>适配器服务地址或备注</Label>
                <Input
                  value={personalWechatEndpoint}
                  onChange={(event) => setConfig((prev) => ({ ...prev, astrbotWechatPersonalEndpoint: event.target.value }))}
                  placeholder="例如 http://127.0.0.1:2531 或留空"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>本地备注</Label>
              <Textarea
                value={personalWechatNotes}
                onChange={(event) => setConfig((prev) => ({ ...prev, astrbotWechatPersonalNotes: event.target.value }))}
                placeholder="记录 AstrBot 机器人名称、适配器实例名、登录账号用途等非敏感信息。不要填写 token、cookie 或扫码登录凭据。"
                rows={3}
              />
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <Switch checked={personalWechatEnabled} onCheckedChange={(checked) => setConfig((prev) => ({ ...prev, astrbotWechatPersonalEnabled: checked }))} />
                <div>
                  <div className="text-sm font-medium">允许后端处理个人微信消息</div>
                  <div className="text-xs text-muted-foreground">关闭后，即使 AstrBot 收到个人微信消息，后端也会按平台开关拒绝回复。</div>
                </div>
              </div>
              <Button onClick={savePersonalWechatConfig} disabled={savingKey === 'wechat_personal_config'}>
                保存个人微信配置
              </Button>
            </div>

            <div className="grid gap-3 md:grid-cols-5">
              {personalWechatSteps.map((step, index) => (
                <div key={step} className="rounded-md border bg-background p-3 text-sm">
                  <div className="mb-2 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">{index + 1}</div>
                  <p className="text-muted-foreground">{step}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {platforms.map((platform) => {
            const Icon = platform.icon;
            const configEnabled = boolFromConfig(config, platform.configKey, platform.key !== 'wechat_personal');
            const status = platformStatus[platform.key]?.status || (configEnabled ? 'idle' : 'disabled');
            const enabled = platformStatus[platform.key]?.enabled ?? configEnabled;
            return (
              <Card key={platform.key} className="overflow-hidden">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Icon className="h-5 w-5" />
                        {platform.name}
                      </CardTitle>
                      <CardDescription>{platform.adapter}</CardDescription>
                    </div>
                    <Switch
                      checked={enabled}
                      disabled={loading || savingKey === platform.configKey}
                      onCheckedChange={(checked) => updatePlatform(platform.configKey, checked)}
                    />
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Badge variant={statusBadgeVariant(status)}>{statusLabel[status] || status}</Badge>
                    <span className="text-xs text-muted-foreground">{platformStatus[platform.key]?.lastEvent || '暂无消息事件'}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{platform.description}</p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    {status === 'connected' || status === 'running' ? <CheckCircle2 className="h-4 w-4 text-green-600" /> : status === 'disabled' ? <CircleOff className="h-4 w-4" /> : <WifiOff className="h-4 w-4 text-yellow-600" />}
                    {status === 'connected' ? '后端最近收到该平台消息。' : status === 'disabled' ? '该平台已关闭自动接入。' : '等待 AstrBot 适配器连接或首条消息事件。'}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </AppLayout>
  );
}
