'use client';

import { useState, useEffect, useCallback } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Settings, Bot, Database, Bell, Shield, Save, RefreshCw, AlertCircle, CheckCircle2, Zap } from 'lucide-react';
import { api, SystemConfig, AvailableModel } from '@/lib/api';
import { toast } from 'sonner';
import { useSettings } from '@/contexts/SettingsContext';
import { getSupportedLocales, getSupportedTimezones } from '@/lib/i18n';

export default function SettingsPage() {
  return (
    <AuthGuard requireAdmin>
      <SettingsContent />
    </AuthGuard>
  );
}

function SettingsContent() {
  const { t, syncSettings } = useSettings();
  const [config, setConfig] = useState<SystemConfig>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [configData, modelsData] = await Promise.all([
        api.getConfig(),
        api.getModels(),
      ]);
      setConfig(configData.config);
      setAvailableModels(modelsData.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('settings.loadFailed'));
      console.error('Failed to fetch config:', err);
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {

    fetchConfig();
  }, [fetchConfig]);

  const updateField = (key: string, value: string | number | boolean) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const result = await api.updateConfig(config);
      if (result.success) {
        setConfig(result.config);
        // 如果模型提供商变更，同步切换后端提供商
        if (config.modelProvider) {
          try {
            // H2 fix: 此前使用裸 fetch，不处理 401 重定向。
            // 现通过 api 助手调用，token 过期时自动跳转登录页。
            await api.setModelProvider(String(config.modelProvider));
          } catch (err) {
            // 提供商切换失败不阻塞保存，但必须告知用户
            toast.warning(
              `配置已保存，但模型提供商切换失败：${err instanceof Error ? err.message : '未知错误'}`
            );
          }
        }
        // 立即刷新全局设置，使语言/时区变更即时生效
        syncSettings(result.config);
        toast.success(t('settings.saved'));
      }
    } catch (err) {
      toast.error(t('settings.saveFailed'));
      console.error('Failed to save config:', err);
    } finally {
      setSaving(false);
    }
  };

  const getBool = (key: string, fallback = true): boolean => {
    const val = config[key];
    if (typeof val === 'boolean') return val;
    if (typeof val === 'string') return val.toLowerCase() === 'true';
    return fallback;
  };

  const getStr = (key: string, fallback = ''): string => {
    const val = config[key];
    return val != null ? String(val) : fallback;
  };

  const getNum = (key: string, fallback = 0): number => {
    const val = config[key];
    if (typeof val === 'number') return val;
    if (typeof val === 'string') {
      const n = Number(val);
      return isNaN(n) ? fallback : n;
    }
    return fallback;
  };

  if (error && !Object.keys(config).length) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">{t('settings.loadError')}</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
          <Button onClick={fetchConfig}>
            <RefreshCw className="mr-2 h-4 w-4" />
            {t('settings.retry')}
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{t('settings.title')}</h2>
            <p className="text-muted-foreground">{t('settings.description')}</p>
          </div>
          <Button variant="ghost" size="icon" onClick={fetchConfig} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <Tabs defaultValue="general" className="space-y-4">
          <TabsList>
            <TabsTrigger value="general" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              {t('settings.tab.general')}
            </TabsTrigger>
            <TabsTrigger value="bot" className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              {t('settings.tab.bot')}
            </TabsTrigger>
            <TabsTrigger value="integrations" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Integrations
            </TabsTrigger>
            <TabsTrigger value="model" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              {t('settings.tab.model')}
            </TabsTrigger>
            <TabsTrigger value="notifications" className="flex items-center gap-2">
              <Bell className="h-4 w-4" />
              {t('settings.tab.notifications')}
            </TabsTrigger>
            <TabsTrigger value="security" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              {t('settings.tab.security')}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{t('settings.general.title')}</CardTitle>
                <CardDescription>{t('settings.general.description')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-10 w-full" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="system-name">{t('settings.general.systemName')}</Label>
                      <Input
                        id="system-name"
                        value={getStr('botName', 'MultiPersonal Chat System')}
                        onChange={(e) => updateField('botName', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="language">{t('settings.general.language')}</Label>
                      <Select
                        value={getStr('language', 'zh-CN')}
                        onValueChange={(v) => updateField('language', v)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t('settings.general.language')} />
                        </SelectTrigger>
                        <SelectContent>
                          {getSupportedLocales().map((loc) => (
                            <SelectItem key={loc.value} value={loc.value}>
                              {t(`settings.general.language.${loc.value === 'zh-CN' ? 'zhCN' : loc.value === 'zh-TW' ? 'zhTW' : 'en'}`)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="timezone">{t('settings.general.timezone')}</Label>
                      <Select
                        value={getStr('timezone', 'Asia/Shanghai')}
                        onValueChange={(v) => updateField('timezone', v)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t('settings.general.timezone')} />
                        </SelectTrigger>
                        <SelectContent>
                          {getSupportedTimezones().map((tz) => (
                            <SelectItem key={tz.value} value={tz.value}>
                              {t(tz.labelKey)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bot" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{t('settings.bot.title')}</CardTitle>
                <CardDescription>{t('settings.bot.description')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-6 w-10" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.bot.autoReply')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.bot.autoReplyDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('autoReply', true)}
                        onCheckedChange={(v) => updateField('autoReply', v)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.bot.groupReply')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.bot.groupReplyDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('groupReply', true)}
                        onCheckedChange={(v) => updateField('groupReply', v)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.bot.privateReply')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.bot.privateReplyDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('privateReply', true)}
                        onCheckedChange={(v) => updateField('privateReply', v)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.bot.replyDelay')}</Label>
                      <Input
                        type="number"
                        value={getNum('replyDelay', 1)}
                        min={0}
                        max={10}
                        onChange={(e) => updateField('replyDelay', Number(e.target.value))}
                      />
                      <p className="text-sm text-muted-foreground">{t('settings.bot.replyDelayDesc')}</p>
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.bot.defaultTemplate')}</Label>
                      <Textarea
                        placeholder={t('settings.bot.defaultTemplatePlaceholder')}
                        value={getStr('defaultReplyTemplate', '')}
                        onChange={(e) => updateField('defaultReplyTemplate', e.target.value)}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="integrations" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>AstrBot 多平台接入</CardTitle>
                <CardDescription>配置 AstrBot 网关到后端的内部回调和平台开关。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-10 w-full" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label>后端回调地址</Label>
                      <Input value="/api/integrations/astrbot/messages" readOnly />
                      <p className="text-sm text-muted-foreground">AstrBot 插件会把 QQ、Telegram、微信系消息标准化后发送到这个内部接口。</p>
                    </div>
                    <div className="rounded-md border p-3 text-sm text-muted-foreground">
                      共享密钥仅从服务器环境变量 ASTRBOT_INTEGRATION_TOKEN / ASTRBOT_INTEGRATION_TOKENS 读取，管理台不会读取或保存密钥。
                    </div>
                    <div className="grid gap-3 md:grid-cols-2">
                      {[
                        ['astrbotQQEnabled', 'QQ / NapCat'],
                        ['astrbotTelegramEnabled', 'Telegram'],
                        ['astrbotWecomEnabled', '企业微信'],
                        ['astrbotWechatOfficialEnabled', '微信公众号'],
                        ['astrbotWechatPersonalEnabled', '个人微信（实验）'],
                      ].map(([key, label]) => (
                        <div key={key} className="flex items-center justify-between rounded-md border p-3">
                          <Label>{label}</Label>
                          <Switch checked={getBool(key, key !== 'astrbotWechatPersonalEnabled')} onCheckedChange={(v) => updateField(key, v)} />
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="model" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{t('settings.model.title')}</CardTitle>
                <CardDescription>{t('settings.model.description')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-10 w-full" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label>{t('settings.model.provider')}</Label>
                      <p className="text-sm text-muted-foreground">{t('settings.model.providerDesc')}</p>
                      <Select
                        value={getStr('modelProvider', 'vllm')}
                        onValueChange={(v) => updateField('modelProvider', v)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t('settings.model.provider')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="vllm">vLLM (本地GPU推理)</SelectItem>
                          <SelectItem value="openai_compat">DeepSeek / OpenAI 兼容 API</SelectItem>
                          <SelectItem value="transformers_peft">Transformers + PEFT (本地)</SelectItem>
                          <SelectItem value="ollama">Ollama (本地)</SelectItem>
                          <SelectItem value="llama_cpp">llama.cpp (本地)</SelectItem>
                          <SelectItem value="mock">Mock (仅开发测试)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {getStr('modelProvider', 'vllm') === 'openai_compat' && (
                      <div className="space-y-4 rounded-lg border p-4 bg-muted/30">
                        <div className="space-y-2">
                          <Label>{t('settings.model.apiBaseUrl')}</Label>
                          <p className="text-sm text-muted-foreground">{t('settings.model.apiBaseUrlDesc')}</p>
                          <Input
                            value={getStr('openaiCompatBaseUrl', 'https://api.deepseek.com')}
                            onChange={(e) => updateField('openaiCompatBaseUrl', e.target.value)}
                            placeholder="https://api.deepseek.com"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>{t('settings.model.apiKey')}</Label>
                          <p className="text-sm text-muted-foreground">{t('settings.model.apiKeyDesc')}</p>
                          <Input
                            type="password"
                            value={getStr('openaiCompatApiKey', '')}
                            onChange={(e) => updateField('openaiCompatApiKey', e.target.value)}
                            placeholder="sk-..."
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>{t('settings.model.apiModel')}</Label>
                          <p className="text-sm text-muted-foreground">{t('settings.model.apiModelDesc')}</p>
                          <Input
                            value={getStr('openaiCompatModel', 'deepseek-chat')}
                            onChange={(e) => updateField('openaiCompatModel', e.target.value)}
                            placeholder="deepseek-chat"
                          />
                        </div>
                      </div>
                    )}
                    {getStr('modelProvider', 'mock') === 'vllm' && (
                      <div className="space-y-3 rounded-lg border p-4 bg-muted/30">
                        <div className="space-y-1">
                          <Label>vLLM 推理服务</Label>
                          <p className="text-sm text-muted-foreground">
                            vLLM 通过环境变量配置 (VLLM_ENABLED, VLLM_BASE_URLS)。
                            请确保 vLLM 服务已启动并正常运行。
                          </p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            try {
                              // H1 fix: 此前使用裸 fetch，既不检查 response.ok 也不处理 401 重定向。
                              // 现通过 api 助手调用，确保 token 过期时自动跳转登录页。
                              const data = await api.getVllmStatus();
                              if (data.enabled && data.summary?.all_healthy) {
                                toast.success(`vLLM 运行正常 (${data.summary.healthy}/${data.summary.total} 实例健康)`);
                              } else if (data.enabled) {
                                toast.warning(`vLLM 已启用但有实例不健康 (${data.summary?.healthy}/${data.summary?.total})`);
                              } else {
                                toast.error('vLLM 未启用，请检查环境变量 VLLM_ENABLED 和 VLLM_BASE_URLS');
                              }
                            } catch {
                              toast.error('无法获取 vLLM 状态');
                            }
                          }}
                        >
                          <Zap className="mr-2 h-4 w-4" />
                          检查 vLLM 连接状态
                        </Button>
                      </div>
                    )}
                    <div className="space-y-2">
                      <Label>{t('settings.model.baseModel')}</Label>
                      <Select
                        value={getStr('baseModel', 'qwen3-8b')}
                        onValueChange={(v) => updateField('baseModel', v)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t('settings.model.baseModel')} />
                        </SelectTrigger>
                        <SelectContent>
                          {availableModels.map((model) => (
                            <SelectItem key={model.name} value={model.name}>
                              {model.display_name} {!model.downloaded ? '(未下载)' : ''}
                            </SelectItem>
                          ))}
                          {availableModels.length === 0 && (
                            <SelectItem value="qwen3-8b">Qwen3-8B</SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.model.temperature')}</Label>
                      <Input
                        type="number"
                        value={getNum('temperature', 0.7)}
                        min={0}
                        max={2}
                        step={0.1}
                        onChange={(e) => updateField('temperature', Number(e.target.value))}
                      />
                      <p className="text-sm text-muted-foreground">{t('settings.model.temperatureDesc')}</p>
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.model.maxLength')}</Label>
                      <Input
                        type="number"
                        value={getNum('maxTokens', 2048)}
                        min={256}
                        max={8192}
                        step={256}
                        onChange={(e) => updateField('maxTokens', Number(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.model.contextWindow')}</Label>
                      <Select
                        value={getStr('contextWindow', '8k')}
                        onValueChange={(v) => updateField('contextWindow', v)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t('settings.model.contextWindow')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="4k">4K tokens</SelectItem>
                          <SelectItem value="8k">8K tokens</SelectItem>
                          <SelectItem value="16k">16K tokens</SelectItem>
                          <SelectItem value="32k">32K tokens</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.model.useKnowledge')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.model.useKnowledgeDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('useKnowledgeBase', true)}
                        onCheckedChange={(v) => updateField('useKnowledgeBase', v)}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="notifications" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{t('settings.notifications.title')}</CardTitle>
                <CardDescription>{t('settings.notifications.description')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-6 w-10" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.notifications.errorAlert')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.notifications.errorAlertDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('errorAlert', true)}
                        onCheckedChange={(v) => updateField('errorAlert', v)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.notifications.dailyStats')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.notifications.dailyStatsDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('dailyStats', true)}
                        onCheckedChange={(v) => updateField('dailyStats', v)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.notifications.anomalyDetection')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.notifications.anomalyDetectionDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('anomalyDetection', false)}
                        onCheckedChange={(v) => updateField('anomalyDetection', v)}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="security" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{t('settings.security.title')}</CardTitle>
                <CardDescription>{t('settings.security.description')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="space-y-2">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-10 w-full" />
                    </div>
                  ))
                ) : (
                  <>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.security.contentFilter')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.security.contentFilterDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('contentFilter', true)}
                        onCheckedChange={(v) => updateField('contentFilter', v)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>{t('settings.security.contentReview')}</Label>
                        <p className="text-sm text-muted-foreground">{t('settings.security.contentReviewDesc')}</p>
                      </div>
                      <Switch
                        checked={getBool('contentReview', true)}
                        onCheckedChange={(v) => updateField('contentReview', v)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>{t('settings.security.adminQQ')}</Label>
                      <Textarea
                        placeholder={t('settings.security.adminQQPlaceholder')}
                        value={getStr('adminQqList', '')}
                        onChange={(e) => updateField('adminQqList', e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">{t('settings.security.adminQQDesc')}</p>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

        </Tabs>

        <div className="flex items-center justify-between">
          {Object.keys(config).length > 0 && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              {t('settings.loaded')} {Object.keys(config).length} {t('settings.items')}
            </div>
          )}
          <Button onClick={handleSave} disabled={saving || loading} className="ml-auto">
            <Save className="mr-2 h-4 w-4" />
            {saving ? t('settings.saving') : t('settings.save')}
          </Button>
        </div>
      </div>
    </AppLayout>
  );
}
