'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { api } from '@/lib/api';

function applicationKey(value: string): Uint8Array<ArrayBuffer> {
  const base64 = value.replaceAll('-', '+').replaceAll('_', '/').padEnd(Math.ceil(value.length / 4) * 4, '=');
  return Uint8Array.from(atob(base64), character => character.charCodeAt(0));
}

export function BrowserPushControl({ sessionId, authorized }: { sessionId: string; authorized: boolean }) {
  const localKey = `character-browser-push:${sessionId}`;
  const [supported, setSupported] = useState<boolean>();
  const [publicKey, setPublicKey] = useState<string | null>();
  const [subscriptionId, setSubscriptionId] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    let active = true;
    setSupported(window.isSecureContext && 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window);
    void (async () => {
      const config = await api.getCharacterBrowserPush(sessionId);
      if (!active) return;
      setPublicKey(config.public_key);
      const local = localStorage.getItem(localKey);
      setSubscriptionId(local && config.subscription_ids.includes(local) ? local : '');
    })().catch(() => { if (active) setError('暂时无法读取浏览器通知设置。'); });
    return () => { active = false; };
  }, [authorized, localKey, sessionId]);

  async function toggle() {
    setBusy(true);
    setError('');
    try {
      if (subscriptionId) {
        await api.removeCharacterBrowserPush(sessionId, subscriptionId);
        setSubscriptionId('');
        localStorage.removeItem(localKey);
        return;
      }
      if (!authorized || !publicKey || !supported) return;
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') throw new Error('浏览器未授予通知权限，应用内消息仍可查看。');
      await navigator.serviceWorker.register('/character-push-worker.js', { scope: '/' });
      const registration = await navigator.serviceWorker.ready;
      const key = applicationKey(publicKey);
      const existing = await registration.pushManager.getSubscription();
      const existingKey = existing?.options.applicationServerKey;
      if (existingKey && new Uint8Array(existingKey).join(',') !== key.join(',')) {
        throw new Error('浏览器订阅使用了旧密钥，请清除该站点的通知权限后重试。');
      }
      const subscription = existing ?? await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key });
      const { endpoint, keys } = subscription.toJSON();
      if (!endpoint || !keys?.p256dh || !keys.auth) throw new Error('浏览器未提供完整通知订阅。');
      const result = await api.registerCharacterBrowserPush(sessionId, { endpoint, p256dh: keys.p256dh, auth: keys.auth });
      setSubscriptionId(result.id);
      localStorage.setItem(localKey, result.id);
    } catch (error) {
      setError(error instanceof Error ? error.message : '浏览器通知设置失败。');
    } finally {
      setBusy(false);
    }
  }

  return <div className="space-y-2 rounded-md border p-3 text-sm">
    <p className="font-medium">浏览器提醒</p>
    {supported === false ? <p>此浏览器或地址不支持 Web Push。应用内消息仍可查看。</p>
      : publicKey === null ? <p>服务器尚未启用浏览器提醒。</p>
      : <p>提醒不显示聊天内容，点击后可进入收件箱。</p>}
    {error && <p role="alert" className="text-destructive">{error}</p>}
    <Button type="button" variant="outline" disabled={busy || (!subscriptionId && (!authorized || !publicKey || !supported))}
      onClick={() => void toggle()}>
      {subscriptionId ? '关闭此浏览器提醒' : '为此会话开启浏览器提醒'}
    </Button>
  </div>;
}
