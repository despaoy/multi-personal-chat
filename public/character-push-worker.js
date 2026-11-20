/* Browser notifications are generic pointers to the authenticated inbox. */
self.addEventListener('push', event => {
  let tag;
  try { tag = event.data?.json().tag; } catch { return; }
  if (typeof tag !== 'string' || !/^[a-f0-9]{32}$/.test(tag)) return;
  event.waitUntil(self.registration.showNotification('有一条角色消息', {
    body: '打开应用查看消息。',
    tag: 'character-' + tag,
    renotify: false,
  }));
});

self.addEventListener('notificationclick', event => {
  if (!/^character-[a-f0-9]{32}$/.test(event.notification.tag)) return;
  event.notification.close();
  event.waitUntil(self.clients.openWindow(new URL('/chat', self.location.origin).href));
});
