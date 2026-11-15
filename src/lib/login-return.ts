/** Accept only same-origin page paths, never arbitrary redirect URLs. */
export function loginReturnPath(value: string | null): string {
  if (!value || !value.startsWith('/') || value.startsWith('//') || /[\\\u0000-\u001f]/.test(value)) return '/';
  try {
    const target = new URL(value, 'https://local.invalid');
    if (target.origin !== 'https://local.invalid' || target.pathname === '/login' || target.pathname.startsWith('/api/')) return '/';
    return target.pathname + target.search + target.hash;
  } catch { return '/'; }
}
