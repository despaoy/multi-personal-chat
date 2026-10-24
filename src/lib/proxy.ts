/**
 * API 代理工具函数
 * 统一处理后端代理请求，包括 Cookie→Authorization 转换、错误处理、超时控制
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const PROXY_TIMEOUT = Number(process.env.PROXY_TIMEOUT_MS || 30000);
const PROXY_LONG_TIMEOUT = Number(process.env.PROXY_LONG_TIMEOUT_MS || 210000);

type ProxyTimeoutClass = 'default' | 'long';

interface ProxyRoutePolicy {
  prefix: string;
  timeout?: ProxyTimeoutClass;
}

// 安全：允许转发的后端路径前缀白名单。
// 路径安全：拒绝含 '..' 或 '//' 的路径，防止目录穿越。
// 说明：auth/* 同时有专用 route（处理 Set-Cookie）和 catch-all 两条路径，
//       Next.js 路由优先级下专用 route 优先匹配，此处保留 auth/* 是因为
//       专用 route 内部也调用 proxyRequest，需要通过白名单校验。
//       后端安全中间件独立校验认证（AUTH_WHITELIST 保护 login/register），
//       所以前端白名单不会降低安全性。
// 超时等级也在同一份清单中声明，新增接口不需要同步维护第二个数组。
const PROXY_ROUTE_POLICIES: readonly ProxyRoutePolicy[] = [
  { prefix: '/api/generate', timeout: 'long' },
  { prefix: '/api/narrative-branches', timeout: 'long' },
  { prefix: '/api/training/start', timeout: 'long' },
  { prefix: '/api/training/generate-dialogues', timeout: 'long' },
  { prefix: '/api/knowledge/bases/', timeout: 'long' },
  { prefix: '/api/knowledge/scan/import', timeout: 'long' },
  { prefix: '/api/knowledge/train-intent', timeout: 'long' },
  { prefix: '/api/evaluation/run', timeout: 'long' },
  { prefix: '/api/experiments/', timeout: 'long' },
  { prefix: '/api/auth' },
  { prefix: '/api/messages' },
  { prefix: '/api/sessions' },
  { prefix: '/api/vllm/' },
  { prefix: '/api/loras' },
  { prefix: '/api/characters' },
  { prefix: '/api/training' },
  { prefix: '/api/knowledge' },
  { prefix: '/api/vector/' },
  { prefix: '/api/model' },
  { prefix: '/api/models' },
  { prefix: '/api/config' },
  { prefix: '/api/user/' },
  { prefix: '/api/stats' },
  { prefix: '/api/services' },
  { prefix: '/api/claw' },
  { prefix: '/api/enhanced' },
  { prefix: '/api/evaluation' },
  { prefix: '/api/experiments' },
  { prefix: '/api/retrieval-eval' },
  { prefix: '/api/preferences' },
  { prefix: '/api/router' },
  { prefix: '/api/feedback' },
  { prefix: '/health' },
  { prefix: '/ready' },
];

function matchesProxyPrefix(path: string, prefix: string): boolean {
  if (path === prefix) return true;
  if (prefix.endsWith('/')) return path.startsWith(prefix);
  return path.startsWith(prefix + '/');
}

function defaultProxyTimeout(path: string): number {
  const pathOnly = path.split('?')[0];
  const policy = PROXY_ROUTE_POLICIES.find(
    ({ prefix, timeout }) => timeout === 'long' && matchesProxyPrefix(pathOnly, prefix)
  );
  return policy ? PROXY_LONG_TIMEOUT : PROXY_TIMEOUT;
}

/**
 * 校验目标后端路径是否允许通过 catch-all 代理。
 * 拒绝：空路径、含 `..`、未在白名单前缀中的路径。
 */
export function isProxyPathAllowed(backendPath: string): boolean {
  if (!backendPath) return false;
  // 仅取 path 部分（剥离 query）
  const pathOnly = backendPath.split('?')[0];
  if (pathOnly.includes('..') || pathOnly.includes('//')) return false;
  // 必须命中白名单前缀：精确匹配或子路径匹配
  // 对于以 / 结尾的前缀（如 /api/vllm/），直接用 startsWith
  // 对于不以 / 结尾的前缀（如 /api/auth），用精确匹配或 path + '/'，避免 /api/model 误匹配 /api/modelxyz
  return PROXY_ROUTE_POLICIES.some(({ prefix }) => matchesProxyPrefix(pathOnly, prefix));
}

function isUnsafeMethod(method: string): boolean {
  return ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method.toUpperCase());
}

function isLoopbackHost(hostname: string): boolean {
  return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(hostname.toLowerCase());
}

function originsMatch(actualOrigin: string, expectedOrigin: string): boolean {
  if (actualOrigin === expectedOrigin) return true;

  try {
    const actual = new URL(actualOrigin);
    const expected = new URL(expectedOrigin);
    return (
      actual.protocol === expected.protocol &&
      actual.port === expected.port &&
      isLoopbackHost(actual.hostname) &&
      isLoopbackHost(expected.hostname)
    );
  } catch {
    return false;
  }
}

function requestVisibleOrigin(request: Request): string {
  const internalUrl = new URL(request.url);
  const forwardedHost = request.headers.get('x-forwarded-host')?.split(',')[0]?.trim();
  // Host is browser-visible and cannot be replaced by a client-supplied forwarding header.
  // A trusted reverse proxy must preserve it (including non-standard ports).
  const host = request.headers.get('host') || forwardedHost;
  if (!host) return internalUrl.origin;

  const forwardedProtocol = request.headers.get('x-forwarded-proto')?.split(',')[0]?.trim();
  const protocol = forwardedProtocol || internalUrl.protocol.replace(':', '');
  return `${protocol}://${host}`;
}

function isSameOriginRequest(request: Request): boolean {
  // request.url may contain Next's internal listening port when accessed through
  // an SSH/reverse-proxy tunnel. Host preserves the browser-visible origin.
  const expectedOrigin = requestVisibleOrigin(request);
  const origin = request.headers.get('Origin');
  if (origin) {
    return originsMatch(origin, expectedOrigin);
  }
  const referer = request.headers.get('Referer');
  if (referer) {
    try {
      return originsMatch(new URL(referer).origin, expectedOrigin);
    } catch {
      return false;
    }
  }
  return true;
}

/** Forward the original request; own cancellation until the response body finishes. */
export async function proxyRequest(
  request: Request,
  path: string,
  options: { timeout?: number } = {}
): Promise<Response> {
  const method = request.method;
  if (isUnsafeMethod(method) && !isSameOriginRequest(request)) {
    return Response.json({ detail: 'CSRF check failed' }, { status: 403 });
  }
  if (!isProxyPathAllowed(path)) {
    return Response.json({ detail: '禁止访问该路径' }, { status: 403 });
  }

  const headers = new Headers();
  const cookie = request.headers.get('Cookie')?.split(';')
    .map(part => part.trim()).find(part => part.startsWith('access_token='));
  const token = cookie?.substring('access_token='.length);
  const authorization = token ? `Bearer ${token}` : request.headers.get('Authorization');
  if (authorization) headers.set('Authorization', authorization);
  for (const name of ['Content-Type', 'Content-Length']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const controller = new AbortController();
  const signal = AbortSignal.any([request.signal, controller.signal]);
  const finish = () => clearTimeout(timeoutId);
  const timeoutId = setTimeout(
    () => controller.abort(new DOMException('请求超时', 'TimeoutError')),
    options.timeout ?? defaultProxyTimeout(path)
  );

  try {
    const fetchOptions: RequestInit & { duplex?: 'half' } = { method, headers, signal };
    if (request.body) {
      fetchOptions.body = request.body;
      fetchOptions.duplex = 'half';
    }
    const response = await fetch(`${BACKEND_URL}${path}`, fetchOptions);
    const responseHeaders = new Headers(response.headers);
    // fetch has already decoded compressed responses; their wire headers are stale.
    responseHeaders.delete('Content-Encoding');
    responseHeaders.delete('Content-Length');
    const init = { status: response.status, statusText: response.statusText, headers: responseHeaders };
    if (!response.body || response.headers.get('Content-Type')?.includes('application/json')) {
      const body = response.body ? await response.arrayBuffer() : null;
      finish();
      return new Response(body, init);
    }

    const stream = new TransformStream<Uint8Array, Uint8Array>();
    // pipeTo forwards failure/cancellation to the readable side; both outcomes release the timer.
    void response.body.pipeTo(stream.writable, { signal }).then(finish, finish);
    return new Response(stream.readable, init);
  } catch (error) {
    finish();
    request.signal.throwIfAborted();
    if (controller.signal.aborted) {
      return Response.json({ detail: '请求超时' }, { status: 504 });
    }
    if (error instanceof TypeError) {
      return Response.json({ detail: '后端服务不可用' }, { status: 502 });
    }
    throw error;
  }
}
