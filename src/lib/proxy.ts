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

interface ProxyOptions {
  method?: string;
  body?: unknown;
  headers?: Record<string, string>;
  timeout?: number;
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

/**
 * 代理请求到后端 FastAPI 服务
 *
 * 认证策略：
 * 1. 从 httpOnly Cookie (access_token) 中提取 JWT，转为 Authorization: Bearer 头转发
 *    （Node.js fetch 会忽略手动设置的 Cookie 头，因此必须转为 Authorization 头）
 * 2. 回退到原始 Authorization header（向后兼容）
 */
export async function proxyRequest(
  request: Request,
  path: string,
  options: ProxyOptions = {}
): Promise<Response> {
  const { method = 'GET', body, headers: optHeaders = {}, timeout } = options;
  const effectiveTimeout = timeout ?? defaultProxyTimeout(path);
  const isFormDataBody = typeof FormData !== 'undefined' && body instanceof FormData;
  const isStreamBody = typeof ReadableStream !== 'undefined' && body instanceof ReadableStream;

  if (isUnsafeMethod(method) && !isSameOriginRequest(request)) {
    return new Response(JSON.stringify({ detail: 'CSRF check failed' }), {
      status: 403,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // 安全：禁止转发到未授权的后端路径，防止把整张后端 API 表面开放成代理
  if (!isProxyPathAllowed(path)) {
    return new Response(
      JSON.stringify({ detail: '禁止访问该路径' }),
      { status: 403, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const headers: Record<string, string> = { ...optHeaders };

  // 认证：从 Cookie 中提取 JWT token，转为 Authorization 头转发
  // 注意：Node.js fetch 会忽略手动设置的 Cookie 头，必须用 Authorization 头传递认证信息
  const cookieHeader = request.headers.get('Cookie') || '';
  if (cookieHeader) {
    // 正确解析 Cookie：用 substring 而非 split('=') 避免 JWT 中的 = 字符被截断
    const cookieParts = cookieHeader.split(';');
    for (const part of cookieParts) {
      const trimmed = part.trim();
      if (trimmed.startsWith('access_token=')) {
        const token = trimmed.substring('access_token='.length);
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        break;
      }
    }
  }
  if (!headers['Authorization']) {
    const authHeader = request.headers.get('Authorization');
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
  }

  // 转发 Content-Type
  const contentType = request.headers.get('Content-Type');
  if (contentType && method !== 'GET' && !isFormDataBody) {
    headers['Content-Type'] = contentType;
  }
  if (isStreamBody) {
    const contentLength = request.headers.get('Content-Length');
    if (contentLength) headers['Content-Length'] = contentLength;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), effectiveTimeout);

  try {
    const fetchOptions: RequestInit = {
      method,
      headers,
      signal: controller.signal,
    };

    if (body && method !== 'GET') {
      if (isFormDataBody || isStreamBody) {
        fetchOptions.body = body as BodyInit;
        if (isStreamBody) {
          (fetchOptions as RequestInit & { duplex: 'half' }).duplex = 'half';
        }
      } else {
        fetchOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
      }
    }

    const response = await fetch(`${BACKEND_URL}${path}`, fetchOptions);
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return new Response(JSON.stringify({ detail: '请求超时' }), {
        status: 504,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response(JSON.stringify({ detail: '后端服务不可用' }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' },
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * 代理 GET 请求
 */
export async function proxyGet(request: Request, path: string): Promise<Response> {
  return proxyRequest(request, path);
}

/**
 * 代理 POST 请求，自动读取请求体
 */
export async function proxyPost(request: Request, path: string): Promise<Response> {
  let body: unknown;
  try {
    const text = await request.text();
    body = JSON.parse(text);
  } catch {
    return new Response(JSON.stringify({ detail: '请求体不是有效的 JSON' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  return proxyRequest(request, path, {
    method: 'POST',
    body,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * 代理 PUT 请求，自动读取请求体
 */
export async function proxyPut(request: Request, path: string): Promise<Response> {
  const body = await request.json();
  return proxyRequest(request, path, {
    method: 'PUT',
    body,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * 代理 PATCH 请求，自动读取请求体
 */
export async function proxyPatch(request: Request, path: string): Promise<Response> {
  const body = await request.json();
  return proxyRequest(request, path, {
    method: 'PATCH',
    body,
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * 代理 DELETE 请求
 */
export async function proxyDelete(request: Request, path: string): Promise<Response> {
  return proxyRequest(request, path, { method: 'DELETE' });
}

export { BACKEND_URL };
