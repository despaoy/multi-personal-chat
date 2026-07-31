import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * 集中式路由守卫（Next.js Proxy）
 *
 * 此前仅靠各页面单独的 <AuthGuard> 做客户端守卫，存在两个问题：
 *  1. 仪表盘首页（src/app/page.tsx）未包裹 AuthGuard，未登录可直接进入
 *  2. 客户端守卫依赖 /api/auth/me 往返，未登录用户在重定向前会先看到页面骨架/闪烁
 *
 * 此 proxy 在请求进入页面前做轻量 Cookie 存在性检查（无法解 JWT，只看 access_token 是否存在），
 * 真正的 token 有效性仍由后端 + AuthContext 的 /api/auth/me 验证。
 * 形成双层防护：Proxy 拦截无 Cookie → 客户端 AuthGuard 拦截无效/过期 token → 后端 401。
 *
 * 不在此处解 JWT；Cookie 存在性检查只负责阻止未登录直访，
 * token 的真实性和有效期仍由后端校验。
 */
export function proxy(request: NextRequest) {
  const path = request.nextUrl.pathname;
  const isLoginPage = path === '/login';
  // C6 fix: ?expired=1 表示客户端刚检测到 401，正在主动跳转登出。
  // 此时 Cookie 可能尚未被后端清除（异步请求），必须放行到 /login，
  // 否则会因 Cookie 仍存在而被重定向回 /，形成死锁。
  const isExpired = request.nextUrl.searchParams.get('expired') === '1';

  const token = request.cookies.get('access_token')?.value;

  // 已登录用户访问 /login → 重定向到首页，避免重复登录
  // 但 expired 流程例外：强制放行到 /login 让用户重新登录
  if (isLoginPage && token && !isExpired) {
    return NextResponse.redirect(new URL('/', request.url));
  }

  // 未登录访问受保护页面 → 重定向到 /login
  if (!token && !isLoginPage) {
    const loginUrl = new URL('/login', request.url);
    // 记录来源路径，登录后可回跳
    loginUrl.searchParams.set('redirect', path);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  // 排除 API 路由、Next.js 静态资源、图片、favicon，只守卫页面路由
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico|robots.txt).*)'],
};
