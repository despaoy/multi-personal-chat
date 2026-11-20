import { proxyRequest } from '@/lib/proxy';
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const response = await proxyRequest(request, '/api/auth/logout');
  if (response.status === 403) return response;

  const nextResponse = new NextResponse(response.body, {
    status: response.status, statusText: response.statusText, headers: response.headers,
  });

  // 本站负责向浏览器签发 Cookie，因此即使后端暂时不可用也必须完成本地登出。
  // 后端可用时，proxyRequest 仍会转发 JWT 以写入吊销列表。
  // 普通 HTTP 也需要接受这个过期标记。HTTPS 会按名称和路径清除原有的 Secure Cookie。
  nextResponse.cookies.set({
    name: 'access_token',
    value: '',
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
    expires: new Date(0),
  });

  return nextResponse;
}
