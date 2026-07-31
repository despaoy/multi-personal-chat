import { proxyRequest } from '@/lib/proxy';
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const response = await proxyRequest(request, '/api/auth/logout', {
    method: 'POST',
  });

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    data = { detail: '服务器返回了非 JSON 响应' };
  }

  const nextResponse = NextResponse.json(data, { status: response.status });

  // 本站负责向浏览器签发 Cookie，因此即使后端暂时不可用也必须完成本地登出。
  // 后端可用时，proxyRequest 仍会转发 JWT 以写入吊销列表。
  nextResponse.cookies.set({
    name: 'access_token',
    value: '',
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
    expires: new Date(0),
  });

  return nextResponse;
}