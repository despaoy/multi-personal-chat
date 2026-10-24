import { proxyRequest } from '@/lib/proxy';

type RouteContext = { params: Promise<{ path?: string[] }> };

async function forward(request: Request, { params }: RouteContext) {
  const { path = [] } = await params;
  if (!path.length) return Response.json({ error: 'Not Found' }, { status: 404 });
  const backendPath = `/api/${path.join('/')}${new URL(request.url).search}`;
  return proxyRequest(request, backendPath);
}

export { forward as GET, forward as POST, forward as PUT, forward as PATCH, forward as DELETE };
