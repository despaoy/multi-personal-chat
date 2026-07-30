# Next.js Frontend - Multi-stage build
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
# 使用 npmjs.org 官方源，避免镜像源在 CI/海外环境不可用
RUN corepack enable && pnpm config set registry https://registry.npmjs.org/ && pnpm install --frozen-lockfile

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN corepack enable && pnpm build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV BACKEND_URL=http://backend:8000

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 5000
# H3 fix: 显式设置 HOSTNAME=0.0.0.0，确保容器内服务可从外部访问。
# Next.js standalone server.js 在某些版本默认监听 localhost，会导致容器外不可达。
ENV PORT=5000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
