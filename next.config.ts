import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
  logging: {
    fetches: {
      fullUrl: false,
    },
  },
  // Turbopack配置（Next.js 16默认使用Turbopack）
  turbopack: {},

};

export default nextConfig;
