// next.config.mjs

/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        // 모든 경로에 적용
        source: "/(.*)",
        headers: [
          {
            key: "Cross-Origin-Opener-Policy",
            value: "*",
          },
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "*",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
