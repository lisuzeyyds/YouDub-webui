import type { NextConfig } from "next";

function apiProxyTarget() {
  const configured =
    process.env.NEXT_SERVER_API_BASE_URL ||
    "http://127.0.0.1:8000";
  return configured.replace(/\/$/, "");
}

const nextConfig: NextConfig = {
  allowedDevOrigins: ["172.27.2.90", "100.94.222.54"],
  experimental: {
    // 大视频上传走 rewrites 代理时，默认 30s 超时 + 10MB 请求体截断
    // 这里与后端 LOCAL_UPLOAD_MAX_BYTES (4GB) 对齐
    proxyTimeout: 30 * 60 * 1000,
    proxyClientMaxBodySize: "5gb",
    // Next 16 的 Turbopack dev 会把路由校验放在一个 worker 线程里跑
    // (numWorkers: 1, maxRetries: 0)。该 worker 一旦退出（内存紧张时很容易发生），
    // 就会抛出 "Jest worker encountered N child process exceptions"，即浏览器里
    // 那个 Runtime Error 遮罩。关掉它改为进程内校验，代价只是 dev 下稍慢一点。
    devValidationWorker: false,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyTarget()}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
