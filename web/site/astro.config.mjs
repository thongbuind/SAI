// Một mã nguồn, hai bản deploy. Chọn bản bằng PUBLIC_DEPLOY_TARGET lúc build:
// - static: GitHub Pages, chạy dưới BASE_PATH (vd. /SAI), trang /try/ là hướng dẫn tải mô hình.
// - server: FastAPI phục vụ ở gốc /, trang /try/ là chat, có thêm phòng làm việc /lab/workbench/<phòng>/ (nút bấm
//   chạy được khi server bật --lab). Khi chạy dev:server, /api chuyển tới FastAPI (python -m web) để frontend và
//   API cùng origin như lúc deploy.
import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";

const TARGETS = ["static", "server"];
const DEV_API_ORIGIN = "http://127.0.0.1:8000";
const TRY_PAGES = {
  static: "./src/components/try/TryStatic.astro",
  server: "./src/components/try/TryServer.astro",
};
const target = process.env.PUBLIC_DEPLOY_TARGET;
if (!TARGETS.includes(target)) {
  throw new Error(`PUBLIC_DEPLOY_TARGET phải là "static" hoặc "server", nhận được: ${target}`);
}
if (target === "static" && !process.env.BASE_PATH) {
  throw new Error("Bản static cần BASE_PATH (vd. /SAI) vì GitHub Pages phục vụ web dưới tên repo");
}

// Phòng làm việc chỉ có ở bản server: route được đăng ký ở đây thay vì đặt trong src/pages, vì Astro gom JS/CSS của
// mọi trang trong src/pages vào bản build kể cả khi getStaticPaths không trả trang nào.
const WORKBENCH_ROUTE = {
  name: "sai-lab-workbench",
  hooks: {
    "astro:config:setup": ({ injectRoute }) => {
      injectRoute({ pattern: "/lab/workbench/[department]", entrypoint: "./src/components/lab/WorkbenchPage.astro" });
    },
  },
};

export default defineConfig({
  site: process.env.SITE_URL,
  base: target === "static" ? process.env.BASE_PATH : "/",
  outDir: `./dist/${target}`,
  trailingSlash: "always",
  integrations: target === "server" ? [WORKBENCH_ROUTE] : [],
  markdown: {
    // Khối code trong tài liệu Markdown dùng chung cách tô màu với <CodeBlock> (ánh xạ trong base.css).
    shikiConfig: { theme: "css-variables" },
    // Tài liệu có nhiều cờ dòng lệnh (--resume, --set); smartypants sẽ đổi "--" thành gạch dài.
    smartypants: false,
  },
  vite: {
    // src/pages/try.astro import "@try-page": chỉ component của bản đang build vào bundle.
    resolve: { alias: { "@try-page": fileURLToPath(new URL(TRY_PAGES[target], import.meta.url)) } },
    server: target === "server" ? { proxy: { "/api": DEV_API_ORIGIN } } : {},
  },
});
