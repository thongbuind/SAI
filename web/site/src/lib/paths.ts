// Gắn base path vào đường dẫn nội bộ: bản static chạy dưới /SAI, bản server chạy ở gốc /.
// path bắt đầu bằng "/" và kết thúc bằng "/" (trailingSlash: "always"), có thể kèm #anchor.
export function url(path: string): string {
  if (!path.startsWith("/")) {
    throw new Error(`Đường dẫn nội bộ phải bắt đầu bằng "/": ${path}`);
  }
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return base + path;
}
