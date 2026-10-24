// Liên kết dùng chung cho navbar, menu mobile và footer.
export type PageKey = "home" | "lab" | "try";

export const GITHUB_URL = "https://github.com/thongbuind/SAI";
export const HUGGING_FACE_URL = "https://huggingface.co/collections/thongbuind/sai";
export const FACEBOOK_URL = "https://facebook.com/sai.nidai";

// path là đường dẫn nội bộ (chưa gắn base), dùng với url() trong lib/paths.ts.
export const NAV_LINKS: { key: PageKey; label: string; path: string }[] = [
  { key: "home", label: "Giới thiệu", path: "/" },
  { key: "lab", label: "Foundation Lab", path: "/lab/" },
];
