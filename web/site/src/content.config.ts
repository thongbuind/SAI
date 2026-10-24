// Tài liệu Foundation Lab: mỗi file Markdown trong src/content/lab là một trang /lab/<tên file>/.
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

export const LAB_GROUPS = ["Bắt đầu", "Kiến trúc lab"] as const;

const lab = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/lab" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    group: z.enum(LAB_GROUPS),
    // Thứ tự trong sidebar và nút trang trước/sau, tính trên toàn bộ tài liệu.
    order: z.number().int(),
  }),
});

export const collections = { lab };
