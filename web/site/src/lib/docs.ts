// Danh sách trang tài liệu Lab theo thứ tự, dùng cho route, sidebar và nút trang trước/sau.
import { getCollection, type CollectionEntry } from "astro:content";

export type LabDoc = CollectionEntry<"lab">;

export async function labDocs(): Promise<LabDoc[]> {
  const docs = await getCollection("lab");
  return docs.sort((a, b) => a.data.order - b.data.order);
}
