// Dựng form nhập spec từ schema mà lab mô tả (describe) cộng thông tin trình bày của web (src/data/lab_fields.json).
// Schema quyết định có những khoá nào, kiểu gì, mặc định ra sao; lab_fields.json chỉ thêm nhãn, kiểu ô, giới hạn,
// điều kiện hiện, nhóm "Nâng cao", dòng tóm tắt và giá trị ban đầu. Lab vẫn kiểm tra lại toàn bộ spec khi chạy.
//
// Mọi ô phát sự kiện "input"/"change" nổi bọt (ô tự dựng phát thủ công), nên khối cha chỉ cần nghe hai sự kiện
// này để cập nhật tóm tắt, số tham số nâng cao đã đổi và các ô ẩn/hiện.
import type { Description, FieldSchema, ObjectSchema } from "./labApi";
import { listItemYaml, withoutDefaults } from "./specYaml";
import { copyToClipboard, iconSvg } from "../chatRender";

export interface FieldHint {
  label?: string;
  help?: string;
  hidden?: boolean;
  // "checkpoints": model đang có trọng số (khoá của "checkpoints" trong describe), hoặc trọng số của model chọn ở
  // khoá anh em "filter_by"; "registry:<phòng>/<kind>": tên đã đăng ký trong registry.
  source?: string;
  // Lựa chọn phụ thuộc khoá anh em này (vd. checkpoint theo model): đổi khoá đó thì nạp lại danh sách.
  filter_by?: string;
  // "basename": hiện tên file không đường dẫn, không đuôi (giá trị gửi đi vẫn là đường dẫn đầy đủ).
  display?: "basename";
  choices?: (string | number | boolean)[];
  widget?: "lines" | "segmented" | "slider";
  min?: number;
  max?: number;
  // Chỉ dùng cho slider: bước kéo.
  step?: number;
  min_items?: number;
  max_items?: number;
  initial?: unknown;
  // Chỉ hiện khi khoá anh em có một trong các giá trị này (vd. temperature chỉ khi strategy=sample).
  visible_when?: Record<string, unknown[]>;
  // "advanced": gom vào mục "Nâng cao" thu gọn.
  group?: "advanced";
  // Mẫu dòng tóm tắt của object chứa khoá này, "{}" là giá trị; phần đầu tiên là tiêu đề (vd. tên biến thể).
  summary?: string;
  width?: "half";
  // Object lồng: trải các khoá ra ngay trong object cha thay vì đóng khung riêng.
  inline?: boolean;
}
export type FieldHints = Record<string, FieldHint>;

// Giá trị người dùng nhập không hợp lệ; message đã kèm nhãn của ô.
export class FormError extends Error {}

export interface Control {
  element: HTMLElement;
  read(): unknown;
  // Chỉ object: các phần của dòng tóm tắt (đã định dạng), theo thứ tự khoá.
  summary?: () => string[];
  // Chỉ object: ô của từng khoá (trang dùng để đọc riêng một khoá, vd. câu hỏi).
  children?: Record<string, Control>;
  // Chỉ ô chọn: thay danh sách lựa chọn, giữ giá trị đang chọn nếu nó còn trong danh sách mới.
  setChoices?: (values: (string | number | boolean)[]) => void;
}

const LINE_CHIP_CHARS = 28;
// Danh sách dài (benchmark) chỉ hiện vài chip đầu, phần còn lại gom thành "+N câu".
const LINE_CHIPS_SHOWN = 6;
let controlCount = 0;
const nextId = () => `lab-field-${++controlCount}`;

// "variants" + phần tử danh sách → "variants[]"; "variants[]" + "generation" → "variants[].generation".
const childPath = (parent: string, name: string) => (parent ? `${parent}.${name}` : name);

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toLocaleString("vi-VN", { maximumFractionDigits: 2 });
  if (typeof value === "boolean") return value ? "có" : "không";
  return String(value);
}

// Ô tự dựng (segmented, chip) đổi giá trị không qua thẻ nhập liệu nên tự phát sự kiện như thẻ nhập liệu.
function notifyChange(element: HTMLElement) {
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function safeRead(control: Control): unknown {
  // Tóm tắt và bộ đếm chỉ để hiển thị: ô đang nhập dở (FormError) hiện "?" thay vì chặn giao diện.
  try {
    return control.read();
  } catch (error) {
    if (error instanceof FormError) return "?";
    throw error;
  }
}

function sourceValues(source: string, description: Description): string[] {
  if (source === "checkpoints") return Object.keys(description.checkpoints);
  const match = /^registry:(\w+)\/(\w+)$/.exec(source);
  const registry = match && description.departments[match[1]]?.find(item => item.kind === match[2]);
  if (!registry) throw new Error(`lab_fields.json: nguồn "${source}" không có trong describe`);
  return registry.names;
}

// Lựa chọn của khoá có "filter_by", theo giá trị đang chọn ở khoá anh em.
function filteredValues(source: string, description: Description, siblingValue: unknown): string[] {
  if (source !== "checkpoints") throw new Error(`lab_fields.json: "filter_by" chỉ dùng với nguồn "checkpoints", nhận "${source}"`);
  return description.checkpoints[String(siblingValue)] ?? [];
}

const basename = (path: string) => path.split("/").at(-1)!.replace(/\.[^.]+$/, "");

function selectControl(id: string, values: (string | number | boolean)[], initial: unknown, hint: FieldHint, label: string): Control {
  const select = document.createElement("select");
  select.id = id;
  function setChoices(choices: (string | number | boolean)[]) {
    // Lần đầu (chưa có lựa chọn) ưu tiên giá trị ban đầu; sau đó giữ giá trị đang chọn nếu còn trong danh sách.
    const current = select.value ? JSON.parse(select.value) : initial;
    const kept = choices.includes(current) ? current : choices.includes(initial as never) ? initial : choices[0];
    select.replaceChildren(...choices.map(value => {
      const option = document.createElement("option");
      option.value = JSON.stringify(value);
      option.textContent = hint.display === "basename" ? basename(String(value)) : String(value);
      option.title = String(value);
      option.selected = value === kept;
      return option;
    }));
  }
  setChoices(values);
  return {
    element: select,
    read() {
      if (!select.value) throw new FormError(`${label}: không có lựa chọn nào`);
      return JSON.parse(select.value);
    },
    setChoices,
  };
}

function segmentedControl(values: (string | number | boolean)[], initial: unknown): Control {
  const group = document.createElement("div");
  group.className = "lab-segmented";
  group.setAttribute("role", "radiogroup");
  let selected = values.includes(initial as never) ? initial : values[0];
  const buttons = values.map(value => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "radio");
    button.textContent = String(value);
    button.addEventListener("click", () => select(value));
    group.append(button);
    return button;
  });
  function render() {
    values.forEach((value, index) => {
      const checked = value === selected;
      buttons[index].setAttribute("aria-checked", String(checked));
      buttons[index].tabIndex = checked ? 0 : -1;
    });
  }
  function select(value: unknown) {
    if (value === selected) return;
    selected = value;
    render();
    notifyChange(group);
  }
  // Mũi tên trái/phải chuyển lựa chọn như nhóm radio.
  group.addEventListener("keydown", event => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const index = (values.indexOf(selected as never) + step + values.length) % values.length;
    select(values[index]);
    buttons[index].focus();
  });
  render();
  return { element: group, read: () => selected };
}

function sliderControl(id: string, schema: FieldSchema, hint: FieldHint, initial: unknown): Control {
  if (hint.min === undefined || hint.max === undefined) throw new Error(`lab_fields.json: slider cần min và max`);
  const wrapper = document.createElement("div");
  wrapper.className = "lab-slider";
  const range = document.createElement("input");
  range.type = "range";
  range.id = id;
  range.min = String(hint.min);
  range.max = String(hint.max);
  range.step = String(hint.step ?? (schema.type === "int" ? 1 : "any"));
  range.value = String(initial ?? hint.min);
  const output = document.createElement("output");
  output.htmlFor.add(id);
  const show = () => { output.textContent = formatValue(Number(range.value)); };
  range.addEventListener("input", show);
  show();
  wrapper.append(range, output);
  return { element: wrapper, read: () => Number(range.value) };
}

function numberControl(id: string, schema: FieldSchema, hint: FieldHint, label: string, initial: unknown): Control {
  const input = document.createElement("input");
  input.type = "number";
  input.id = id;
  // Số thực không đặt bước: bước của ô number tính từ min, nên mặc định như 1.2 với min 0.01, bước 0.05 bị coi là sai.
  input.step = schema.type === "int" ? "1" : "any";
  if (hint.min !== undefined) input.min = String(hint.min);
  if (hint.max !== undefined) input.max = String(hint.max);
  input.value = initial === null || initial === undefined ? "" : String(initial);
  return {
    element: input,
    read() {
      if (input.value === "") {
        if (schema.nullable) return null;
        throw new FormError(`${label}: cần một số`);
      }
      if (!input.checkValidity()) throw new FormError(`${label}: ${input.validationMessage}`);
      return Number(input.value);
    },
  };
}

// Danh sách chuỗi: chip gọn, bấm "Sửa" mới mở ô nhiều dòng.
function linesControl(id: string, hint: FieldHint, label: string, initial: unknown): Control {
  const wrapper = document.createElement("div");
  wrapper.className = "lab-lines";
  const head = document.createElement("div");
  head.className = "lab-lines-head";
  const chips = document.createElement("div");
  chips.className = "lab-chips";
  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "lab-link";
  edit.setAttribute("aria-expanded", "false");
  edit.setAttribute("aria-controls", id);
  edit.textContent = "Sửa";
  const textarea = document.createElement("textarea");
  textarea.id = id;
  textarea.rows = 5;
  textarea.hidden = true;
  textarea.value = Array.isArray(initial) ? initial.join("\n") : "";
  head.append(chips, edit);
  wrapper.append(head, textarea);

  const lines = () => textarea.value.split("\n").map(line => line.trim()).filter(Boolean);
  function renderChips() {
    const values = lines();
    chips.replaceChildren(...values.slice(0, LINE_CHIPS_SHOWN).map(line => {
      const chip = document.createElement("span");
      chip.className = "lab-chip";
      chip.title = line;
      chip.textContent = line.length > LINE_CHIP_CHARS ? `${line.slice(0, LINE_CHIP_CHARS)}…` : line;
      return chip;
    }));
    if (values.length > LINE_CHIPS_SHOWN) {
      const more = document.createElement("span");
      more.className = "lab-chip lab-chip-more";
      more.textContent = `+${values.length - LINE_CHIPS_SHOWN} câu`;
      chips.append(more);
    }
    if (values.length === 0) {
      const empty = document.createElement("span");
      empty.className = "t-caption";
      empty.textContent = "Chưa có câu nào";
      chips.append(empty);
    }
  }
  edit.addEventListener("click", () => {
    textarea.hidden = !textarea.hidden;
    edit.textContent = textarea.hidden ? "Sửa" : "Xong";
    edit.setAttribute("aria-expanded", String(!textarea.hidden));
    if (!textarea.hidden) textarea.focus();
  });
  textarea.addEventListener("input", renderChips);
  renderChips();
  return {
    element: wrapper,
    read() {
      const values = lines();
      if (hint.max_items !== undefined && values.length > hint.max_items) {
        throw new FormError(`${label}: tối đa ${hint.max_items} dòng, đang có ${values.length}`);
      }
      return values;
    },
  };
}

function iconButton(icon: "code" | "copy" | "trash" | "plus", label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "lab-icon-btn";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.innerHTML = iconSvg(icon);
  return button;
}

// Danh sách object: mỗi mục mở/thu độc lập (bấm thanh tiêu đề), mục thu chỉ hiện dòng tóm tắt. Lúc dựng form mọi
// mục đều thu để nhìn tổng quan; mục vừa thêm hoặc nhân bản thì mở sẵn. Nút </> hiện YAML của mục (chỉ các giá trị
// khác mặc định), dán được vào file spec của lab.
function listOfObjectsControl(path: string, schema: ObjectSchema, hint: FieldHint, hints: FieldHints, description: Description, initial: unknown): Control {
  const itemPath = `${path}[]`;
  const label = hint.label ?? path;
  const minItems = hint.min_items ?? 0;
  const maxItems = hint.max_items ?? Infinity;
  const container = document.createElement("div");
  container.className = "lab-list";
  const head = document.createElement("div");
  head.className = "lab-list-head";
  const count = document.createElement("span");
  count.className = "t-caption";
  const addButton = iconButton("plus", `Thêm ${label.toLocaleLowerCase("vi-VN")}`);
  head.append(count, addButton);
  const itemsElement = document.createElement("div");
  itemsElement.className = "lab-list-items";
  const error = document.createElement("p");
  error.className = "lab-list-error";
  error.setAttribute("role", "alert");
  error.hidden = true;
  container.append(head, itemsElement, error);

  interface Item {
    card: HTMLElement; body: Control; toggle: HTMLButtonElement; title: HTMLElement; detail: HTMLElement;
    showYaml: HTMLButtonElement; yaml: HTMLElement; yamlCode: HTMLElement; duplicate: HTMLButtonElement; remove: HTMLButtonElement;
  }
  const items: Item[] = [];

  function setOpen(item: Item, isOpen: boolean) {
    item.card.dataset.open = String(isOpen);
    item.toggle.setAttribute("aria-expanded", String(isOpen));
    item.body.element.hidden = !isOpen;
  }

  function renderYaml(item: Item) {
    if (item.yaml.hidden) return;
    try {
      item.yamlCode.textContent = listItemYaml(withoutDefaults(item.body.read() as Record<string, unknown>, schema));
    } catch (readError) {
      if (!(readError instanceof FormError)) throw readError;
      item.yamlCode.textContent = `# Chưa hợp lệ: ${readError.message}`;
    }
  }

  function renderSummary(item: Item) {
    const [title = "", ...rest] = item.body.summary!();
    item.title.textContent = title;
    item.detail.textContent = rest.join(" · ");
  }

  function refresh() {
    count.textContent = `${items.length}/${maxItems === Infinity ? "∞" : maxItems}`;
    addButton.disabled = items.length >= maxItems;
    for (const item of items) {
      item.remove.disabled = items.length <= minItems;
      item.duplicate.disabled = items.length >= maxItems;
    }
  }

  // Tên (khoá "name") không được trùng: lab báo lỗi biến thể trùng tên.
  function uniqueName(base: string): string {
    const names = new Set(items.map(item => (safeRead(item.body) as Record<string, unknown>).name));
    if (!names.has(base)) return base;
    let index = 2;
    while (names.has(`${base}_${index}`)) index++;
    return `${base}_${index}`;
  }

  function addItem(value: Record<string, unknown>, after?: Item) {
    const body = objectControl(itemPath, schema, hints, description, value);
    const card = document.createElement("div");
    card.className = "lab-item";
    const bar = document.createElement("div");
    bar.className = "lab-item-bar";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "lab-item-toggle";
    toggle.innerHTML = iconSvg("chevron-down");
    const title = document.createElement("span");
    title.className = "lab-item-title";
    const detail = document.createElement("span");
    detail.className = "lab-item-detail";
    toggle.append(title, detail);
    const showYaml = iconButton("code", "Xem YAML");
    showYaml.setAttribute("aria-pressed", "false");
    const duplicate = iconButton("copy", "Nhân bản");
    const remove = iconButton("trash", "Xoá");
    bar.append(toggle, showYaml, duplicate, remove);
    const yaml = document.createElement("div");
    yaml.className = "lab-yaml";
    yaml.hidden = true;
    const yamlCode = document.createElement("pre");
    const copyYaml = iconButton("copy", "Sao chép YAML");
    copyYaml.addEventListener("click", () => copyToClipboard(copyYaml, yamlCode.textContent ?? ""));
    yaml.append(yamlCode, copyYaml);
    card.append(bar, yaml, body.element);
    const item: Item = { card, body, toggle, title, detail, showYaml, yaml, yamlCode, duplicate, remove };
    if (after) {
      after.card.after(card);
      items.splice(items.indexOf(after) + 1, 0, item);
    } else {
      itemsElement.append(card);
      items.push(item);
    }

    toggle.addEventListener("click", () => setOpen(item, card.dataset.open !== "true"));
    showYaml.addEventListener("click", () => {
      yaml.hidden = !yaml.hidden;
      showYaml.setAttribute("aria-pressed", String(!yaml.hidden));
      renderYaml(item);
    });
    for (const kind of ["input", "change"]) {
      body.element.addEventListener(kind, () => {
        renderSummary(item);
        renderYaml(item);
      });
    }
    duplicate.addEventListener("click", () => {
      error.hidden = true;
      let copy: Record<string, unknown>;
      try {
        copy = body.read() as Record<string, unknown>;
      } catch (readError) {
        if (!(readError instanceof FormError)) throw readError;
        error.textContent = `Chưa nhân bản được: ${readError.message}`;
        error.hidden = false;
        return;
      }
      setOpen(addItem({ ...copy, name: uniqueName(String(copy.name ?? "bien_the")) }, item), true);
    });
    remove.addEventListener("click", () => {
      items.splice(items.indexOf(item), 1);
      card.remove();
      refresh();
      notifyChange(container);
    });

    renderSummary(item);
    setOpen(item, false);
    refresh();
    notifyChange(container);
    return item;
  }

  for (const value of Array.isArray(initial) ? initial : []) addItem(value as Record<string, unknown>);
  addButton.addEventListener("click", () => setOpen(addItem({ name: uniqueName("bien_the") }), true));
  refresh();
  return { element: container, read: () => items.map(item => item.body.read()) };
}

function fieldControl(path: string, schema: FieldSchema, hints: FieldHints, description: Description, initial: unknown, advanced: AdvancedGroup): Control {
  const hint = hints[path] ?? {};
  const id = nextId();
  const label = hint.label ?? path.split(".").at(-1)!;
  // Khoá có "filter_by" bắt đầu rỗng; objectControl nạp lựa chọn khi đã có ô của khoá anh em.
  const values = hint.filter_by ? [] : hint.source ? sourceValues(hint.source, description) : hint.choices ?? (schema.type === "choice" ? schema.values : null);
  if (values) {
    return hint.widget === "segmented" ? segmentedControl(values, initial) : selectControl(id, values, initial, hint, label);
  }
  switch (schema.type) {
    case "bool": {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = id;
      input.checked = initial === true;
      return { element: input, read: () => input.checked };
    }
    case "int":
    case "float":
      return hint.widget === "slider" ? sliderControl(id, schema, hint, initial) : numberControl(id, schema, hint, label, initial);
    case "str": {
      const input = document.createElement("input");
      input.type = "text";
      input.id = id;
      input.value = typeof initial === "string" ? initial : "";
      return { element: input, read: () => input.value.trim() };
    }
    case "object":
      // Object trải phẳng gom tham số nâng cao vào cùng mục "Nâng cao" của object cha.
      return objectControl(path, schema, hints, description, initial, hint.inline ? advanced : undefined);
    case "list":
      if (schema.items.type === "str" && hint.widget === "lines") return linesControl(id, hint, label, initial);
      if (schema.items.type === "object") return listOfObjectsControl(path, schema.items, hint, hints, description, initial);
  }
  throw new Error(`Khoá "${path}" có kiểu ${schema.type} chưa có ô nhập; thêm "source" hoặc "hidden" trong lab_fields.json`);
}

function fieldWrapper(control: Control, label: string, hint: FieldHint, isCheckbox: boolean): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = ["lab-field", isCheckbox && "lab-field-inline", hint.width === "half" && "lab-field-half"].filter(Boolean).join(" ");
  const labelElement = document.createElement("label");
  labelElement.textContent = label;
  const target = control.element;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) {
    labelElement.htmlFor = target.id;
  } else {
    // Ô tự dựng (segmented, slider, chip) không gắn được <label for>: đặt tên bằng aria-labelledby.
    labelElement.id = nextId();
    const labelled = target.querySelector("input, textarea") ?? target;
    labelled.setAttribute("aria-labelledby", labelElement.id);
    if (target.getAttribute("role") === "radiogroup") target.setAttribute("aria-labelledby", labelElement.id);
  }
  wrapper.append(labelElement, target);
  if (hint.help) {
    const help = document.createElement("p");
    help.className = "t-caption";
    help.textContent = hint.help;
    wrapper.append(help);
  }
  return wrapper;
}

// Mục "Nâng cao" thu gọn của một object, kèm số tham số đang khác mặc định.
interface AdvancedEntry { control: Control; wrapper: HTMLElement; fallback: unknown }
interface AdvancedGroup { details: HTMLDetailsElement; body: HTMLElement; entries: AdvancedEntry[]; refresh(): void }

function advancedGroup(): AdvancedGroup {
  const details = document.createElement("details");
  details.className = "lab-advanced";
  const summary = document.createElement("summary");
  summary.innerHTML = iconSvg("chevron-down");
  const title = document.createElement("span");
  title.textContent = "Nâng cao";
  const count = document.createElement("span");
  count.className = "lab-advanced-count";
  summary.append(title, count);
  const body = document.createElement("div");
  body.className = "lab-object";
  details.append(summary, body);
  const group: AdvancedGroup = {
    details, body, entries: [],
    refresh() {
      const changed = group.entries.filter(entry =>
        !entry.wrapper.hidden && JSON.stringify(safeRead(entry.control)) !== JSON.stringify(entry.fallback ?? null),
      ).length;
      count.textContent = changed > 0 ? `${changed} đã đổi` : "";
    },
  };
  body.addEventListener("input", group.refresh);
  body.addEventListener("change", group.refresh);
  return group;
}

// Ô nhập của một object: mỗi khoá (trừ khoá ẩn) một ô. Giá trị ban đầu: initial của object cha, rồi
// "initial" trong lab_fields.json, rồi mặc định trong schema. ``advanced``: mục "Nâng cao" của object cha khi
// object này được trải phẳng (inline); không truyền thì object tự có mục riêng nếu có khoá nâng cao.
export function objectControl(path: string, schema: ObjectSchema, hints: FieldHints, description: Description, initial: unknown, advanced?: AdvancedGroup): Control {
  const element = document.createElement("div");
  element.className = path ? "lab-object" : "lab-object lab-object-root";
  const ownsAdvanced = advanced === undefined;
  const advancedHost = advanced ?? advancedGroup();
  const given = (initial ?? {}) as Record<string, unknown>;
  const controls: Record<string, Control> = {};
  const wrappers: Record<string, HTMLElement> = {};
  const summaryParts: (() => string[])[] = [];

  for (const [name, field] of Object.entries(schema.fields)) {
    const fieldPath = childPath(path, name);
    const hint = hints[fieldPath] ?? {};
    if (hint.hidden) continue;
    const value = given[name] ?? hint.initial ?? field.default;
    const control = fieldControl(fieldPath, field, hints, description, value, advancedHost);
    controls[name] = control;
    const label = hint.label ?? name;

    if (field.type === "object" && hint.inline) {
      element.append(control.element);
      summaryParts.push(control.summary!);
      continue;
    }
    let wrapper: HTMLElement;
    if (field.type === "object" || (field.type === "list" && field.items.type === "object")) {
      wrapper = document.createElement("section");
      wrapper.className = "lab-section";
      const heading = document.createElement("h3");
      heading.className = "lab-section-title";
      heading.textContent = label;
      wrapper.append(heading, control.element);
    } else {
      wrapper = fieldWrapper(control, label, hint, field.type === "bool");
    }
    wrapper.dataset.path = fieldPath;
    wrappers[name] = wrapper;
    if (hint.group === "advanced") {
      advancedHost.body.append(wrapper);
      advancedHost.entries.push({ control, wrapper, fallback: field.default });
    } else {
      element.append(wrapper);
    }
    if (hint.summary) {
      const template = hint.summary;
      summaryParts.push(() => (wrapper.hidden ? [] : [template.replace("{}", formatValue(safeRead(control)))]));
    }
  }

  // Ẩn/hiện theo khoá anh em: chỉ đổi hiển thị, giá trị vẫn được gửi (lab bỏ qua tham số không dùng).
  for (const [name, wrapper] of Object.entries(wrappers)) {
    const condition = hints[childPath(path, name)]?.visible_when;
    if (!condition) continue;
    const update = () => {
      wrapper.hidden = !Object.entries(condition).every(([sibling, values]) => values.includes(safeRead(controls[sibling])));
      advancedHost.refresh();
    };
    for (const sibling of Object.keys(condition)) {
      if (!controls[sibling]) throw new Error(`lab_fields.json: "${childPath(path, name)}" phụ thuộc khoá "${sibling}" không có`);
      controls[sibling].element.addEventListener("change", update);
    }
    update();
  }

  for (const [name, control] of Object.entries(controls)) {
    const hint = hints[childPath(path, name)] ?? {};
    if (!hint.filter_by) continue;
    const sibling = controls[hint.filter_by];
    if (!sibling) throw new Error(`lab_fields.json: "${childPath(path, name)}" lọc theo khoá "${hint.filter_by}" không có`);
    const update = () => control.setChoices!(filteredValues(hint.source!, description, safeRead(sibling)));
    sibling.element.addEventListener("change", update);
    update();
  }

  if (ownsAdvanced && advancedHost.entries.length > 0) {
    element.append(advancedHost.details);
    advancedHost.refresh();
  }

  return {
    element,
    read: () => Object.fromEntries(Object.entries(controls).map(([name, control]) => [name, control.read()])),
    summary: () => summaryParts.flatMap(part => part()),
    children: controls,
  };
}
