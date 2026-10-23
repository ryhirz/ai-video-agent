/* 前端阶段2 验收验证（node，无浏览器）：
 * 1) 以 DOM 桩真实加载 editor.html 的 <script>；
 * 2) 正例：speed+grayscale+add_text 组合 → validate 通过 + buildTimeline 形态正确；
 * 3) 反例：transform.x 负偏移 / speed 越界 → 前端拦截且提示可读；
 * 4) 正例 timeline 写盘供 python 侧真渲染。
 * 运行：node tests/editor_verify.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "server/static/editor.html"), "utf-8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];

/* ---- DOM 桩：够编辑器顶层执行 + 交互函数调用即可 ---- */
function makeEl() {
  return {
    innerHTML: "", value: "", textContent: "", style: {}, checked: false,
    className: "", children: [],
    appendChild(c) { this.children.push(c); },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    closest() { return null; },
    setAttribute() {}, getAttribute() { return null; },
    onclick: null, onchange: null,
  };
}
const elCache = {};
globalThis.document = {
  createElement: () => makeEl(),
  getElementById: (id) => (elCache[id] ??= makeEl()),
};
const inputs = {};                        // 模拟用户输入框的值
globalThis.fetch = () => new Promise(() => {});   // manifest 拉取挂起 → 走兜底

const window = globalThis;
/* ESM 严格模式下 eval 不泄漏 var —— 用 new Function 包裹并显式引出 API */
const api = new Function(
  "document", "fetch", "window",
  js + "\n;return { state, validate, buildTimeline, renderAll };")(
  globalThis.document, globalThis.fetch, window);
const { state, validate, buildTimeline } = api;

/* 注入 manifest 兜底开关：强制走 FALLBACK（fetch 已挂起） */
// 直接操纵 state（模拟用户在面板上的编辑结果）
state.clips = [
  { id: "v1", track: "v_main", sourceRef: "E:/media/sample.mp4",
    "in": 0, out: 4, sourceIn: 0, sourceOut: 4,
    transform: { x: 0, y: 0, scale: 1.0, rotation: 0, opacity: 1.0, crop: null },
    effects: [{ effectType: "speed", params: { rate: 1.5 } },
              { effectType: "grayscale", params: {} }] },
  { id: "a1", track: "a_main", sourceRef: "E:/media/sample.mp4",
    "in": 0, out: 4, sourceIn: 0, sourceOut: 4,
    transform: { x: 0, y: 0, scale: 1.0, rotation: 0, opacity: 1.0, crop: null },
    effects: [{ effectType: "speed", params: { rate: 1.5 } }] },
];
state.texts = [{ id: "tx1", text: "水印TEST", start: 0, end: 2,
  position: "top-right", margin: 20, size: 28, color: "#FFFFFF" }];

/* ---- 正例：validate 通过 + buildTimeline 形态 ---- */
let errs = validate();
if (errs.length) { console.error("正例误报:", errs); process.exit(1); }
const tl = buildTimeline();
const cv = tl.clips[0];
const okFx = cv.effects.some(e => e.effectType === "speed" && e.params.rate === 1.5)
          && cv.effects.some(e => e.effectType === "grayscale");
if (!okFx || tl.texts.length !== 1 || tl.duration !== 4
    || cv.transform.x !== 0 || tl.schemaVersion !== "1.0.0") {
  console.error("buildTimeline 形态不符:", JSON.stringify(tl, null, 1));
  process.exit(1);
}
console.log("PASS 正例：validate 通过，组合 timeline 形态正确");

/* ---- 反例 1：transform.x 负偏移 → 前端拦截 ---- */
state.clips[0].transform.x = -5;
errs = validate();
if (!errs.some(e => e.includes("transform.x 不可为负"))) {
  console.error("负偏移未被拦截:", errs); process.exit(1);
}
console.log("PASS 反例1：transform.x=-5 被前端拦截 →", errs[0].trim());

/* ---- 反例 2：speed rate 越界 → 前端拦截 ---- */
state.clips[0].transform.x = 0;
state.clips[0].effects[0].params.rate = 2000;
errs = validate();
if (!errs.some(e => e.includes("speed rate"))) {
  console.error("speed 越界未被拦截:", errs); process.exit(1);
}
console.log("PASS 反例2：rate=2000 被前端拦截 →",
            errs.find(e => e.includes("speed rate")).trim());

/* ---- 还原正例并写盘供 python 真渲染 ---- */
state.clips[0].effects[0].params.rate = 1.5;
const final = buildTimeline();
writeFileSync(join(root, "server/server_data/_editor_combo.json"),
              JSON.stringify(final, null, 1));
console.log("组合 timeline 已写盘：server/server_data/_editor_combo.json");
