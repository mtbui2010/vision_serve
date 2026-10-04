// Parse every ```mermaid block in the repository's Markdown files with the real mermaid parser,
// so a diagram that GitHub or the docs site would show as "Unable to render" fails CI instead.
// (A ';' inside a sequence-diagram message broke the README's diagram this way.)
//
//   npm install --no-save mermaid@11 jsdom@24
//   node website/tools/check_mermaid.mjs [root]        # default root: the current directory
import { JSDOM } from "jsdom";
import fs from "fs";
import path from "path";

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.DOMParser = dom.window.DOMParser;
globalThis.Element = dom.window.Element;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.Node = dom.window.Node;

const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false });

const skip = new Set([".git", ".claude", "node_modules", "site"]);
function* markdown(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (skip.has(e.name)) continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* markdown(p);
    else if (e.name.endsWith(".md")) yield p;
  }
}

const root = process.argv[2] || ".";
let bad = 0, n = 0;
for (const file of markdown(root)) {
  const src = fs.readFileSync(file, "utf8");
  const re = /```mermaid\n([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(src))) {
    n++;
    const line = src.slice(0, m.index).split("\n").length;
    try {
      await mermaid.parse(m[1]);
    } catch (e) {
      bad++;
      const msg = String(e.message || e).split("\n").slice(0, 3).join(" | ");
      console.log(`${file}:${line}: ${msg}`);
    }
  }
}
console.log(`${n} mermaid diagrams, ${bad} failed`);
process.exit(bad ? 1 : 0);
