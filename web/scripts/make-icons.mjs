// Regenerates the raster icons from app/icon.svg (the source of truth for the mark's look):
//   app/favicon.ico      16, 32 and 48 px PNGs in one .ico, the fallback for browsers that ignore svg icons
//   app/apple-icon.png   180 px, full-bleed square: iOS applies its own rounded mask
// Run after changing app/icon.svg (and lib/brandMark.ts):   node scripts/make-icons.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const appDir = resolve(here, "../app");
const svg = readFileSync(resolve(appDir, "icon.svg"), "utf-8");

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });

async function render(source, size) {
  await page.setViewportSize({ width: size, height: size });
  const uri = "data:image/svg+xml;base64," + Buffer.from(source).toString("base64");
  await page.setContent(`<body style="margin:0;background:transparent"><img src="${uri}" width="${size}" height="${size}" style="display:block"></body>`);
  await page.waitForFunction(() => document.images[0].complete);
  return page.screenshot({ type: "png", omitBackground: true });
}

// .ico with PNG-compressed images (supported by every browser that reads .ico today)
function ico(images) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(images.length, 4);
  const entries = Buffer.alloc(16 * images.length);
  let offset = header.length + entries.length;
  images.forEach(({ size, png }, i) => {
    const at = i * 16;
    entries.writeUInt8(size >= 256 ? 0 : size, at); // width
    entries.writeUInt8(size >= 256 ? 0 : size, at + 1); // height
    entries.writeUInt16LE(1, at + 4); // colour planes
    entries.writeUInt16LE(32, at + 6); // bits per pixel
    entries.writeUInt32LE(png.length, at + 8);
    entries.writeUInt32LE(offset, at + 12);
    offset += png.length;
  });
  return Buffer.concat([header, entries, ...images.map(({ png }) => png)]);
}

const sizes = [16, 32, 48];
const images = [];
for (const size of sizes) images.push({ size, png: await render(svg, size) });
writeFileSync(resolve(appDir, "favicon.ico"), ico(images));

// the touch icon has no transparent corners: square it off
const square = svg.replace('rx="15"', 'rx="0"');
writeFileSync(resolve(appDir, "apple-icon.png"), await render(square, 180));

await browser.close();
console.log("wrote app/favicon.ico (" + sizes.join(", ") + " px) and app/apple-icon.png (180 px)");
