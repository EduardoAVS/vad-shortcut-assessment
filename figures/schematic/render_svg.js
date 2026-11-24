// Renders the schematic SVG figures to PNG with resvg and the IBM Plex Sans fonts (Figs. 1, 2, 3, 4, 7, 9).
// Usage (from the repo root):
//   NODE_PATH=<folder with node_modules> node figures/schematic/render_svg.js [fonts dir] [figures dir] [width px]
// Defaults: figures/fonts, figures/output, 2400 px. Renders every *.svg of the figures dir to a .png next to it.
// Dependency (not in the repo): npm install @resvg/resvg-js
const fs = require("fs");
const path = require("path");
const { Resvg } = require("@resvg/resvg-js");

const figures = path.resolve(__dirname, "..");
const [fontsDir = path.join(figures, "fonts"), svgDir = path.join(figures, "output"), width = "2400"] =
  process.argv.slice(2);
const fontFiles = fs.readdirSync(fontsDir).filter((f) => f.endsWith(".ttf")).map((f) => path.join(fontsDir, f));

for (const name of fs.readdirSync(svgDir).filter((f) => f.endsWith(".svg")).sort()) {
  const svg = fs.readFileSync(path.join(svgDir, name), "utf8");
  const png = new Resvg(svg, {
    fitTo: { mode: "width", value: Number(width) },
    background: "#FFFFFF",
    font: { fontFiles, loadSystemFonts: true, defaultFontFamily: "IBM Plex Sans" },
  }).render().asPng();
  const dest = path.join(svgDir, name.replace(/\.svg$/, ".png"));
  fs.writeFileSync(dest, png);
  console.log("ok", dest, png.length, "bytes");
}
