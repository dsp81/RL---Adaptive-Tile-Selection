# Adaptive HR Tile Selection — report site

The Kaggle run, rendered as one static page. **No JavaScript at all**: every number, table
and figure is written into `index.html` by `prerender.py`, so the page renders identically
for a browser, a crawler, an LLM fetcher and `curl`.

```
index.html      generated — the whole report (do not hand-edit between the PRERENDER markers)
prerender.py    the generator: data/data.json -> index.html, report.md, llms.txt, clusters.csv
style.css       tokens, light/dark, layout; figures inherit the colour tokens
report.md       generated — the report as Markdown
llms.txt        generated — llmstxt.org index
robots.txt      generated — explicit allow
data/           the Kaggle run's frontend_assets.zip, unzipped
```

## Rebuild and deploy

```sh
../fetch_results.sh          # download the run into data/ and prerender (does both)
python prerender.py          # or just re-render after editing the generator
```

Published from the `gh-pages` branch of the project repo:

```sh
git checkout gh-pages && cp -r ~/povrl/frontend/. . && touch .nojekyll
git add -A && git commit && git push
```

## Figures

Charts are inline SVG emitted by `prerender.py`, filled with the CSS custom properties from
`style.css` — so light and dark mode need no second palette and no script. Marks follow one
spec: bars ≤24px with a 4px rounded data-end and a 2px surface gap, 2px lines, ≥8px markers
with a 2px surface ring, hairline solid gridlines. The categorical pair (`--series-1` /
`--series-2`) passes the six-check colour validation — lightness band, chroma floor, CVD
separation, normal-vision floor, contrast — against both surfaces. Every mark carries an SVG
`<title>`, which browsers show as a tooltip without any JavaScript.

There is no hover/crosshair layer and no map tiles, because there is no script; anything a
tooltip would have told you is also in the table under each figure.

## Readable without JavaScript

~42,000 characters of extractable text, plus JSON-LD `Dataset` metadata in `<head>`. Machine
entry points, in ascending order of detail: `/llms.txt`, `/report.md`, `/`,
`/data/clusters.csv`, `/data/data.json`.

`netlify.toml` is kept for the drag-and-drop Netlify path; GitHub Pages ignores it and gets
the content types right on its own.

## Local preview

```sh
python -m http.server 8000     # http://localhost:8000
```

Opening `index.html` straight off disk also works now — nothing fetches anything.
