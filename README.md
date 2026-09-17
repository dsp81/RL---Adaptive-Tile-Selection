<!-- This branch is the deployed site, not project history. -->
# Adaptive HR Tile Selection — deployed viewer (`gh-pages`)

This branch holds **only the built static site** served at
<https://dsp81.github.io/RL---Adaptive-Tile-Selection/>. The project itself lives on `main`.
Regenerate it with `python prerender.py` after refreshing `data/`.

Static site (no build step, no framework, no API keys) for the Kaggle run of the AAAI-2021
adaptive high-resolution tile acquisition method on the Uganda LSMS poverty dataset.

```
index.html      shell + the prerendered Findings report (between the PRERENDER markers)
app.js          map, charts and the per-cluster detail panel, from data/data.json
style.css       tokens, light/dark, layout
prerender.py    bakes data/data.json into static HTML, report.md, llms.txt, clusters.csv
netlify.toml    publish dir + content types (so .md serves as text, not a download)
report.md       generated — the whole run as Markdown
llms.txt        generated — llmstxt.org index
robots.txt      generated — explicit allow
data/           the Kaggle run's frontend_assets.zip, unzipped
vendor/leaflet  vendored, not CDN-loaded: a slow CDN would otherwise stall the parser
```

## Deploy

```sh
../fetch_results.sh     # downloads the run, unzips into data/, prerenders the report
python prerender.py     # only if you edited data/ or the template by hand
```

Then **drag this folder onto <https://app.netlify.com/drop>**. Nothing to install, and
`netlify.toml` travels with the folder, so the content-type headers apply.

The CLI route (`npx netlify-cli deploy --prod --dir=.`) does not work on this machine as
set up: netlify-cli now requires Node ≥ 20, the system Node is 10 (with an npm that refuses
to run on it), and the `nodejs-bin` pip package here is 18.4. Install Node 20+ first if you
want the CLI.

## Readable without JavaScript

The Findings tab is not rendered by `app.js` — `prerender.py` writes it into `index.html` as
plain HTML, tables and all, and the tab is the page's landing view. So a crawler, an LLM
fetcher, a chat unfurl or `curl https://<site>/` sees the entire report (~42k characters of
text, every number, every table) without executing a line of script. The interactive tabs
render the same `data/data.json` on top.

Three redundant entry points, in descending order of how much a machine has to parse:

| URL | What it is |
|---|---|
| `/llms.txt` | short index: headline numbers, then links to everything else |
| `/report.md` | the whole report as Markdown, served as `text/markdown` |
| `/` | the report as static HTML, plus JSON-LD `Dataset` metadata in `<head>` |
| `/data/data.json` | the run itself: config, metrics, detector report, SHAP, per-cluster masks |
| `/data/clusters.csv` | per-cluster predictions and budget spent, flat |

**Never hand-edit `index.html` between the `PRERENDER` markers**, or the next
`prerender.py` run will overwrite it. Edit the generator instead — the prose in there is
parameterised from the numbers, so a new run cannot leave a stale claim on the page.

## Local preview

```sh
python -m http.server 8000     # then open http://localhost:8000
```

Opening `index.html` from disk works for the Findings tab but not the interactive ones:
browsers block `fetch()` on `file://`, and the page says so when it happens.
