// Checks that every link between pages of the built site, and every
// #fragment, points to something that exists.
//
//   node scripts/check-links.mjs [dist directory]

import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const dist = process.argv[2] ?? 'dist';

// The base path of the site, as in astro.config.mjs, always ending in a
// slash, which SITE_URL may omit.
const basePath = new URL(process.env.SITE_URL ?? 'http://localhost:4321/').pathname.replace(/\/?$/, '/');

function htmlFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return htmlFiles(path);
    return name.endsWith('.html') ? [path] : [];
  });
}

// The URL path of a built page, e.g. dist/start/why/index.html -> /start/why/
// for the base path /.
function urlOf(file) {
  const path = basePath + relative(dist, file).split(sep).join('/');
  return path.endsWith('/index.html') ? path.slice(0, -'index.html'.length) : path;
}

// The file a URL path refers to, if any.
function fileOf(urlPath) {
  if (!urlPath.startsWith(basePath)) return undefined;
  const sitePath = urlPath.slice(basePath.length);
  const path = join(dist, ...decodeURIComponent(sitePath).split('/'));
  if (existsSync(join(path, 'index.html'))) return join(path, 'index.html');
  if (existsSync(path) && statSync(path).isFile()) return path;
  return undefined;
}

const ids = new Map();
function idsOf(file) {
  if (!ids.has(file)) {
    const html = readFileSync(file, 'utf8');
    ids.set(file, new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
  }
  return ids.get(file);
}

let errors = 0;
for (const file of htmlFiles(dist)) {
  const html = readFileSync(file, 'utf8');
  const base = new URL(urlOf(file), 'http://site');
  for (const [, href] of html.matchAll(/\shref="([^"]*)"/g)) {
    if (/^(?:[a-z]+:|\/\/)/i.test(href)) continue;
    const url = new URL(href.replaceAll('&amp;', '&'), base);
    const target = fileOf(url.pathname);
    if (!target) {
      console.error(`${urlOf(file)}: link to missing page ${href}`);
      errors++;
      continue;
    }
    const fragment = decodeURIComponent(url.hash.slice(1));
    if (fragment && target.endsWith('.html') && !idsOf(target).has(fragment)) {
      console.error(`${urlOf(file)}: link to missing anchor ${href}`);
      errors++;
    }
  }
}

if (errors > 0) {
  console.error(`${errors} broken link(s)`);
  process.exit(1);
}
console.log('All links are valid.');
