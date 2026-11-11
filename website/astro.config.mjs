// Documentation website of csl++.
//
// The site is static. SITE_URL sets the public URL; the base path is taken
// from its path component, so the same build works at the root of a domain
// and under a path, as on GitHub Pages.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';

// The base path needs a trailing slash, which SITE_URL may omit.
const siteUrl = new URL(process.env.SITE_URL ?? 'http://localhost:4321/');
siteUrl.pathname = siteUrl.pathname.replace(/\/?$/, '/');

// The VS Code extension's grammar also highlights CSL code on this site.
const cslGrammar = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL('../editors/vscode/syntaxes/csl.tmLanguage.json', import.meta.url),
    ),
    'utf8',
  ),
);

// MLIR listings on the internals pages use the grammar of the MLIR VS Code
// extension, copied from llvm/mlir/utils/textmate/mlir.json so that the site
// builds without the LLVM submodule.
const mlirGrammar = JSON.parse(
  readFileSync(fileURLToPath(new URL('./src/grammars/mlir.tmLanguage.json', import.meta.url)), 'utf8'),
);

export default defineConfig({
  site: siteUrl.origin,
  base: siteUrl.pathname,
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: 'csl++',
      description:
        'An open-source language frontend for the Cerebras Software Language (CSL).',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [
        {
          icon: 'github',
          label: 'Source code',
          href: 'https://github.com/esa-tu-darmstadt/csl-pp',
        },
      ],
      customCss: ['./src/styles/custom.css'],
      expressiveCode: {
        shiki: {
          langs: [
            { ...cslGrammar, name: 'csl', aliases: ['CSL'] },
            { ...mlirGrammar, name: 'mlir', aliases: ['MLIR'] },
          ],
        },
      },
      plugins: [starlightLlmsTxt()],
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Introduction', slug: 'start/introduction' },
            { label: 'Installation', slug: 'start/installation' },
            { label: 'Quickstart', slug: 'start/quickstart' },
          ],
        },
        {
          label: 'Tutorial',
          items: [{ autogenerate: { directory: 'tutorial' } }],
        },
        {
          label: 'Language',
          items: [
            { label: 'Overview', slug: 'language/overview' },
            {
              label: 'Async CSL',
              items: [
                { label: 'Async functions and tasks', slug: 'language/async' },
                { label: 'Calling async functions', slug: 'language/calls' },
                { label: 'Stacks', slug: 'language/stacks' },
              ],
            },
            {
              label: 'Resource allocation',
              items: [
                { label: 'Claims', slug: 'language/resources' },
                { label: 'Binding tasks at runtime', slug: 'language/runtime-binding' },
              ],
            },
          ],
        },
        {
          label: 'Libraries',
          items: [{ autogenerate: { directory: 'libraries' } }],
        },
        {
          label: 'Guides',
          items: [{ autogenerate: { directory: 'guides' } }],
        },
        {
          label: 'Tools',
          items: [
            { label: 'csl-pp', slug: 'tools/csl-pp' },
            { label: 'VS Code extension', slug: 'tools/vscode' },
            { label: 'Simulator tracing', slug: 'tools/simtracer' },
          ],
        },
        {
          label: 'Internals',
          items: [{ autogenerate: { directory: 'internals' } }],
        },
        {
          label: 'Project',
          items: [{ autogenerate: { directory: 'project' } }],
        },
      ],
    }),
  ],
});
