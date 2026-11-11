// Tiny test runner: imports every test_*.mjs in this directory.

import { readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = dirname(fileURLToPath(import.meta.url));
const files = readdirSync(dir).filter((f) => f.startsWith('test_') && f.endsWith('.mjs')).sort();

let passed = 0;
let failed = 0;
let skipped = 0;

globalThis.test = async (name, fn) => {
  try {
    const r = await fn();
    if (r === 'SKIP') {
      skipped++;
      console.log(`  SKIP ${name}`);
    } else {
      passed++;
      console.log(`  ok   ${name}`);
    }
  } catch (e) {
    failed++;
    console.error(`  FAIL ${name}`);
    console.error(`       ${e.message}`);
    if (e.stack) console.error(e.stack.split('\n').slice(1, 4).join('\n'));
  }
};

for (const f of files) {
  console.log(`\n${f}`);
  await import(join(dir, f));
}

console.log(`\n${passed} passed, ${failed} failed, ${skipped} skipped`);
process.exit(failed === 0 ? 0 : 1);
