#!/usr/bin/env node
/**
 * Unattended Bilibili upload.
 *
 * Bilibili exposes no投稿 API to ordinary accounts, so this drives the real
 * upload page in a persistent Chrome profile. The profile directory is what
 * carries the login: once the user signs in there, every later run reuses it,
 * and when the cookies expire the run fails with `login_required` instead of
 * guessing - re-signing in is the only manual step in the whole pipeline.
 *
 * Usage:  node publish.js <payload.json>
 * Result: one line `RESULT {json}` on stdout, progress on stderr.
 *
 * payload keys:
 *   video, cover        absolute paths (cover optional)
 *   title, description, tags[], zone, collection, declare
 *   chrome, profile     browser executable + user-data-dir
 *   headless            default true
 *   dryRun              fill everything, stop before submitting
 */
'use strict';

const fs = require('fs');
const { chromium } = require('playwright-core');

const payloadPath = process.argv[2];
if (!payloadPath) {
  emit({ ok: false, stage: 'init', error: 'payload path is required' });
  process.exit(1);
}

const payload = JSON.parse(fs.readFileSync(payloadPath, 'utf8'));
const {
  video = '',
  cover = '',
  title = '',
  description = '',
  tags = [],
  zone = '',
  collection = '',
  declare = '内容无需标注',
  chrome = '',
  profile = '',
  headless = true,
  dryRun = false,
  shotDir = '',
  uploadTimeoutMs = 60 * 60 * 1000,
  settleMs = 6000,
} = payload;

const UPLOAD_URL = 'https://member.bilibili.com/platform/upload/video/frame';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (message) => process.stderr.write(`[bili] ${message}\n`);
const emit = (obj) => process.stdout.write('RESULT ' + JSON.stringify(obj) + '\n');

let stage = 'init';
let context = null;

class StepError extends Error {
  constructor(stageName, message) {
    super(message);
    this.stageName = stageName;
  }
}

function fail(stageName, message) {
  throw new StepError(stageName, message);
}

async function shot(page, name) {
  if (!shotDir) return;
  try {
    await page.screenshot({ path: require('path').join(shotDir, `bili_${name}.png`), fullPage: false });
  } catch (err) {
    log(`screenshot ${name} failed: ${err.message}`);
  }
}

/** Visible text of the page, whitespace collapsed. */
async function bodyText(page) {
  return page.evaluate(() => (document.body ? document.body.innerText : '').replace(/\s+/g, ' '));
}

async function waitForText(page, pattern, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const text = await bodyText(page);
    if (pattern.test(text)) return text;
    await sleep(2000);
  }
  fail(stage, `timed out waiting for ${what}`);
}

/** Click the first visible element whose trimmed text equals `text`. */
async function clickExact(page, text) {
  return page.evaluate((wanted) => {
    const leaves = Array.from(document.querySelectorAll('*')).filter(
      (el) => el.children.length === 0
        && (el.innerText || '').trim() === wanted
        && el.getBoundingClientRect().width > 0
    );
    if (!leaves.length) return false;
    leaves[leaves.length - 1].click();
    return true;
  }, text);
}

/**
 * Click an option in an open dropdown, tolerating the page's own wording.
 * Exact match first, then a containment match, so "单机游戏" still resolves
 * when the model answered "游戏".
 */
async function clickOption(page, wanted) {
  const result = await page.evaluate((want) => {
    const visible = Array.from(document.querySelectorAll('*')).filter(
      (el) => el.children.length === 0
        && (el.innerText || '').trim()
        && el.getBoundingClientRect().width > 0
        && el.getBoundingClientRect().top > 0
    );
    const pick = (el) => {
      let node = el;
      for (let i = 0; i < 5 && node; i += 1) {
        if (/option|item|list|select/i.test(String(node.className || ''))) {
          node.click();
          return true;
        }
        node = node.parentElement;
      }
      el.click();
      return true;
    };
    let target = visible.find((el) => el.innerText.trim() === want);
    if (!target) target = visible.find((el) => el.innerText.trim().replace(/\s+/g, '') === want.replace(/\s+/g, ''));
    if (!target) target = visible.find((el) => el.innerText.includes(want));
    if (!target) target = visible.find((el) => want.includes(el.innerText.trim()) && el.innerText.trim().length >= 2);
    if (!target) return 'not-found';
    return pick(target) ? 'clicked' : 'failed';
  }, wanted);
  return result;
}

/** Read the currently selected value of the zone dropdown. */
async function currentZone(page) {
  return page.evaluate(() => {
    const el = document.querySelector('.select-controller .select-item-cont')
      || document.querySelector('.select-controller');
    return el ? (el.innerText || '').trim().split('\n')[0] : '';
  });
}

async function fillForm(page) {
  // --- title ---
  stage = 'form';
  const titleInput = page.locator('input[placeholder="请输入稿件标题"]').first();
  await titleInput.waitFor({ state: 'visible', timeout: 30000 });
  await titleInput.click();
  await titleInput.fill('');
  await titleInput.fill(title);
  log(`title set (${title.length} chars)`);

  // --- description (Quill editor) ---
  const editor = page.locator('.ql-editor').first();
  await editor.waitFor({ state: 'visible', timeout: 30000 });
  await editor.click();
  await editor.evaluate((el, text) => {
    el.innerHTML = '';
    el.textContent = text;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, description);
  log(`description set (${description.length} chars)`);

  // --- tags ---
  const box = page.locator('#tag-container').first();
  if (await box.count()) {
    for (let i = 0; i < 25; i += 1) {
      const close = box.locator('.label-item-v2-container svg.close').first();
      if ((await close.count()) === 0) break;
      await close.click().catch(() => {});
      await sleep(300);
    }
    const input = box.locator('input[placeholder*="创建标签"]').first();
    if (await input.count()) {
      for (const tag of tags) {
        await input.click().catch(() => {});
        await input.fill(tag).catch(() => {});
        await input.press('Enter').catch(() => {});
        await sleep(650);
      }
      log(`tags set (${tags.length})`);
    } else {
      log('tag input not found; keeping the default tags');
    }
  }
}

async function setCover(page) {
  if (!cover || !fs.existsSync(cover)) {
    log('no cover file; leaving the cover empty');
    return;
  }
  stage = 'cover';
  const already = await page.evaluate(() => !!document.querySelector('.cover-img, .cover-preview-img'));
  if (already) {
    log('cover already present; skipping');
    return;
  }
  const opened = await page.evaluate(() => {
    const el = document.querySelector('.cover-empty')
      || Array.from(document.querySelectorAll('*')).find(
        (node) => node.children.length === 0 && (node.innerText || '').trim() === '添加封面'
      );
    if (el) { el.click(); return true; }
    return false;
  });
  if (!opened) fail('cover', 'cover entry not found');
  await sleep(3500);

  const imageInput = page.locator('input[accept="image/png, image/jpeg"]').first();
  await imageInput.waitFor({ state: 'attached', timeout: 30000 });
  await imageInput.setInputFiles(cover);
  await sleep(4000);

  const confirmed = await clickExact(page, '完成');
  if (!confirmed) log('cover confirm button not found; the dialog may already be closed');
  await sleep(2000);
  log('cover set');
}

async function setZone(page, wanted) {
  if (!wanted) {
    log('no zone suggested; keeping the page default');
    return '';
  }
  stage = 'zone';
  const before = await currentZone(page);
  await page.evaluate(() => {
    const el = document.querySelector('.select-controller');
    if (el) el.click();
  });
  await sleep(1800);
  let outcome = await clickOption(page, wanted);
  if (outcome === 'not-found') {
    log(`zone "${wanted}" not in the dropdown; keeping "${before}"`);
    await page.keyboard.press('Escape').catch(() => {});
    await sleep(600);
    return before;
  }
  await sleep(1200);
  const after = await currentZone(page);
  log(`zone: "${before}" -> "${after}" (wanted "${wanted}")`);
  return after;
}

async function setCollection(page, wanted) {
  if (!wanted) {
    log('no collection suggested; skipping');
    return '';
  }
  stage = 'collection';
  await page.evaluate(() => {
    const el = document.querySelector('.season-select .season-enter') || document.querySelector('.season-enter');
    if (el) el.click();
  });
  await sleep(1800);
  const outcome = await clickOption(page, wanted);
  if (outcome === 'not-found') {
    log(`collection "${wanted}" not in the dropdown; skipping`);
    await page.keyboard.press('Escape').catch(() => {});
    await sleep(600);
    return '';
  }
  await sleep(1200);
  log(`collection set to "${wanted}"`);
  return wanted;
}

async function setDeclaration(page, wanted) {
  stage = 'declare';
  const box = page.locator('.creation-statement-container');
  if ((await box.count()) === 0) {
    log('declaration block not found; the page layout may have changed');
    return '';
  }
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await box.locator('.bcc-select').first().click().catch(() => {});
    await sleep(1500);
    const option = box.locator('li.bcc-option').filter({ hasText: wanted }).first();
    if ((await option.count()) > 0) {
      await option.click().catch(() => {});
      await sleep(1500);
      const value = await box.locator('input.bcc-select-input-inner').first().inputValue().catch(() => '');
      if (value.includes(wanted)) {
        log(`declaration set to "${value}"`);
        return value;
      }
    }
  }
  fail('declare', `could not select the creation declaration "${wanted}"`);
}

async function uploadVideo(page) {
  stage = 'upload';
  const resuming = await page.evaluate(() => /更换视频/.test(document.body.innerText));
  if (resuming) {
    log('a finished upload is already on the page (draft restored); reusing it');
    return;
  }
  // Bilibili drives the upload from a custom control and keeps the real
  // <input type=file> hidden, so it is never "visible" - wait for it to be
  // attached instead. It is handed the file through CDP below, which does not
  // care about visibility.
  await page.waitForSelector('input[type=file][accept*="mp4"]', {
    state: 'attached',
    timeout: 90000,
  });
  await sleep(2000);

  const client = await context.newCDPSession(page);
  const { root } = await client.send('DOM.getDocument', { depth: -1 });
  const { nodeId } = await client.send('DOM.querySelector', {
    nodeId: root.nodeId,
    selector: 'input[type=file][accept*="mp4"]',
  });
  if (!nodeId) fail('upload', 'video file input not found');
  // Files above 50 MB cannot go through setInputFiles over CDP, so the browser
  // is told to read the path itself instead of receiving the bytes.
  await client.send('DOM.setFileInputFiles', { files: [video], nodeId });
  log(`upload started: ${video}`);

  await waitForText(page, /更换视频/, uploadTimeoutMs, 'the upload to finish');
  log('upload finished');
  await sleep(settleMs);
}

async function submit(page) {
  stage = 'submit';
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await sleep(800);
  const clicked = await clickExact(page, '立即投稿');
  if (!clicked) fail('submit', '"立即投稿" button not found');
  log('submit clicked');

  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    await sleep(3000);
    const text = await bodyText(page);
    if (/稿件投递成功|投稿成功/.test(text)) return;
    if (/请添加创作声明|请选择创作声明/.test(text)) {
      fail('submit', 'Bilibili rejected the submission because of the creation declaration');
    }
    if (/请填写|请输入标题|请添加/.test(text)) {
      const hint = (text.match(/请[^ ]{2,20}/) || [''])[0];
      fail('submit', `Bilibili rejected the submission: ${hint}`);
    }
  }
  fail('submit', 'timed out waiting for the submission result');
}

function bvidFromText(text) {
  const match = (text || '').match(/BV[0-9A-Za-z]{10}/);
  return match ? match[0] : '';
}

async function bvidOnPage(page) {
  return page.evaluate(() => {
    const link = document.querySelector('a[href*="/video/BV"]');
    if (link) {
      const match = link.href.match(/BV[0-9A-Za-z]{10}/);
      if (match) return match[0];
    }
    const match = (document.body.innerText || '').match(/BV[0-9A-Za-z]{10}/);
    return match ? match[0] : '';
  });
}

async function readBvid(page) {
  stage = 'bvid';
  let bvid = await bvidOnPage(page);
  if (!bvid) {
    const clicked = await clickExact(page, '查看进度');
    if (clicked) {
      await sleep(6000);
      bvid = await bvidOnPage(page);
    }
  }
  if (!bvid) {
    // The success screen does not always expose the id. The upload manager
    // lists the newest submission first - match the title we just sent, and
    // fall back to the first row so the id still gets recorded.
    try {
      await page.goto('https://member.bilibili.com/platform/upload-manager/article', {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
      });
      await sleep(6000);
      const rows = await page.evaluate(() =>
        Array.from(document.querySelectorAll('a[href*="/video/BV"]')).map((link) => ({
          href: link.href,
          text: (link.innerText || '').trim(),
          row: ((link.closest('.video-card, .article-card, .video-item, tr, li') || {}).innerText || '')
            .replace(/\s+/g, ' ')
            .trim(),
        })),
      );
      const wanted = (title || '').replace(/\s+/g, '');
      const match = wanted
        ? rows.find((row) => (row.row + row.text).replace(/\s+/g, '').includes(wanted.slice(0, 12)))
        : null;
      bvid = bvidFromText((match || rows[0] || {}).href);
      if (bvid) log(`bvid ${bvid} (from the upload manager)`);
    } catch (error) {
      log(`could not read the upload manager for the id: ${error.message}`);
    }
  }
  if (bvid) log(`bvid ${bvid}`);
  else log('bvid not found on the page; the submission itself succeeded');
  return bvid;
}

async function main() {
  if (!video || !fs.existsSync(video)) fail('init', `video not found: ${video}`);
  if (!chrome) fail('init', 'chrome executable path is required');
  if (!profile) fail('init', 'profile directory is required');

  stage = 'launch';
  log(`launching Chrome (headless=${headless})`);
  context = await chromium.launchPersistentContext(profile, {
    headless,
    executablePath: chrome,
    args: [
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-session-crashed-bubble',
      '--hide-crash-restore-bubble',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--window-size=1600,1200',
    ],
    viewport: { width: 1600, height: 1200 },
    acceptDownloads: false,
  });

  const page = context.pages()[0] || (await context.newPage());

  stage = 'login';
  await page.goto(UPLOAD_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await sleep(8000);
  if (!page.url().includes('member.bilibili.com')) {
    fail('login', 'not signed in (redirected to the login page)');
  }
  log('signed in; on the upload page');

  await uploadVideo(page);
  await shot(page, 'uploaded');
  await fillForm(page);
  await setCover(page);
  const finalZone = await setZone(page, zone);
  const finalCollection = await setCollection(page, collection);
  await setDeclaration(page, declare);
  await shot(page, 'filled');

  if (dryRun) {
    log('dry run: stopping before submit');
    return { ok: true, stage: 'dry-run', bvid: '', zone: finalZone, collection: finalCollection };
  }

  await submit(page);
  await shot(page, 'submitted');
  const bvid = await readBvid(page);
  return { ok: true, stage: 'done', bvid, zone: finalZone, collection: finalCollection };
}

main()
  .then((result) => {
    emit(result);
    return context ? context.close().catch(() => {}) : null;
  })
  .then(() => process.exit(0))
  .catch(async (err) => {
    const stageName = err instanceof StepError ? err.stageName : stage;
    emit({ ok: false, stage: stageName, error: err && err.message ? err.message : String(err) });
    if (context) await context.close().catch(() => {});
    process.exit(1);
  });
