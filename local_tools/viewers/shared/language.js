'use strict';
const ViewerLanguage = (() => {
  const storageKey = 'sclvmi-language';
  let language = localStorage.getItem(storageKey) === 'en' ? 'en' : 'zh';
  const originals = new WeakMap(), attributes = new WeakMap(), dictionary = new Map(), translators = [];
  function english(text) {
    const trimmed = text.trim();
    if (dictionary.has(trimmed)) return text.replace(trimmed, dictionary.get(trimmed));
    for (const translate of translators) {
      const translated = translate(text);
      if (translated !== text) return translated;
    }
    return text;
  }
  function apply() {
    observer.disconnect();
    const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      if (['SCRIPT', 'STYLE'].includes(node.parentElement?.tagName) || node.parentElement?.closest('#language-toggle,[data-localized]')) continue;
      const old = originals.get(node);
      const source = old && node.nodeValue === old.last ? old.source : node.nodeValue;
      const next = language === 'en' ? english(source) : source;
      originals.set(node, {source, last: next});
      if (node.nodeValue !== next) node.nodeValue = next;
    }
    for (const node of document.querySelectorAll('[aria-label],[placeholder],[alt],[title]')) {
      if (node.id === 'language-toggle' || node.closest('[data-localized]')) continue;
      const record = attributes.get(node) || {};
      for (const attribute of ['aria-label', 'placeholder', 'alt', 'title']) {
        if (!node.hasAttribute(attribute)) continue;
        const current = node.getAttribute(attribute), old = record[attribute];
        const source = old && current === old.last ? old.source : current;
        const next = language === 'en' ? english(source) : source;
        record[attribute] = {source, last: next};
        if (current !== next) node.setAttribute(attribute, next);
      }
      attributes.set(node, record);
    }
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    const toggle = document.getElementById('language-toggle');
    if (toggle) {
      toggle.querySelector('span').textContent = language === 'en' ? '中文' : 'English';
      toggle.setAttribute('aria-label', language === 'en' ? '切换到中文' : 'Switch to English');
    }
    observer.observe(document.documentElement, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['aria-label', 'placeholder', 'alt', 'title']});
  }
  const observer = new MutationObserver(apply);
  function set(value, persist = true) {
    if (!['zh', 'en'].includes(value)) throw new Error('Unknown viewer language');
    const changed = language !== value;
    language = value;
    localStorage.setItem(storageKey, value);
    if (persist && window.cookieStore) cookieStore.set({name: storageKey, value, path: '/', sameSite: 'strict', expires: Date.now() + 365 * 86400000});
    apply();
    if (changed) window.dispatchEvent(new CustomEvent('viewer-language-change', {detail: value}));
  }
  document.getElementById('language-toggle')?.addEventListener('click', () => set(language === 'en' ? 'zh' : 'en'));
  window.addEventListener('storage', event => {if (event.key === storageKey && event.newValue !== language) set(event.newValue, false);});
  if (window.cookieStore) {
    const syncCookie = async () => {const saved = await cookieStore.get(storageKey); if (saved && saved.value !== language) set(saved.value, false);};
    cookieStore.addEventListener('change', syncCookie);
    window.addEventListener('focus', syncCookie);
    syncCookie();
  }
  apply();
  return {
    get value() {return language;}, get locale() {return language === 'en' ? 'en-US' : 'zh-CN';},
    set, apply, text: english,
    add(entries) {for (const [source, target] of entries) dictionary.set(source, target); apply();},
    register(translate) {translators.push(translate); apply();},
  };
})();
