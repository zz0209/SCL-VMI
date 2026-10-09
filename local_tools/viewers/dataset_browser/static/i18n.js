"use strict";
let language = ViewerLanguage.value;
const tr = (zh, en) => language === "en" ? en : zh;
function translateStatic() {
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  document.title = "SCL-VMI · " + tr("数据浏览器", "Dataset browser");
  document.querySelectorAll("[data-en]").forEach(element => {
    element.dataset.zh ??= element.textContent;
    element.textContent = tr(element.dataset.zh, element.dataset.en);
  });
  for (const [attribute, key] of [["aria-label", "Aria"], ["placeholder", "Placeholder"]]) {
    document.querySelectorAll(`[data-en-${key.toLowerCase()}]`).forEach(element => {
      element.dataset[`zh${key}`] ??= element.getAttribute(attribute);
      element.setAttribute(attribute, tr(element.dataset[`zh${key}`], element.dataset[`en${key}`]));
    });
  }
}
translateStatic();
