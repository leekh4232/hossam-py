document.addEventListener('DOMContentLoaded', function () {
  if (window.mermaid) {
    try {
      window.mermaid.initialize({ startOnLoad: true });
      // Manually trigger in case of dynamic content
      window.mermaid.init();
    } catch (e) {
      console.warn('Mermaid init failed:', e);
    }
  }
});
