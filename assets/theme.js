// Light/dark layout of the project site. The layout is dark by default; the link at the bottom of the page
// switches it (data-theme on <html>, tokens in assets/site.css) and remembers the choice in
// localStorage, and fires a `themechange` event on document (the playground redraws its plots on it). It runs
// in <head> so that the layout is set before the page paints.
(() => {
  const root = document.documentElement;
  const KEY = 'site-theme';
  const read = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const saved = read();
  root.dataset.theme = saved === 'light' ? 'light' : 'dark';      // dark unless the visitor chose light

  const isDark = () => root.dataset.theme === 'dark';

  document.addEventListener('DOMContentLoaded', () => {
    const footer = document.querySelector('footer');
    if (!footer) return;
    const button = Object.assign(document.createElement('button'), { type: 'button', className: 'theme-toggle' });
    const label = () => { button.textContent = isDark() ? 'Light layout' : 'Dark layout'; };
    button.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      root.dataset.theme = next;
      try { localStorage.setItem(KEY, next); } catch (e) { /* the choice just is not remembered */ }
      label();
      document.dispatchEvent(new Event('themechange'));
    });
    label();
    footer.append(button);
  });
})();
