/* ─────────────────────────────────────────────────────────────────────────────
   StyleForge v2 — main.js
───────────────────────────────────────────────────────────────────────────── */

'use strict';

/* ── Image preview ───────────────────────────────────────────────────────── */
function previewFile(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.addEventListener('load', function () {
        preview.innerHTML = `<img src="${reader.result}" alt="Preview">`;
    });
    reader.readAsDataURL(file);
}

/* ── Range slider — filled track + badge ─────────────────────────────────── */
function initSlider() {
    const slider  = document.getElementById('alphaRange');
    const display = document.getElementById('alphaDisplay');
    if (!slider || !display) return;

    function update() {
        const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
        slider.style.background = `linear-gradient(to right,
            var(--accent) 0%, var(--accent) ${pct}%,
            var(--border) ${pct}%, var(--border) 100%)`;
        display.textContent = parseFloat(slider.value).toFixed(1);
    }

    slider.addEventListener('input', update);
    update();
}

/* ── Form validation + loader ────────────────────────────────────────────── */
function initForm() {
    const form      = document.getElementById('uploadForm');
    const loader    = document.getElementById('loader');
    const submitBtn = document.getElementById('submitBtn');
    if (!form || !loader) return;

    form.addEventListener('submit', function (e) {
        const contentInput  = document.getElementById('contentInput');
        const styleInput    = document.getElementById('styleInput');
        const contentFilled = (contentInput && contentInput.files.length > 0)
                           || document.getElementById('contentPreview').querySelector('img');
        const styleFilled   = (styleInput   && styleInput.files.length   > 0)
                           || document.getElementById('stylePreview').querySelector('img');

        if (!contentFilled || !styleFilled) {
            e.preventDefault();
            showError(!contentFilled
                ? 'Please select a content image.'
                : 'Please select a style image.');
            return;
        }

        // Show loader
        loader.classList.add('active');
        loader.setAttribute('aria-hidden', 'false');

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing\u2026';
        }
    });
}

/* ── Inline error banner ─────────────────────────────────────────────────── */
function showError(msg) {
    const existing = document.getElementById('js-error');
    if (existing) existing.remove();

    const el = document.createElement('div');
    el.id = 'js-error';
    el.className = 'field-error';
    el.setAttribute('role', 'alert');
    el.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i><span>${msg}</span>`;

    const btn = document.querySelector('.transfer-btn');
    if (btn) btn.insertAdjacentElement('afterend', el);

    setTimeout(() => { if (el.parentNode) el.remove(); }, 5000);
}

/* ── FAQ accordion ───────────────────────────────────────────────────────── */
function initFaq() {
    document.querySelectorAll('.faq-q').forEach(function (btn) {
        btn.addEventListener('click', function () {
            const targetId = btn.getAttribute('data-target');
            const body     = document.getElementById(targetId);
            if (!body) return;

            const isOpen = body.classList.contains('open');

            // Close all
            document.querySelectorAll('.faq-a.open').forEach(b => b.classList.remove('open'));
            document.querySelectorAll('.faq-q.open').forEach(b => b.classList.remove('open'));

            // Toggle clicked
            if (!isOpen) {
                body.classList.add('open');
                btn.classList.add('open');
            }
        });
    });
}

/* ── Mobile nav ──────────────────────────────────────────────────────────── */
function initMobileNav() {
    const toggle = document.getElementById('navToggle');
    const drawer = document.getElementById('mobileDrawer');
    if (!toggle || !drawer) return;

    toggle.addEventListener('click', function () {
        const open = drawer.classList.toggle('open');
        toggle.setAttribute('aria-expanded', open);
    });

    // Close on link tap
    drawer.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => {
            drawer.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        });
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
        if (!toggle.contains(e.target) && !drawer.contains(e.target)) {
            drawer.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });
}

/* ── Auto-scroll to result ───────────────────────────────────────────────── */
function initAutoScroll() {
    const result = document.getElementById('resultSection');
    if (result) {
        setTimeout(() => result.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
    }
}

/* ── Boot ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
    initSlider();
    initForm();
    initFaq();
    initMobileNav();
    initAutoScroll();
});