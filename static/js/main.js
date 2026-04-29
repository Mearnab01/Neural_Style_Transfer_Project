/* ─────────────────────────────────────────────────────────────────────────────
   StyleForge — main.js
   Handles: image preview, slider, form submit loader, accordion, mobile nav
───────────────────────────────────────────────────────────────────────────── */

'use strict';

/* ── Image Preview ───────────────────────────────────────────────────────── */
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

/* ── Range Slider ────────────────────────────────────────────────────────── */
function initSlider() {
    const slider = document.getElementById('alphaRange');
    const display = document.getElementById('alphaDisplay');
    if (!slider || !display) return;

    function update() {
        const val = parseFloat(slider.value).toFixed(1);
        display.textContent = val;

        // Fill track up to thumb position
        const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
        slider.style.background = `linear-gradient(to right,
            var(--accent) 0%,
            var(--accent) ${pct}%,
            var(--surface-2) ${pct}%,
            var(--surface-2) 100%)`;
    }

    slider.addEventListener('input', update);
    update(); // initialise on page load
}

/* ── Form Submit — Loading Overlay ──────────────────────────────────────── */
function initFormLoader() {
    const form = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const submitBtn = document.getElementById('submitBtn');
    if (!form || !loader) return;

    form.addEventListener('submit', function (e) {
        // Basic validation: both files must be chosen
        const contentInput = form.querySelector('input[type="file"][name="content"]');
        const styleInput   = form.querySelector('input[type="file"][name="style"]');

        const contentFilled = contentInput && (contentInput.files.length > 0 || document.getElementById('contentPreview').querySelector('img'));
        const styleFilled   = styleInput   && (styleInput.files.length   > 0 || document.getElementById('stylePreview').querySelector('img'));

        if (!contentFilled || !styleFilled) {
            e.preventDefault();
            showInlineError('Please select both a content image and a style image before submitting.');
            return;
        }

        loader.classList.add('active');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing…';
        }
    });
}

/* ── Inline validation error ─────────────────────────────────────────────── */
function showInlineError(msg) {
    // Remove existing inline error if present
    const existing = document.getElementById('sf-inline-error');
    if (existing) existing.remove();

    const alert = document.createElement('div');
    alert.id = 'sf-inline-error';
    alert.className = 'sf-alert sf-alert-error';
    alert.setAttribute('role', 'alert');
    alert.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i><span>${msg}</span>`;

    const controls = document.querySelector('.sf-controls-row');
    if (controls) controls.insertAdjacentElement('afterend', alert);

    // Auto-dismiss after 5 s
    setTimeout(() => { if (alert.parentNode) alert.remove(); }, 5000);
}

/* ── Accordion ───────────────────────────────────────────────────────────── */
function initAccordion() {
    const buttons = document.querySelectorAll('.sf-accordion-btn');
    buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            const targetId = btn.getAttribute('data-target');
            const body = document.getElementById(targetId);
            if (!body) return;

            const isOpen = body.classList.contains('open');

            // Close all
            document.querySelectorAll('.sf-accordion-body.open').forEach(function (b) {
                b.classList.remove('open');
            });
            document.querySelectorAll('.sf-accordion-btn.open').forEach(function (b) {
                b.classList.remove('open');
            });

            // Open clicked (if it was closed)
            if (!isOpen) {
                body.classList.add('open');
                btn.classList.add('open');
            }
        });
    });
}

/* ── Mobile Nav Toggle ───────────────────────────────────────────────────── */
function initMobileNav() {
    const toggle = document.getElementById('navToggle');
    const links  = document.getElementById('navLinks');
    if (!toggle || !links) return;

    toggle.addEventListener('click', function () {
        const expanded = links.classList.toggle('open');
        toggle.setAttribute('aria-expanded', expanded);
    });

    // Close nav when a link is clicked on mobile
    links.querySelectorAll('a').forEach(function (a) {
        a.addEventListener('click', function () {
            links.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        });
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
        if (!toggle.contains(e.target) && !links.contains(e.target)) {
            links.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });
}

/* ── Auto-scroll to result ───────────────────────────────────────────────── */
function initAutoScroll() {
    const result = document.getElementById('resultSection');
    if (result) {
        // Small delay so the page has painted
        setTimeout(function () {
            result.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    }
}

/* ── Boot ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
    initSlider();
    initFormLoader();
    initAccordion();
    initMobileNav();
    initAutoScroll();
});