'use strict';

/* ── Image preview ───────────────────────────────────────────────────────── */
function previewFile(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];

    if (!file) {
        preview.innerHTML = `
            <div class="dz-placeholder">
                <i class="fa-regular fa-file-image"></i>
                <span>No image selected</span>
            </div>`;
        updateClearBtn();
        return;
    }

    const reader = new FileReader();
    reader.addEventListener('load', function () {
        preview.innerHTML = `<img src="${reader.result}" alt="Preview">`;
        updateClearBtn();
    });
    reader.readAsDataURL(file);
}

/* ── Clear button state ──────────────────────────────────────────────────── */
function updateClearBtn() {
    const btn = document.getElementById('clearBtn');
    if (!btn) return;

    const hasContentImg = !!document.querySelector('#contentPreview img');
    const hasStyleImg   = !!document.querySelector('#stylePreview img');
    const hasResult     = !!document.querySelector('#resultSection');

    const shouldBeActive = hasContentImg || hasStyleImg || hasResult;
    btn.disabled = !shouldBeActive;
    btn.classList.toggle('active', shouldBeActive);
}

/* ── Clear button logic ──────────────────────────────────────────────────── */
function clearAll() {
    const pairs = [
        { inputId: 'contentInput', previewId: 'contentPreview' },
        { inputId: 'styleInput',   previewId: 'stylePreview'   },
    ];

    pairs.forEach(function ({ inputId, previewId }) {
        const input   = document.getElementById(inputId);
        const preview = document.getElementById(previewId);

        if (input)   input.value = '';
        if (preview) preview.innerHTML = `
            <div class="dz-placeholder">
                <i class="fa-regular fa-file-image"></i>
                <span>No image selected</span>
            </div>`;
    });

    const result = document.getElementById('resultSection');
    if (result) result.remove();

    updateClearBtn();
}

/* ── Inject + init Clear button ──────────────────────────────────────────── */
function initClearButton() {
    const btn = document.createElement('button');
    btn.id        = 'clearBtn';
    btn.type      = 'button';
    btn.className = 'clear-btn';
    btn.textContent = 'Clear';
    btn.disabled  = true;
    btn.setAttribute('aria-label', 'Clear all images and result');

    btn.addEventListener('click', clearAll);

    // Insert right after the submit button, falling back to end of form
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.insertAdjacentElement('afterend', btn);
    } else {
        const form = document.getElementById('uploadForm');
        if (form) form.appendChild(btn);
    }
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

            document.querySelectorAll('.faq-a.open').forEach(b => b.classList.remove('open'));
            document.querySelectorAll('.faq-q.open').forEach(b => b.classList.remove('open'));

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

    drawer.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => {
            drawer.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        });
    });

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
        updateClearBtn(); // result present on load (e.g. after form POST) → activate button
        setTimeout(() => result.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
    }
}
/* ── Download 5 random style samples directly ────────────────────────────── */
async function downloadSamples() {
    const btn  = document.getElementById('samplesBtn');
    const span = btn.querySelector('span');

    const TOTAL = 60;
    const PICK  = 5;

    const indices = Array.from({ length: TOTAL }, (_, i) => i + 1)
        .sort(() => Math.random() - 0.5)
        .slice(0, PICK);

    btn.disabled = true;
    span.textContent = 'Downloading\u2026';

    try {
        for (const n of indices) {
            const res  = await fetch(`https://raw.githubusercontent.com/Mearnab01/Neural_Style_Transfer_Project/main/styled_data/styled_${n}.jpg`);
            if (!res.ok) throw new Error(`Failed: styled_${n}.jpg`);
            const blob = await res.blob();

            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `styled_${n}.jpg`;
            a.click();
            URL.revokeObjectURL(a.href);

            // small gap so browser doesn't block rapid-fire downloads
            await new Promise(r => setTimeout(r, 300));
        }
    } catch (err) {
        console.error('Download failed:', err);
        showError('Could not download sample images. Check the styled_data path.');
    } finally {
        btn.disabled = false;
        span.textContent = 'Sample Styles';
    }
}

/* ── Boot ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
    initSlider();
    initForm();
    initClearButton();
    initFaq();
    initMobileNav();
    initAutoScroll();
});