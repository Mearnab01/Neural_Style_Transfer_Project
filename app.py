import os
import logging
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY        = os.environ.get('SECRET_KEY', 'change-me-in-production'),
    UPLOAD_FOLDER     = os.path.join('static', 'uploads'),
    MAX_CONTENT_LENGTH= 16 * 1024 * 1024,   # 16 MB hard cap
    ALLOWED_EXTENSIONS= {'png', 'jpg', 'jpeg', 'webp'},
    DECODER_PATH      = os.environ.get('DECODER_PATH', 'weights/checkpoint_decoder.pth'),
    ENCODER_PATH      = os.environ.get('ENCODER_PATH', 'weights/vgg_normalised.pth'),
) 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Form ──────────────────────────────────────────────────────────────────────
class UploadForm(FlaskForm):
    content      = FileField('Content Image')
    style        = FileField('Style Image')
    content_path = HiddenField()
    style_path   = HiddenField()
    alpha        = FloatField('Style Strength', default=1.0)
    submit       = SubmitField('Transfer Style')

# ── Model loading ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info('Using device: %s', device)

def load_models():
    """Load encoder and decoder once at startup."""
    enc = VGGEncoder(app.config['ENCODER_PATH']).to(device)
    enc.eval()

    dec = Decoder().to(device)
    dec.load_state_dict(
        torch.load(app.config['DECODER_PATH'], map_location=device)
    )
    dec.eval()
    log.info('Models loaded successfully.')
    return enc, dec

encoder, decoder = load_models()

# Shared image transform — built once, reused for every request
_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def save_upload(file_storage) -> str | None:
    """Validate, save, and return the secure filename, or None on failure."""
    if not file_storage or not file_storage.filename:
        return None
    if not allowed_file(file_storage.filename):
        return None
    filename = secure_filename(file_storage.filename)
    file_storage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename


def run_style_transfer(content_path: str, style_path: str, alpha: float) -> str:
    """
    Run AdaIN style transfer and return the saved result filename.
    Raises on any error so the caller can surface it to the user.
    """
    alpha = max(0.0, min(1.0, alpha))   # clamp — never trust form input

    content_img = Image.open(content_path).convert('RGB')
    style_img   = Image.open(style_path).convert('RGB')

    c_tensor = _transform(content_img).unsqueeze(0).to(device)
    s_tensor = _transform(style_img).unsqueeze(0).to(device)

    with torch.inference_mode():        # faster + safer than no_grad
        c_feats = encoder(c_tensor, is_test=True)
        s_feats = encoder(s_tensor, is_test=True)

        stylised = adaptive_instance_normalization(c_feats, s_feats)
        stylised  = alpha * stylised + (1.0 - alpha) * c_feats
        output    = decoder(stylised)

    # Build a result name that encodes both inputs so concurrent
    # requests with same content name don't overwrite each other.
    content_stem = os.path.splitext(os.path.basename(content_path))[0]
    style_stem   = os.path.splitext(os.path.basename(style_path))[0]
    result_name  = f'stylised_{content_stem}_x_{style_stem}.jpg'
    result_path  = os.path.join(app.config['UPLOAD_FOLDER'], result_name)

    # Tensor → PIL → disk
    pil = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1).cpu())
    pil.save(result_path, format='JPEG', quality=92)

    log.info('Result saved: %s', result_path)
    return result_name

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    form         = UploadForm()
    result_image = None
    error        = None

    # Resolve filenames: new upload takes priority, then the hidden-field fallback
    content_filename = save_upload(form.content.data) or form.content_path.data or None
    style_filename   = save_upload(form.style.data)   or form.style_path.data   or None

    if form.is_submitted() and form.validate():
        if not content_filename:
            error = 'Please select a content image.'
        elif not style_filename:
            error = 'Please select a style image.'
        else:
            try:
                result_image = run_style_transfer(
                    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename),
                    style_path   = os.path.join(app.config['UPLOAD_FOLDER'], style_filename),
                    alpha        = float(form.alpha.data or 1.0),
                )
            except FileNotFoundError as exc:
                error = 'Uploaded file could not be found. Please re-upload.'
                log.error('Missing file: %s', exc)
            except Exception as exc:
                error = 'Style transfer failed. Please try again.'
                log.exception('Unexpected error during style transfer: %s', exc)

    return render_template(
        'index.html',
        form          = form,
        result_image  = result_image,
        content_image = content_filename,
        style_image   = style_filename,
        error         = error,
    )


@app.route('/uploads/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


@app.route('/health')
def health():
    """Lightweight liveness check — useful behind a reverse proxy."""
    return {'status': 'ok', 'device': str(device)}, 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)