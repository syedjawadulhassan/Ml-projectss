"""
Pneumonia Detection - Flask Web Application.
Loads model once at startup; handles secure upload, preprocessing, prediction, and Grad-CAM.
"""

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch

from utils import (
    load_pneumonia_model,
    preprocess_image,
    get_uploads_dir,
    create_gradcam_heatmap_image,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Ensure directories exist
uploads_dir = get_uploads_dir()
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'model'), exist_ok=True)


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = load_pneumonia_model(device=DEVICE)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', model_loaded=MODEL is not None)


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page: GET shows form, POST handles file upload and redirects to result."""
    if request.method == 'GET':
        return render_template('upload.html', model_loaded=MODEL is not None)

    if MODEL is None:
        flash('Model not loaded. Please train the model first (run train_model.py).', 'danger')
        return redirect(url_for('upload'))

    if 'file' not in request.files:
        flash('No file selected.', 'warning')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('upload'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed: PNG, JPG, JPEG, BMP, GIF.', 'danger')
        return redirect(url_for('upload'))

    filepath = None
    try:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(uploads_dir, unique_name)
        file.save(filepath)

        # Preprocess and predict
        input_tensor = preprocess_image(filepath)
        with torch.no_grad():
            input_tensor = input_tensor.to(DEVICE)
            logits = MODEL(input_tensor)
            probs = torch.softmax(logits, dim=1)
            prob_numpy = probs.cpu().numpy()[0]
        pred_idx = int(prob_numpy.argmax())
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(prob_numpy[pred_idx])

        # Grad-CAM
        gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
        gradcam_path = os.path.join(uploads_dir, gradcam_filename)
        input_for_cam = preprocess_image(filepath)
        create_gradcam_heatmap_image(MODEL, input_for_cam, filepath, gradcam_path, DEVICE)

        # Relative paths for templates (under static/uploads)
        upload_url = f'/static/uploads/{unique_name}'
        gradcam_url = f'/static/uploads/{gradcam_filename}'

        return redirect(url_for(
            'result',
            prediction=pred_label,
            confidence=round(confidence, 4),
            upload_url=upload_url,
            gradcam_url=gradcam_url,
        ))
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'danger')
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        return redirect(url_for('upload'))


@app.route('/result')
def result():
    """
    Result page. Expects prediction, confidence, upload_url, gradcam_url in query args.
    """
    prediction = request.args.get('prediction', 'NORMAL')
    confidence = request.args.get('confidence', '0')
    upload_url = request.args.get('upload_url', '')
    gradcam_url = request.args.get('gradcam_url', '')
    try:
        confidence = float(confidence)
    except ValueError:
        confidence = 0.0
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        upload_url=upload_url,
        gradcam_url=gradcam_url,
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 8 MB.', 'danger')
    return redirect(url_for('upload'))


@app.errorhandler(500)
def server_error(e):
    flash('An internal error occurred. Please try again.', 'danger')
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if MODEL is None:
        print('Warning: model/pneumonia_model.pth not found. Train with: python train_model.py')
    app.run(debug=True, host='0.0.0.0', port=5000)
