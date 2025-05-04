from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['COLOR_SCHEME'] = {
    'primary': '#1A73E8',
    'secondary': '#34A853',
    'accent': '#FBBC04',
    'background': '#F8F9FA'
}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_HEIGHT, IMG_WIDTH = 224, 224

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_pdf(report_id, diagnosis, confidence, original_img, heatmap_img):
    report_folder = "static/reports"
    os.makedirs(report_folder, exist_ok=True)
    pdf_filename = f"report_{report_id}.pdf"
    pdf_path = os.path.join(report_folder, pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 770, "Pneumonia Detection Report")

    # Report ID & Timestamp
    c.setFont("Helvetica", 10)
    c.drawString(100, 750, f"Report ID: {report_id}")
    c.drawString(100, 735, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Diagnosis & Confidence
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 700, f"Diagnosis: {diagnosis}")
    c.setFont("Helvetica", 12)
    c.drawString(100, 680, f"Confidence: {float(confidence.strip('%')):.2f}%")

    # Images
    try:
        original_img_reader = ImageReader(original_img)
        heatmap_img_reader = ImageReader(heatmap_img)
        c.drawImage(original_img_reader, 100, 450, width=200, height=200)
        c.drawImage(heatmap_img_reader, 350, 450, width=200, height=200)
        c.setFont("Helvetica", 10)
        c.drawString(130, 440, "Original X-ray with ROI")
        c.drawString(380, 440, "Grad-CAM Heatmap")
    except Exception as e:
        print(f"Error adding images: {e}")

    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(100, 100, "Generated using AI-powered Pneumonia Detection System")
    c.save()
    return pdf_filename

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_pneumonia(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0][0]

def process_image(filepath):
    # Preprocess image
    img_array = preprocess_image(filepath)
    
    # Get prediction
    prediction = predict_pneumonia(img_array)
    diagnosis = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = f"{prediction if diagnosis == 'Pneumonia' else 1 - prediction:.2%}"

    # Generate visualizations (simplified for TFLite)
    original_roi_path = filepath  # Using original image for demo
    heatmap_roi_path = filepath   # Replace with actual heatmap logic if needed

    return diagnosis, confidence, original_roi_path, heatmap_roi_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            diagnosis, confidence, original_roi, heatmap_roi = process_image(filepath)
            report_id = filename.split('.')[0] 
            
            original_url = url_for('static', filename=f'uploads/{os.path.basename(original_roi)}')
            heatmap_url = url_for('static', filename=f'uploads/{os.path.basename(heatmap_roi)}')
            og = url_for('static', filename=f'uploads/{os.path.basename(filepath)}')
            
            pdf_path = generate_pdf(
                report_id=filename.split('.')[0],
                diagnosis=diagnosis,
                confidence=confidence,
                original_img=filepath,
                heatmap_img=filepath
            )
            
            return render_template('result.html',
                diagnosis=diagnosis,
                confidence=confidence,
                original_img=original_url,
                heatmap_img=heatmap_url,
                og_img=og,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_id=report_id,
                pdf_url=pdf_path)
    
    return render_template('upload.html')

@app.route('/download_report/<report_id>')
def download_report(report_id):
    pdf_path = f"static/reports/report_{report_id}.pdf"
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return "Report not found", 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
