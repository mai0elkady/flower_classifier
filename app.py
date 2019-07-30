from flask import Flask, request, render_template, send_from_directory
app = Flask(__name__)
from werkzeug.utils import secure_filename
from commons import get_tensor
from inference import predict
import numpy as np
import os

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def flower_classify():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        filename = secure_filename(file.filename)
        pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pic_path)
        file.seek(0)
        image = file.read()
        #category, flower_name = get_flower_name(image_bytes=image)
        tensor = get_tensor(image_bytes=image)
        top_flower_name, flowers, probs  = predict(tensor)
        rounded_probs = list(np.around(np.array(probs),2))
        return render_template('bar_chart.html', pic_path = pic_path, title=top_flower_name, max=1,labels=flowers, values=rounded_probs)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT',5000))