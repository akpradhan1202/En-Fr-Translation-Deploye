from flask import Flask, request, jsonify, render_template
from load import * 

global encode_model,decode_model

encode_model,decode_model = init()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    en_sent = str(request.form.get("eng_sent"))
    encode_data = convert_sent_encode(en_sent.strip().lower())
    encode = encode_data.reshape(1,56,27)
    output = decode_sequence(encode,encode_model,decode_model)
        

    return render_template('index.html', fr_text = output,en_text=en_sent)


if __name__ == "__main__":
    app.run(debug=True)
