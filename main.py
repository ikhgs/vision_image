from flask import Flask, jsonify, request
from clarifai.client.model import Model
from clarifai.client.input import Inputs

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_model_prediction():
    # Récupérer la question et l'URL de l'image à partir des paramètres de la requête
    prompt = request.args.get('question', default="What time of day is it?", type=str)
    image_url = request.args.get('image_url', default="https://www.gstatic.com/webp/gallery/1.jpg", type=str)
    inference_params = dict(temperature=0.2, max_tokens=100)

    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(
        inputs=[Inputs.get_multimodal_input(input_id="", image_url=image_url, raw_text=prompt)],
        inference_params=inference_params
    )

    result = model_prediction.outputs[0].data.text.raw
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
