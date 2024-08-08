from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

app = Flask(__name__)
CORS(app)

def transcribe(audio_file, prompt):
    with open("./uploads/recording.wav", "rb") as audio:
        translation = client.audio.translations.create(
          model="whisper-1",
          file=audio,
          prompt=prompt
        )
    print(translation.text)
    return translation.text

def generate_corrected_transcript(temperature, system_prompt, audio_file):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcribe(audio_file, "")
            }
        ]
    )
    return response.choices[0].message.content
    
def whisper():
    system_prompt = """You are a helpful assistant in a hotel, you will be given some content which describes a fault type, device name, device code and description. Analyze the problem and return a JSON object with the following attributes and follow the format strictly (case sensitive):
    {
        faultType: '',
        deviceName: '',
        deviceCode: '',
        description: '',
        priority: 'Low',
    },
        attribute description:
        faultType: extract the type of fault if mentioned, else set the value to na
        deviceName: extract the device name if mentioned, else set the value to na
        deviceCode: extract the device code if mentioned, else set the value to na
        description: describe the problem in 1-2 lines. 

        Note: give me the response strictly in a JSON format in a single line and nothing else.
    """
    corrected_text = generate_corrected_transcript(0, system_prompt, "./uploads/recording.wav")
    # print(corrected_text)
    return corrected_text

@app.route('/sendaudio', methods=['POST'])
def send_audio():
    if 'audio' not in request.files:
        return 'No audio file part', 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return 'No selected file', 400
    
    # Save the file to the desired location
    audio_file.save(os.path.join('uploads', audio_file.filename))
    res = json.loads(whisper())
    return res, 200

if __name__ == '__main__':
    app.run(debug=True)