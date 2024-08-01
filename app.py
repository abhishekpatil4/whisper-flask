from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from openai import OpenAI
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()
import json


app = Flask(__name__)
CORS(app)

def transcribe(audio_file, prompt):
    with open("./uploads/recording.wav", "rb") as audio:
        translation = client.audio.translations.create(
          model="whisper-1",
          file=audio,
          prompt=prompt
        )
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
    system_prompt = """You are a helpful assistant in a hotel, you will be given some content which describes a complaint, room number, area/location, priority. Analyse the problem and return a json object with the following attributes and follow the format strictly (case sentitive): 
    {
        complaint: '',
        roomNumber: '',
        area: '',
        department: '',
        description: '',
        priority: 'Low',
    },
    attribute description: 
    complaint: Describe the complaint in a short sentence, eg: AC is not working, Hot water unavailable
    roomNumber: See if the room number is mentioned, or else send 0
    area: see if you can identify the area, eg: water is leaking near the balcony, then balcony is the answer. if no specific area is mentioned or you cannot identify then set the value to na
    department: categorise the probelm into one of the following category and set the value (case sentitive) -> plubming, electrical, hvac, housekeeping, general
    description: describe the problem in 1-2 lines. 
    priority: its either "low", "medium" or "high", by default its "low" if the user specifies something then set that value (case sensitive)

    Note: give me the response strictly in a json format in a single line and nothing else.
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