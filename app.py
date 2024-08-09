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
    system_prompt = """You are an intelligent hotel management assistant. You will receive content describing a problem, fault, or issue in the hotel. This content may include details like 'device name', 'device code'. Your task is to analyze this input and return a structured JSON object. Follow these guidelines strictly:
        1. Return only a single-line JSON object with no additional text.
        2. Use the following structure (case-sensitive):

        {
            faultType: '',
            deviceName: '',
            deviceCode: '',
            description: '',
            priority: 'Low',
        }

        3. Attribute details:
        - faultType: Based on the content provided identify the specific problem/issue/fault and mention it in a short sentence (e.g., "AC not working", "WiFi disconnecting", "Heater not working"). If no clear fault is mentioned, use "Unspecified unclear".
        - deviceName: Extract the device or item name. If multiple devices are mentioned, list the primary one. If not applicable, use "na".
        - deviceCode: Extract any alphanumeric code associated with the device. If not mentioned, use "na".
        - description: Based on the content provide a concise summary of the problem in 1-2 sentences. 
        - priority: Assess the urgency:
        * 'High': Critical issues affecting guest safety, comfort, or hotel operations
        * 'Medium': Moderate issues that need attention but aren't immediately critical
        * 'Low': Minor issues

        4. Handle ambiguities:
        - If the input is vague, make reasonable inferences based on context.
        - For conflicting information, prioritize the most recent or severe interpretation.
        - If critical information is missing, use 'unclear' values that indicate the need for more details.

        Parse the input carefully to extract all relevant information, even if not explicitly categorized in the original text.
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