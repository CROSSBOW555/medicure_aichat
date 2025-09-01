import os
import json
from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime
from pymongo import MongoClient

# --- Configuration ---
# Get secrets from Render's environment variables
API_KEY = os.environ.get("GEMINI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# --- Database Setup ---
if not MONGO_URI:
    print("FATAL ERROR: MONGO_URI is not set in the environment.")
    # Exit or handle gracefully if the URI is not found
    # For simplicity, we'll let it fail on connection attempt
    
client = MongoClient(MONGO_URI)
db = client.ai_companion_db # Your database name
appointments_collection = db.appointments # Your collection (table) name

# Initialize the Flask application
app = Flask(__name__, template_folder='.')

# --- Gemini API Helper Function ---
# (This function remains unchanged from the previous version)
def make_gemini_api_call(payload):
    """Makes a request to the Gemini API and returns the JSON response."""
    if not API_KEY:
        print("ERROR: Gemini API key is not set in the environment.")
        raise ValueError("Gemini API key is not configured on the server.")
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    print("\n--- Sending Payload to Gemini ---")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        
        print("\n--- Received Response from Gemini ---")
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)
        
        response.raise_for_status()

        result = response.json()
        
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(ai_response_text)
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise ConnectionError(f"API Error: Received status code {response.status_code}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        raise ConnectionError(f"Network error while contacting Gemini API.")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to parse Gemini response. Error: {e}")
        raise ValueError(f"Could not parse valid JSON from Gemini response.")

# --- AI Logic Functions ---
# (call_data_cleaning_ai and call_triage_ai remain unchanged)
def call_data_cleaning_ai(raw_data):
    """Calls Gemini to clean and format raw user data."""
    system_prompt = """You are a data cleaning and formatting expert. Your task is to take raw, conversational user input and convert it into a structured, clean JSON object.
    - Format 'dob' and 'appointmentDate' into a strict 'dd-mm-yyyy' format. Assume the current year is 2025 if not specified.
    - Format 'appointmentTime' into a strict 'hh:mm' 24-hour format. Interpret terms like 'morning' as 10:00, 'afternoon' as 14:00, and 'evening' as 18:00.
    - Clean up other text fields by correcting typos and ensuring proper capitalization.
    Your response MUST be ONLY the JSON object."""
    
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": json.dumps(raw_data)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"}, "dob": {"type": "STRING"}, "gender": {"type": "STRING"},
                    "phone": {"type": "STRING"}, "email": {"type": "STRING"}, "address": {"type": "STRING"},
                    "symptoms": {"type": "STRING"}, "history": {"type": "STRING"}, "medications": {"type": "STRING"},
                    "allergies": {"type": "STRING"}, "conditions": {"type": "STRING"},
                    "appointmentDate": {"type": "STRING"}, "appointmentTime": {"type": "STRING"}
                }
            }
        }
    }
    return make_gemini_api_call(payload)

def call_triage_ai(cleaned_data):
    """Calls Gemini to perform medical triage based on cleaned data."""
    system_prompt = """You are an expert medical triage assistant for a hospital named 'MediCure'. Your task is to analyze a patient's data and determine the most appropriate hospital department. You must also generate helpful, non-prescriptive precautions.
    The available departments are: Cardiology, Orthopedics, Neurology, Dermatology, Gastroenterology, Pulmonology, Endocrinology, and General Physician.
    CRITICAL SAFETY INSTRUCTION: Your generated precautions MUST ALWAYS begin with a bolded disclaimer: "<b>Disclaimer: This is not medical advice. If your symptoms are severe or worsen, please visit the nearest emergency room immediately.</b>"
    Your response MUST be ONLY a valid JSON object."""
    
    triage_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": f"Patient Data: {json.dumps({'symptoms': cleaned_data.get('symptoms'), 'history': cleaned_data.get('history'), 'conditions': cleaned_data.get('conditions')})}"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "department": {"type": "STRING"},
                    "precautions": {"type": "STRING"}
                },
                "required": ["department", "precautions"]
            }
        }
    }
    return make_gemini_api_call(triage_payload)

# --- Routes ---
@app.route('/')
def index():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    """Receives user data, processes it with AI, and saves the result to MongoDB."""
    try:
        raw_data = request.json.get('userData')
        if not raw_data:
            return jsonify({"error": "No user data provided."}), 400

        cleaned_data = call_data_cleaning_ai(raw_data)
        triage_result = call_triage_ai(cleaned_data)

        final_appointment_data = {
            **cleaned_data,
            "referral": {
                "assignedDepartment": triage_result.get("department"),
                "aiPrecautions": triage_result.get("precautions"),
            },
            "bookingTimestampUTC": datetime.utcnow().isoformat()
        }

        # *** NEW: Save to MongoDB instead of a file ***
        insert_result = appointments_collection.insert_one(final_appointment_data)
        print(f"\n--- Success! Saved appointment to MongoDB with ID: {insert_result.inserted_id} ---")

        return jsonify(triage_result)

    except (ValueError, ConnectionError) as e:
        print(f"A known error occurred during processing: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

