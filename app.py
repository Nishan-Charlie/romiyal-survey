from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session
from pydantic import ValidationError
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import json
import logging
from ai_classifier_service import classify_learning_objective # Import the core logic
import os

# Load environment variables from a .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__, static_folder='.', static_url_path='') 
# A secret key is required for session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-default-fallback-secret-key-for-dev")
socketio = SocketIO(app)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# It's best practice to load secrets from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# In-memory data store to hold responses for the current session
user_responses = []
current_question = "What is your primary learning objective regarding the use of AI in medicine?"


# --- Static File Routes ---

@app.route('/')
def serve_login_page():
    """Serves the main login page."""
    return send_from_directory('.', 'login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    """Handles admin login credentials."""
    username = request.form.get('username')
    password = request.form.get('password')

    # Check credentials against environment variables
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        # Set a session variable to mark the user as logged in
        session['logged_in'] = True
        return redirect(url_for('serve_admin_panel'))
    else:
        # On failure, redirect back to the login page with an error flag
        return redirect(url_for('serve_login_page', error='true'))

@app.route('/survey')
def serve_user_panel():
    """Serves the user-facing survey panel."""
    return send_from_directory('.', 'user.html')

@app.route('/admin.html')
def serve_admin_panel():
    """Serves the admin panel only if the user is logged in."""
    if not session.get('logged_in'):
        # If not logged in, redirect to the login page
        return redirect(url_for('serve_login_page'))
    
    # If logged in, serve the admin panel
    return send_from_directory('.', 'admin.html')

@app.route('/logout')
def logout():
    """Logs the user out by clearing the session."""
    session.pop('logged_in', None)
    return redirect(url_for('serve_login_page'))

# --- WebSocket Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Send the current state to a newly connected client."""
    emit('state_update', {'question': current_question, 'responses': user_responses})

@socketio.on('set_question')
def handle_set_question(data):
    """Handles admin setting a new question and broadcasts it."""
    global current_question
    if session.get('logged_in'):
        question_text = data.get('question')
        if question_text:
            current_question = question_text
            # Broadcast the new question to all clients
            socketio.emit('state_update', {'question': current_question, 'responses': user_responses})

@socketio.on('clear_all_data')
def handle_clear_data():
    """Handles admin clearing all data and broadcasts the change."""
    global user_responses
    if session.get('logged_in'):
        user_responses = []
        # Broadcast the empty state to all clients
        socketio.emit('state_update', {'question': current_question, 'responses': []})


# --- API Endpoints ---

@app.route('/classify', methods=['POST'])
def classify_objective():
    """
    API endpoint to receive a user's objective and return the AI classification.
    
    Expected JSON input: {"answer": "My objective text here..."}
    """
    try:
        if not GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY environment variable not set.")
            return jsonify({"error": "Server is not configured for AI classification."}), 500

        # 1. Get and validate input data
        data = request.get_json()
        if not data or 'answer' not in data:
            logging.warning("Missing 'answer' field in request.")
            return jsonify({"error": "Missing required field 'answer'"}), 400

        objective = data['answer']
        logging.info(f"Received classification request for: {objective[:40]}...")

        # Gather unique, existing categories from previous responses
        existing_categories = sorted(list(set(
            res['classification']['primaryDomain'] for res in user_responses
        )))

        # 2. Call the core classification logic (from ai_classifier_service.py)
        classification_result = classify_learning_objective(objective=objective, api_url=GEMINI_API_URL, api_key=GEMINI_API_KEY, question=current_question, existing_categories=existing_categories)

        # Add the new response to our in-memory list
        new_response = {
            'id': f'res-{len(user_responses) + 1}',
            'userId': session.get('user_id', 'anonymous'), # A simple user identifier
            'answer': objective,
            'classification': classification_result.model_dump(),
            'timestamp': {'seconds': int(json.loads(classification_result.model_dump_json())['timestamp'])} if 'timestamp' in classification_result.model_dump() else int(__import__('time').time())
        }
        user_responses.append(new_response)
        socketio.emit('state_update', {'question': current_question, 'responses': user_responses})

        # 3. Return the Pydantic model as a JSON response
        # model_dump() converts the Pydantic object into a dictionary
        return jsonify(classification_result.model_dump()), 200

    except ValidationError as e:
        # Handles errors if the input format or the model output validation fails
        logging.error(f"Validation error: {e}")
        return jsonify({"error": "Data validation failed", "details": e.errors()}), 422
        
    except RuntimeError as e:
        # Handles internal errors, like the AI output not matching the schema
        logging.error(f"Runtime error in classification: {e}")
        return jsonify({"error": "AI classification failed due to internal schema mismatch."}), 500

    except Exception as e:
        # Catch all other unexpected errors
        logging.error(f"Unexpected error processing request: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # To run this file locally, you would execute: python app.py
    # Use socketio.run() to start the server with WebSocket support
    socketio.run(app, debug=True)
