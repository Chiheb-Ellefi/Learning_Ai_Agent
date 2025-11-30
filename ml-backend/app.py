# app.py - UPDATED VERSION

import json
import os
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types 

# Import your local ML models
from model.learning_style_classifier import LearningStyleClassifier
from model.quiz_performance_predictor import QuizPerformancePredictor
from model.smart_resource_ranker import SmartResourceRanker

import dotenv
dotenv.load_dotenv()

app = Flask(__name__)
CORS(app) 

# -----------------------------------------------------
# 1. INITIALIZE API CLIENTS AND ML MODELS
# -----------------------------------------------------

try:
    ai = genai.Client() 
    print("✅ Gemini Client Initialized.")
except Exception as e:
    print(f"❌ Gemini Client Error: {e}")

# --- ML Model Initialization ---
style_classifier = LearningStyleClassifier()
quiz_predictor = QuizPerformancePredictor()
resource_ranker = SmartResourceRanker()

# Load pre-trained models
try:
    style_classifier.load_model()
    quiz_predictor.load_model()
    print("✅ Local ML Models Loaded.")
except Exception as e:
    print(f"❌ Failed to load ML models: {e}")

# --- Session Tracking for Chat ---
chat_sessions = {} 

# -----------------------------------------------------
# 2. HELPER FUNCTIONS FOR ML
# -----------------------------------------------------

def extract_style_features(data):
    """
    MOCK FUNCTION: Replace with real 19-feature extraction logic.
    """
    return [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ] 

# -----------------------------------------------------
# 3. GEMINI PROXY ENDPOINTS
# -----------------------------------------------------
DEFAULT_MODEL = 'gemini-2.5-flash'
ADVANCED_MODEL = 'gemini-2.5-pro' 

@app.route('/generate-roadmap', methods=['POST'])
def generate_roadmap():
    """Proxy for Gemini Roadmap Generation using JSON Mode."""
    data = request.json
    model = data.get('model', DEFAULT_MODEL)
    prompt = data['prompt']
    schema_dict = data['schema']

    try:
        response = ai.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_dict,
            ),
        )

        parsed_json = json.loads(response.text)
        return jsonify(parsed_json)

    except Exception as e:
        print(f"Gemini Roadmap Error: {e}")
        return jsonify({'error': 'Failed to generate roadmap from Gemini API.', 'details': str(e)}), 500


@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    """Proxy for Gemini Quiz Generation using JSON Mode."""
    data = request.json
    model = data.get('model', DEFAULT_MODEL)
    prompt = data['prompt']
    schema_dict = data['schema']

    try:
        response = ai.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_dict,
            ),
        )

        parsed_json = json.loads(response.text)
        return jsonify(parsed_json)

    except Exception as e:
        print(f"Gemini Quiz Error: {e}")
        return jsonify({'error': 'Failed to generate quiz from Gemini API.', 'details': str(e)}), 500


@app.route('/generate-visual-roadmap', methods=['POST'])
def generate_visual_roadmap():
    """Generate Mermaid diagram code for visual roadmap (like roadmap.sh)"""
    data = request.json
    roadmap = data['roadmap']
    
    # Enhanced prompt to create roadmap.sh style diagrams
    prompt = f"""You are an expert at creating learning roadmaps similar to roadmap.sh.

Given this roadmap structure:
{json.dumps(roadmap, indent=2)}

Create a Mermaid flowchart diagram that:
1. Shows clear learning progression from fundamentals to advanced
2. Uses different node shapes for different types:
   - Round nodes (( )) for start/prerequisites
   - Rectangle nodes [ ] for main topics
   - Stadium nodes ([ ]) for milestones/checkpoints
3. Shows dependencies with arrows
4. Uses colors to indicate difficulty (add :::beginner, :::intermediate, :::advanced classes)
5. Groups related topics together when possible

Make it visually similar to roadmap.sh - clean, professional, and easy to follow.

Return ONLY valid Mermaid code starting with "flowchart TD" or "graph TD".
Include CSS styling at the end for the difficulty classes.

Example format:
flowchart TD
    Start((Start Here))
    Topic1[Main Topic 1]
    Topic2[Main Topic 2]
    Milestone1([Checkpoint])
    
    Start --> Topic1
    Topic1 --> Topic2
    Topic2 --> Milestone1
    
    classDef beginner fill:#90EE90
    classDef intermediate fill:#FFD700
    classDef advanced fill:#FF6347
    
    Topic1:::beginner
    Topic2:::intermediate
"""

    try:
        response = ai.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )

        mermaid_code = response.text.strip()
        
        # Clean up the response - remove markdown code blocks if present
        if mermaid_code.startswith('```mermaid'):
            mermaid_code = mermaid_code.replace('```mermaid', '').replace('```', '').strip()
        elif mermaid_code.startswith('```'):
            mermaid_code = mermaid_code.replace('```', '').strip()
        
        return jsonify({'mermaidCode': mermaid_code})

    except Exception as e:
        print(f"Visual Roadmap Error: {e}")
        return jsonify({'error': 'Failed to generate visual roadmap.', 'details': str(e)}), 500


chat_sessions = {}

@app.route('/start-chat', methods=['POST'])
def start_chat():
    """Start a new chat session."""
    data = request.json
    model = data.get('model', DEFAULT_MODEL)
    system_instruction = data['systemInstruction']

    session_id = str(time.time())
    
    chat_sessions[session_id] = {
        'model': model,
        'system_instruction': system_instruction,
        'history': []
    }

    return jsonify({'sessionId': session_id})


@app.route('/send-message', methods=['POST'])
def send_message():
    """Send a message in a chat session."""
    data = request.json
    session_id = data['sessionId']
    message = data['message']
    
    session = chat_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Chat session not found.'}), 404

    try:
        contents = []
        
        for msg in session['history']:
            contents.append(types.Content(
                role=msg['role'],
                parts=[types.Part(text=msg['parts'][0]['text'])]
            ))
        
        contents.append(types.Content(
            role='user',
            parts=[types.Part(text=message)]
        ))
        
        response = ai.models.generate_content(
            model=session['model'],
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=session['system_instruction'],
            ),
        )
        
        session['history'].append({'role': 'user', 'parts': [{'text': message}]})
        session['history'].append({'role': 'model', 'parts': [{'text': response.text}]})
        
        return jsonify({'text': response.text})

    except Exception as e:
        print(f"Gemini Message Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to get response from Gemini API.', 'details': str(e)}), 500

# -----------------------------------------------------
# 4. ML ENDPOINTS
# -----------------------------------------------------

@app.route('/api/analyze-learning-style', methods=['POST'])
def analyze_learning_style():
    """Predict user's learning style using the RandomForest Classifier."""
    data = request.json
    features = extract_style_features(data) 
    
    style_index, probabilities = style_classifier.predict(np.array([features]))
    style_names = ["Visual", "Auditory", "Reading", "Kinesthetic"]
    
    confidence_map = {
        style_names[i]: float(probabilities[i]) 
        for i in range(len(style_names))
    }
    
    return jsonify({
        'learning_style': style_names[style_index],
        'confidence': confidence_map
    })


@app.route('/api/predict-quiz-score', methods=['POST'])
def predict_quiz_score():
    """Predict user's quiz performance using the XGBoost Regressor."""
    data = request.json
    
    prediction = quiz_predictor.predict(data)
    
    return jsonify(prediction)


@app.route('/api/rank-resources', methods=['POST'])
def rank_resources():
    """
    Rank resources using the TF-IDF/Cosine Similarity Ranker.
    FIXED: Now always works, not just after learning style detection
    """
    data = request.json
    
    # 1. Add/re-vectorize resources from the roadmap
    if 'resources' in data and len(data['resources']) > 0:
        try:
            resource_ranker.add_resources(data['resources'])
            
            # 2. Rank them based on user context
            recommendations = resource_ranker.rank_for_topic(
                topic_query=data['topic'],
                learning_style=data.get('learning_style', 'Visual'),  # Default to Visual if not detected
                difficulty_level=data.get('difficulty_level', 0.5)
            )
            
            return jsonify({
                'recommendations': recommendations,
                'ranked': True
            })
        except Exception as e:
            print(f"Resource ranking error: {e}")
            # Fallback: return original resources if ranking fails
            return jsonify({
                'recommendations': data['resources'],
                'ranked': False,
                'error': str(e)
            })
    else:
        return jsonify({
            'recommendations': [],
            'ranked': False,
            'error': 'No resources provided'
        })


# -----------------------------------------------------
# 5. SERVER RUN
# -----------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=5000)