"""
Flask Backend Server for Prompt Anonymization
Uses the main.py PromptAnonymizer class
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys

# Import the PromptAnonymizer from main.py
try:
    from main import PromptAnonymizer, EntityType, AnonymizationMethod, AnonymizationRule
except ImportError:
    print("Error: Could not import from main.py")
    print("Make sure main.py is in the same directory as this file")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global anonymizer instance
anonymizer = None

def initialize_anonymizer():
    """Initialize the anonymizer with default configuration"""
    global anonymizer
    anonymizer = PromptAnonymizer()
    logger.info("PromptAnonymizer initialized")

# Initialize anonymizer on startup
initialize_anonymizer()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('main.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/anonymize', methods=['POST'])
def anonymize():
    """API endpoint to anonymize a prompt"""
    if not anonymizer:
        return jsonify({"error": "Anonymizer not initialized"}), 500

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data['prompt']
    config = data.get('config')

    try:
        if config:
            # Update anonymizer config rules if provided
            rules = config.get('rules')
            if rules:
                for entity_type_str, rule_data in rules.items():
                    try:
                        entity_type = EntityType[entity_type_str]
                        method = AnonymizationMethod[rule_data.get('method', 'SYNTHETIC')]
                        placeholder = rule_data.get('placeholder')
                        preserve_format = rule_data.get('preserve_format', False)
                        anonymizer.add_custom_rule(entity_type, AnonymizationRule(entity_type, method, placeholder, preserve_format))
                    except KeyError:
                        logger.warning(f"Invalid entity type or method in config: {entity_type_str}, {rule_data.get('method')}")
            confidence_threshold = config.get('confidence_threshold')
            if confidence_threshold is not None:
                anonymizer.config['confidence_threshold'] = confidence_threshold

        result = anonymizer.anonymize_prompt(prompt)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during anonymization: {e}")
        return jsonify({"error": "An error occurred during anonymization"}), 500

if __name__ == '__main__':
    app.run(debug=True)
