from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import numpy as np
import torch
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import time
import requests
import markdown2
from PIL import Image
import threading
import logging
import queue
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("mediassist")

# Initialize Flask app immediately
app = Flask(__name__, static_folder='static')

# Global variables and state tracking
APP_STATE = {
    "is_initialized": False,
    "initialization_error": None,
    "loading_progress": {
        "data": {"status": "pending", "message": "Not started"},
        "clinical_embeddings": {"status": "pending", "message": "Not started"},
        "literature_embeddings": {"status": "pending", "message": "Not started"},
        "symptom_embeddings": {"status": "pending", "message": "Not started"},
        "drug_embeddings": {"status": "pending", "message": "Not started"}
    }
}
data = {}
embeddings = {}
embedding_lock = threading.RLock()  # For thread-safe embeddings access
data_lock = threading.RLock()  # For thread-safe data access

# Environment variables
GEMINI_API_KEY =  "AIzaSyCcRxsIv0GyiRl3NtPvr1o8LdfoeDUn_HE"
HF_API_KEY = "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"
MODEL_NAME = "gemini-1.5-flash"
MODEL = None

# Initialize Gemini API
def init_gemini_api():
    global MODEL
    try:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY environment variable not set")
            return False
            
        genai.configure(api_key=GEMINI_API_KEY)
        MODEL = genai.GenerativeModel(MODEL_NAME)
        logger.info("Gemini API initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Gemini API: {str(e)}")
        return False

# Decorator for lazy loading
def ensure_data_loaded(category=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global data
            
            # Check if data is loaded
            with data_lock:
                if not data:
                    try:
                        logger.info("Lazy loading data on first request")
                        data = load_excel_data()
                        if not data:
                            return jsonify({
                                "status": "error",
                                "response": "Failed to load required data. Please try again later."
                            })
                    except Exception as e:
                        logger.error(f"Error in lazy loading data: {str(e)}")
                        return jsonify({
                            "status": "error",
                            "response": f"Error loading data: {str(e)}"
                        })
                
                # If a specific category is requested, ensure it's loaded
                if category and category not in data:
                    return jsonify({
                        "status": "error",
                        "response": f"Data category '{category}' not available"
                    })
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Format markdown response
def format_markdown_response(text):
    """Format the response with proper markdown and styling"""
    try:
        # Convert markdown to HTML with extras
        html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
        
        # Enhance emoji display
        emoji_map = {
            '🏥': '<span class="emoji hospital">🏥</span>',
            '💊': '<span class="emoji medication">💊</span>',
            '⚠️': '<span class="emoji warning">⚠️</span>',
            '📊': '<span class="emoji stats">📊</span>',
            '📋': '<span class="emoji clipboard">📋</span>',
            '👨‍⚕️': '<span class="emoji doctor">👨‍⚕️</span>',
            '🔬': '<span class="emoji research">🔬</span>',
            '📚': '<span class="emoji book">📚</span>',
            '🔍': '<span class="emoji search">🔍</span>',
            '🚨': '<span class="emoji alert">🚨</span>',
            '👁️': '<span class="emoji eye">👁️</span>',
            '🔄': '<span class="emoji repeat">🔄</span>',
            '🔮': '<span class="emoji crystal-ball">🔮</span>'
        }
        
        for emoji, styled_emoji in emoji_map.items():
            html = html.replace(emoji, styled_emoji)
        
        return html
    except Exception as e:
        logger.error(f"Error formatting markdown: {str(e)}")
        return f"<p>Error formatting response: {str(e)}</p><pre>{text}</pre>"

# Image analysis function with caching and optimized processing
def analyze_image(image_data, prompt, cache_results=True):
    """Analyzes the image using Gemini and returns the response with optional caching."""
    global MODEL
    
    # Generate a unique key for potential caching
    cache_key = None
    if cache_results:
        import hashlib
        img_hash = hashlib.md5(image_data).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_key = f"{img_hash}_{prompt_hash}"
        
        # Check if result is cached
        cache_dir = 'cache/image_analysis'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                logger.info(f"Using cached image analysis result for {cache_key}")
                return cached_result.get('text', 'Error retrieving cached result')
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
    
    try:
        if not GEMINI_API_KEY:
            return "API key is not configured. Please check environment variables."
        
        if MODEL is None:
            init_gemini_api()
            if MODEL is None:
                return "Failed to initialize Gemini model. Please try again later."
        
        # Enhanced medical prompt
        enhanced_prompt = f"""As a medical image analysis expert, analyze this medical image with precision and clinical relevance.

Medical Context: {prompt}

Please provide your analysis in a clear, structured format with the following sections:
1. Image Description - Describe what you see in the image
2. Key Findings - Identify notable features or abnormalities
3. Possible Interpretations - Discuss what these findings might indicate
4. Recommendations - Suggest next steps or further tests if applicable

Format your response with clear markdown headings and bullet points for readability."""

        # Process the image
        image = Image.open(io.BytesIO(image_data))
        
        # Configure the model for optimal medical image analysis
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more accurate analysis
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Use a dedicated model instance for image analysis
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response = model.generate_content([enhanced_prompt, image])
        result_text = response.text
        
        # Cache the result if enabled
        if cache_key:
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'text': result_text}, f)
                logger.info(f"Cached image analysis result for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to cache: {str(e)}")
        
        return result_text
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return f"Error analyzing image: {e}"

# Load data from Excel files with performance optimizations
def load_excel_data():
    """Load all medical data from Excel files with optimized performance"""
    result_data = {}
    try:
        APP_STATE["loading_progress"]["data"] = {"status": "in_progress", "message": "Loading data files"}
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Function to load a specific category
        def load_category(category, filename):
            try:
                df = pd.read_excel(f'data/{filename}')
                result_data[category] = df.to_dict(orient='records')
                logger.info(f"Loaded {len(result_data[category])} {category} records")
                return True
            except Exception as e:
                logger.error(f"Error loading {category} data: {str(e)}")
                return False
        
        # Load each category
        categories = [
            ('clinical', 'clinical_cases.xlsx'),
            ('literature', 'medical_literature.xlsx'),
            ('symptom', 'symptom_cases.xlsx'),
            ('drug', 'drug_interactions.xlsx')
        ]
        
        success = True
        for category, filename in categories:
            if not load_category(category, filename):
                success = False
        
        if success:
            APP_STATE["loading_progress"]["data"] = {"status": "complete", "message": f"Loaded {sum(len(v) for v in result_data.values())} total records"}
            logger.info(f"Data loaded successfully: {sum(len(v) for v in result_data.values())} total records")
        else:
            APP_STATE["loading_progress"]["data"] = {"status": "error", "message": "Failed to load some data files"}
            logger.warning("Some data categories failed to load")
        
        return result_data
    
    except Exception as e:
        APP_STATE["loading_progress"]["data"] = {"status": "error", "message": str(e)}
        logger.error(f"Error loading data: {str(e)}")
        return None

# Generate text for embedding with optimizations
def prepare_text_for_embedding(record, category):
    """Prepare a record for embedding by converting it to a comprehensive text string"""
    try:
        if category == 'clinical':
            return f"Case ID: {record.get('case_id', '')}. Patient: {record.get('age', '')} year old {record.get('gender', '')}. Symptoms: {record.get('symptoms', '')}. Medical history: {record.get('medical_history', '')}. Diagnosis: {record.get('diagnosis', '')}. Treatment: {record.get('treatment', '')}. Outcome: {record.get('outcome', '')}. Complications: {record.get('complications', '')}."
        
        elif category == 'literature':
            return f"Paper ID: {record.get('paper_id', '')}. Title: {record.get('title', '')}. Authors: {record.get('authors', '')}. Published: {record.get('publication_date', '')} in {record.get('journal', '')}. Key findings: {record.get('key_findings', '')}. Methodology: {record.get('methodology', '')}. Sample size: {record.get('sample_size', '')}."
        
        elif category == 'symptom':
            return f"Symptom ID: {record.get('symptom_id', '')}. Presenting symptoms: {record.get('presenting_symptoms', '')}. Diagnosis: {record.get('diagnosis', '')}. Risk factors: {record.get('risk_factors', '')}. Specialists: {record.get('recommended_specialists', '')}. Urgency: {record.get('urgency_level', '')}. Tests: {record.get('diagnostic_tests', '')}."
        
        elif category == 'drug':
            return f"Interaction ID: {record.get('interaction_id', '')}. Medications: {record.get('medications', '')}. Severity: {record.get('severity', '')}. Effects: {record.get('effects', '')}. Recommendations: {record.get('recommendations', '')}. Alternatives: {record.get('alternatives', '')}."
        
        return ""
    except Exception as e:
        logger.error(f"Error preparing text for embedding: {str(e)}")
        return ""

# Save embeddings to PT file with compression
def save_category_embeddings(category_embeddings, category, embeddings_dir='data/embeddings'):
    """Save embeddings for a specific category to a PT file with optimization"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create category-specific file path
        file_path = os.path.join(embeddings_dir, f"{category}_embeddings.pt")
        
        # Save embeddings with compression
        torch.save(category_embeddings, file_path, _use_new_zipfile_serialization=True)
        logger.info(f"{category.capitalize()} embeddings saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving {category} embeddings: {str(e)}")
        return False

# Load embeddings from PT file with optimized performance
def load_category_embeddings(category, embeddings_dir='data/embeddings'):
    """Load embeddings for a specific category from a PT file with optimization"""
    try:
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {"status": "in_progress", "message": f"Loading {category} embeddings"}
        
        # Create category-specific file path
        file_path = os.path.join(embeddings_dir, f"{category}_embeddings.pt")
        
        if os.path.exists(file_path):
            # Load embeddings with optimized memory usage
            category_embeddings = torch.load(file_path, map_location=torch.device('cpu'))
            logger.info(f"{category.capitalize()} embeddings loaded from {file_path}")
            
            APP_STATE["loading_progress"][f"{category}_embeddings"] = {
                "status": "complete", 
                "message": f"Loaded {len(category_embeddings)} embeddings"
            }
            return category_embeddings
        else:
            logger.info(f"{category.capitalize()} embeddings file not found")
            APP_STATE["loading_progress"][f"{category}_embeddings"] = {
                "status": "pending", 
                "message": "Embeddings file not found, will generate on demand"
            }
            return None
    except Exception as e:
        logger.error(f"Error loading {category} embeddings: {str(e)}")
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {"status": "error", "message": str(e)}
        return None

# Generate embeddings using HuggingFace with retry mechanism
def get_huggingface_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2", max_retries=3):
    """Get embeddings using HuggingFace Inference API with retry mechanism"""
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url, 
                headers=headers, 
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30  # Add timeout to prevent hanging
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout getting embedding (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for embedding request")
                return None
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error getting embedding from HuggingFace: {str(e)}")
            return None

# Generate embeddings for a category with batch processing
def generate_category_embeddings(category, records, batch_size=10):
    """Generate embeddings for records in batches for better performance"""
    try:
        logger.info(f"Generating embeddings for {category} category...")
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {
            "status": "in_progress", 
            "message": f"Generating embeddings for {len(records)} records"
        }
        
        category_embeddings = []
        total = len(records)
        processed = 0
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = records[i:i+batch_size]
            batch_size = len(batch)
            
            for record in batch:
                # Prepare text representation for embedding
                text = prepare_text_for_embedding(record, category)
                
                # Generate embedding using HuggingFace
                embedding = get_huggingface_embedding(text)
                
                if embedding:
                    category_embeddings.append({
                        'record': record,
                        'embedding': embedding
                    })
                else:
                    id_field = 'case_id' if category == 'clinical' else \
                               'paper_id' if category == 'literature' else \
                               'symptom_id' if category == 'symptom' else \
                               'interaction_id'
                    logger.warning(f"Failed to get embedding for record {record.get(id_field, 'ID')}")
            
            processed += batch_size
            progress = int((processed / total) * 100)
            APP_STATE["loading_progress"][f"{category}_embeddings"]["message"] = f"Generated {processed}/{total} embeddings ({progress}%)"
            logger.info(f"Generated {processed}/{total} embeddings for {category}")
        
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {
            "status": "complete", 
            "message": f"Completed {len(category_embeddings)} embeddings"
        }
        logger.info(f"Completed generating {len(category_embeddings)} embeddings for {category} category")
        return category_embeddings
    
    except Exception as e:
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {"status": "error", "message": str(e)}
        logger.error(f"Error generating embeddings for {category}: {str(e)}")
        return None

# Calculate cosine similarity with optimized implementation
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors with numpy optimization"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Ensure vectors are flattened
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    # Fast dot product calculation
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle zero division
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        
    similarity = dot_product / (norm_v1 * norm_v2)
    return float(similarity)  # Ensure it's a Python float for JSON serialization

# Find similar records based on query with optimized search
def find_similar_records(query, category, category_embeddings, top_k=3):
    """Find records most similar to the query with optimized search"""
    # Generate embedding for the query
    query_embedding = get_huggingface_embedding(query)
    
    if not query_embedding:
        logger.warning("Failed to get embedding for query")
        return []
    
    # Calculate similarities (optimization: we could use numpy vectorization here for larger datasets)
    similarities = []
    
    # Convert query embedding to numpy array once
    query_embedding_np = np.array(query_embedding)
    
    for item in category_embeddings:
        similarity = cosine_similarity(query_embedding_np, item['embedding'])
        similarities.append({
            'record': item['record'],
            'similarity': similarity
        })
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

# Generate response using Gemini with optimized prompting
def generate_gemini_response(query_type, user_query, similar_records):
    """Generate a comprehensive response using Gemini with optimized prompting"""
    try:
        logger.info("Generating response with Gemini...")
        
        # Format similar records for prompt (include only essential fields to reduce token usage)
        simplified_records = []
        
        # Define essential fields by category to reduce token usage
        essential_fields = {
            'clinical': ['case_id', 'age', 'gender', 'symptoms', 'diagnosis', 'treatment', 'outcome'],
            'literature': ['paper_id', 'title', 'authors', 'key_findings', 'methodology'],
            'symptom': ['symptom_id', 'presenting_symptoms', 'diagnosis', 'risk_factors', 'urgency_level'],
            'drug': ['interaction_id', 'medications', 'severity', 'effects', 'recommendations']
        }
        
        for record in [r['record'] for r in similar_records]:
            simplified_record = {key: record.get(key, '') for key in essential_fields.get(query_type, [])}
            simplified_records.append(simplified_record)
            
        formatted_records = json.dumps(simplified_records, indent=2)
        similarity_scores = [f"{r['record'].get('case_id' if query_type == 'clinical' else 'paper_id' if query_type == 'literature' else 'symptom_id' if query_type == 'symptom' else 'interaction_id', 'ID')}: {r['similarity']:.2f}" for r in similar_records]
        
        # Create context-specific prompts with enhanced markdown formatting and emojis
        contexts = {
            'clinical': f"""As a medical AI assistant, analyze this case based on similar cases in our database.

User Query: {user_query}

Similar Cases (with similarity scores):
{', '.join(similarity_scores)}

Detailed Case Information:
{formatted_records}

Provide a clinical analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🏥 Case Similarity Analysis

## 💊 Evidence-Based Treatment Recommendations

## ⚠️ Potential Complications to Monitor

## 📊 Expected Outcomes

## 📋 Follow-up Recommendations

For each section, provide detailed medical analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative.""",

            'literature': f"""As a medical research assistant, analyze this research query based on our literature database.

User Query: {user_query}

Relevant Papers (with similarity scores):
{', '.join(similarity_scores)}

Paper Details:
{formatted_records}

Provide a comprehensive literature review with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 📚 Relevant Studies Analysis

## 🔬 Key Findings Synthesis

## 📈 Treatment Efficacy Data

## 📊 Statistical Evidence

## 🔮 Research Gaps & Future Directions

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative.""",

            'symptom': f"""As a diagnostic assistant, analyze these symptoms based on our symptom database.

User Query: {user_query}

Relevant Cases (with similarity scores):
{', '.join(similarity_scores)}

Case Details:
{formatted_records}

Provide a symptom analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🔍 Potential Diagnoses

## ⚠️ Key Risk Factors

## 👨‍⚕️ Specialist Recommendations

## 🚨 Urgency Assessment

## 📋 Recommended Diagnostic Tests

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative.""",

            'drug': f"""As a pharmaceutical expert, analyze these medication interactions.

User Query: {user_query}

Relevant Interactions (with similarity scores):
{', '.join(similarity_scores)}

Interaction Details:
{formatted_records}

Provide a comprehensive interaction analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## ⚠️ Interaction Severity Assessment

## 👁️ Effects to Monitor

## 💊 Medication Adjustments

## 🔄 Alternative Medications

## 📋 Patient Monitoring Guidelines

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""
        }
        
        # Configure the model with optimized parameters
        generation_config = {
            "temperature": 0.5,  # Reduced for more consistency
            "top_p": 0.90,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Use Gemini API
        global MODEL
        if MODEL is None:
            init_gemini_api()
            if MODEL is None:
                return {
                    "status": "error",
                    "response": "Failed to initialize Gemini model. Please try again later."
                }
        
        # Generate response
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = model.generate_content(contexts[query_type])
        
        logger.info("Response generated successfully")
        return {
            "status": "success",
            "response": response.text
        }
    
    except Exception as e:
        logger.error(f"Error generating Gemini response: {str(e)}")
        return {
            "status": "error",
            "response": f"Error: {str(e)}"
        }

# Process a medical query with optimized flow
@ensure_data_loaded()
def process_medical_query(query_type, user_query):
    """Process a medical query with optimized flow and better error handling"""
    global data, embeddings
    
    try:
        # Validate query type
        valid_types = ['clinical', 'literature', 'symptom', 'drug']
        if query_type not in valid_types:
            return {
                "status": "error",
                "response": f"Invalid query type. Must be one of: {', '.join(valid_types)}"
            }
        
        # Check if embeddings exist for category, load or generate them
        with embedding_lock:
            if query_type not in embeddings or not embeddings[query_type]:
                logger.info(f"Embeddings for {query_type} not loaded, attempting to load")
                
                # Try to load existing embeddings
                embeddings_dir = 'data/embeddings'
                os.makedirs(embeddings_dir, exist_ok=True)
                category_embeddings = load_category_embeddings(query_type, embeddings_dir)
                
                # If embeddings don't exist, generate them
                if not category_embeddings:
                    logger.info(f"Embeddings for {query_type} not found, generating new embeddings")
                    
                    with data_lock:
                        if not data or query_type not in data:
                            data = load_excel_data()
                            if not data or query_type not in data:
                                return {
                                    "status": "error",
                                    "response": f"Failed to load data for {query_type} category."
                                }
                    
                    category_embeddings = generate_category_embeddings(query_type, data[query_type])
                    
                    if not category_embeddings:
                        return {
                            "status": "error",
                            "response": f"Failed to generate embeddings for {query_type} category."
                        }
                    
                    # Save the embeddings for future use
                    save_category_embeddings(category_embeddings, query_type, embeddings_dir)
                
                # Update the global embeddings
                embeddings[query_type] = category_embeddings
        
        # Find similar records
        similar_records = find_similar_records(user_query, query_type, embeddings[query_type])
        
        if not similar_records:
            return {
                "status": "warning",
                "response": "No similar records found in our database. Please try a different query or provide more details."
            }
        
        # Generate response with Gemini
        return generate_gemini_response(query_type, user_query, similar_records)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "status": "error",
            "response": f"An error occurred while processing your query: {str(e)}"
        }

# Background initialization function that doesn't block app startup
def background_initialize():
    """Initialize app data and models in background thread"""
    global APP_STATE, data, embeddings
    
    try:
        logger.info("Starting background initialization...")
        
        # Check API keys
        if not GEMINI_API_KEY:
            APP_STATE["initialization_error"] = "GEMINI_API_KEY environment variable not set"
            logger.warning("GEMINI_API_KEY environment variable not set")
            return False
            
        if not HF_API_KEY:
            APP_STATE["initialization_error"] = "HF_API_KEY environment variable not set"
            logger.warning("HF_API_KEY environment variable not set")
            return False
        
        # Initialize Gemini API
        if not init_gemini_api():
            APP_STATE["initialization_error"] = "Failed to initialize Gemini API"
            return False
        
        # Load data (this updates APP_STATE internally)
        with data_lock:
            data = load_excel_data()
            if not data:
                APP_STATE["initialization_error"] = "Failed to load data"
                logger.error("Failed to load data")
                return False
        
        # Create embeddings directory
        embeddings_dir = 'data/embeddings'
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Check for existing embeddings but don't generate them yet
        # (Lazy loading will handle generation when needed)
        categories = ['clinical', 'literature', 'symptom', 'drug']
        
        for category in categories:
            logger.info(f"Checking embeddings for {category} category...")
            category_embeddings = load_category_embeddings(category, embeddings_dir)
            
            with embedding_lock:
                if category_embeddings:
                    embeddings[category] = category_embeddings
        
        APP_STATE["is_initialized"] = True
        logger.info("Background initialization completed successfully")
        return True
    
    except Exception as e:
        APP_STATE["initialization_error"] = str(e)
        logger.error(f"Error in background initialization: {str(e)}")
        return False

# Start background initialization in a separate thread
initialization_thread = threading.Thread(target=background_initialize)
initialization_thread.daemon = True
initialization_thread.start()

# Routes
@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Get application initialization status"""
    return jsonify({
        "initialized": APP_STATE["is_initialized"],
        "error": APP_STATE["initialization_error"],
        "progress": APP_STATE["loading_progress"]
    })

@app.route('/query', methods=['POST'])
def process_query():
    """Process a medical query"""
    # Get query data from request
    query_data = request.json
    query_type = query_data.get('type')
    user_query = query_data.get('query')
    
    # Validate input
    if not query_type or not user_query:
        return jsonify({
            "status": "error",
            "response": "Missing query type or query text"
        })
    
    # Process the query
    start_time = time.time()
    response = process_medical_query(query_type, user_query)
    processing_time = time.time() - start_time
    
    # Add processing time to response
    response["processing_time"] = f"{processing_time:.2f}s"
    
    return jsonify(response)

@app.route('/analyze-image', methods=['POST'])
def analyze_medical_image():
    """Analyze a medical image"""
    try:
        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({
                "status": "error",
                "response": "Missing image or prompt"
            })
        
        image_file = request.files['image']
        prompt = request.form['prompt']
        
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "response": "No image selected"
            })
            
        # Read the image data
        image_data = image_file.read()
        
        # Analyze the image
        start_time = time.time()
        analysis = analyze_image(image_data, prompt)
        processing_time = time.time() - start_time
        
        # Format the response in markdown and then to HTML
        html_response = format_markdown_response(analysis)
        
        return jsonify({
            "status": "success",
            "response": html_response,
            "raw_response": analysis,
            "processing_time": f"{processing_time:.2f}s"
        })
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return jsonify({
            "status": "error",
            "response": f"Error analyzing image: {str(e)}"
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed diagnostics"""
    global data, embeddings
    
    # Get count of embeddings for each category
    with embedding_lock:
        embedding_counts = {category: len(embs) for category, embs in embeddings.items()}
    
    # Check API keys
    api_status = {
        "gemini_api": GEMINI_API_KEY is not None,
        "huggingface_api": HF_API_KEY is not None
    }
    
    # Check data availability
    with data_lock:
        data_status = {
            "data_loaded": data is not None,
            "categories": list(data.keys()) if data else []
        }
    
    # Check model initialization
    model_status = {
        "model_initialized": MODEL is not None,
        "model_name": MODEL_NAME
    }
    
    # System info
    import platform
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "uptime": time.time() - APP_STATE.get("start_time", time.time())
    }
    
    status = {
        "status": "healthy" if (data and embeddings and GEMINI_API_KEY and HF_API_KEY) else "unhealthy",
        "api_status": api_status,
        "data_status": data_status,
        "model_status": model_status,
        "embedding_status": embedding_counts,
        "system_info": system_info,
        "initialization": {
            "is_initialized": APP_STATE["is_initialized"],
            "error": APP_STATE["initialization_error"],
        }
    }
    
    return jsonify(status)

@app.route('/refresh-embeddings', methods=['POST'])
def refresh_embeddings():
    """Endpoint to regenerate embeddings for a specific category"""
    global data, embeddings
    
    if not HF_API_KEY:
        return jsonify({
            "status": "error",
            "message": "HF_API_KEY environment variable not set"
        })
    
    try:
        # Get category to refresh
        request_data = request.json or {}
        category = request_data.get('category')
        
        if not category:
            return jsonify({
                "status": "error",
                "message": "Category parameter is required"
            })
            
        # Load data if needed
        with data_lock:
            if not data:
                data = load_excel_data()
                
            if not data or category not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid category or failed to load data: {category}"
                })
        
        # Create embeddings directory
        embeddings_dir = 'data/embeddings'
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Update state
        APP_STATE["loading_progress"][f"{category}_embeddings"] = {
            "status": "in_progress", 
            "message": "Refreshing embeddings"
        }
        
        start_time = time.time()
        
        # Generate new embeddings
        category_embeddings = generate_category_embeddings(category, data[category])
        
        if not category_embeddings:
            return jsonify({
                "status": "error",
                "message": f"Failed to generate embeddings for {category}"
            })
        
        # Save the new embeddings
        save_success = save_category_embeddings(category_embeddings, category, embeddings_dir)
        
        if not save_success:
            return jsonify({
                "status": "error",
                "message": f"Failed to save embeddings for {category}"
            })
        
        # Update the global embeddings
        with embedding_lock:
            embeddings[category] = category_embeddings
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "message": f"Embeddings for {category} refreshed successfully",
            "processing_time": f"{processing_time:.2f}s",
            "embedding_count": len(category_embeddings)
        })
        
    except Exception as e:
        logger.error(f"Failed to refresh embeddings: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to refresh embeddings: {str(e)}"
        })

# Server static files from templates directory
@app.route('/templates/<path:path>')
def send_template(path):
    return send_from_directory('templates', path)

# Create necessary directories on startup
def create_folders():
    """Create necessary folders for the application"""
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('cache/image_analysis', exist_ok=True)

# Record start time
APP_STATE["start_time"] = time.time()

# Create folders immediately
create_folders()

if __name__ == '__main__':
    # Create necessary folders
    create_folders()
    
    # Start Flask app without waiting for full initialization
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
