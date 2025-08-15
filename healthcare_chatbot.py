import os
import json
import uuid
import base64
import requests
import tempfile
import time
from datetime import datetime
import gradio as gr
import logging
from typing import Dict, List, Optional, Tuple, Union
from gradio.themes import Soft
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Configuration class
class Config:
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __get_pydantic_core_schema__(self, source_type, handler):
        return handler(source_type)

    def set_model(self, model_name):
        if model_name in self.available_models:
            self.selected_model = model_name
            return True
        return False

    def __init__(self):
        # Load API keys from environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Initialize configuration with no defaults
        self.selected_model = None
        self.voice_id = None
        self.language = None
        self.output_format = None
        
        # Model identifiers for Groq Cloud API
        self.available_models = [
            "llama3-70b-8192"
        ]
        
        # Output format options
        self.output_formats = [
            "medical_advice",
            "concise_summary",
            "step_by_step_instructions",
            "technical_explanation"
        ]
        
        # Available languages
        self.available_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "hi": "Hindi",
            "ar": "Arabic",
            "pt": "Portuguese",
            "ru": "Russian"
        }
        
        # ElevenLabs voice IDs for API calls
        self.voice_options = {
            "21m00Tcm4TlvDq8ikWAM": "Rachel (Female)",
            "AZnzlk1XvdvUeBnXmlld": "Domi (Male)",
            "EXAVITQu4vr4xnSDxMaL": "Bella (Female)",
            "ErXwobaYiN019PkySvjV": "Antoni (Male)",
            "MF3mGyEYCl7XYWbV9V6O": "Elli (Female)",
            "TxGEqnHWrfWFTfGW9XjX": "Josh (Male)",
            "VR6AewLTigWG4xSOukaG": "Arnold (Male)",
            "pNInz6obpgDQGcFmaJgB": "Adam (Male)",
            "yoZ06aMxZJJ28mfd3POQ": "Sam (Male)",
            "jBpfuIE2acCO8z3wKNLl": "Glinda (Female)"
        }

    def is_configured(self) -> bool:
        """Check if all required configurations are set"""
        return all([
            self.selected_model is not None,
            self.voice_id is not None,
            self.language is not None,
            self.output_format is not None
        ])

    def validate_model(self, model) -> tuple[bool, str]:
        """Validate model selection"""
        if not model:
            return False, "No model selected. Please select a model."
        if model not in self.available_models:
            return False, f"Invalid model: {model}. Please select from available models."
        return True, ""

    def validate_voice(self, voice) -> tuple[bool, str]:
        """Validate voice selection"""
        if not voice:
            return False, "No voice selected. Please select a voice."
        if voice not in self.voice_options:
            return False, f"Invalid voice: {voice}. Please select from available voices."
        return True, ""

    def validate_language(self, language) -> tuple[bool, str]:
        """Validate language selection"""
        if not language:
            return False, "No language selected. Please select a language."
        if language not in self.available_languages:
            return False, f"Invalid language: {language}. Please select from available languages."
        return True, ""

# Initialize configuration
config = Config()

# Create model selection dropdown
def create_model_selection():
    """Create model selection dropdown UI"""
    return gr.Dropdown(
        label="Select AI Model",
        choices=config.available_models,
        value=None,
        interactive=True
    )

# Add model selection callback
def update_model_selection(model_name):
    """Update selected model based on user choice"""
    if config.set_model(model_name):
        return f"Model successfully changed to {model_name}"
    return "Invalid model selection"

# Log initial configuration
logging.info(f"Initial configuration loaded - Model: {config.selected_model}, Voice: {config.voice_id}")

# Helper functions
def save_uploaded_file(file_obj):
    if file_obj is None:
        return None

    # Case 1: If already a filepath (e.g., from gr.Audio with type="filepath")
    if isinstance(file_obj, str):
        return file_obj

    # Case 2: If it's a file-like object (e.g., from direct uploads)
    try:
        file_extension = os.path.splitext(file_obj.name)[1]
    except AttributeError:
        # Fall back if .name is missing
        file_extension = ".tmp"

    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{file_extension}")
    
    with open(temp_file_path, "wb") as f:
        f.write(file_obj.read())
    
    return temp_file_path

# API Interaction Functions
class GroqAPI:
    @staticmethod
    def process_multimodal_input(text_input, config):
        """Process text input using Groq API with retry logic"""
        if not config.groq_api_key:
            return "Error: Groq API key not set", None
            
        # Validate selected model
        if not config.selected_model:
            return "Error: No model selected. Please select a model first.", None
        if config.selected_model not in config.available_models:
            return f"Error: Invalid model '{config.selected_model}'. Please select a valid model.", None
        
        headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Build appropriate system prompt based on configuration
        system_prompt = f"""You are an advanced healthcare assistant designed to help with medical inquiries.
        
Current configuration:
- Language: {config.available_languages[config.language]}
- Output Format: {config.output_format}

Guidelines:
1. Provide responses in {config.available_languages[config.language]}.
2. Structure your response as a {config.output_format.replace('_', ' ')}.
3. Always prioritize patient safety.
4. Recommend seeking professional medical help when appropriate.
5. Adhere to medical accuracy and HIPAA compliance.
6. Do not make definitive diagnoses but provide educational information.
"""

        # Prepare request payload
        payload = {
            "model": config.selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input}
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }
        
        logger.info(f"Using Groq model: {config.selected_model}")
        
        # Retry configuration
        max_retries = 3
        base_delay = 1  # seconds
        max_delay = 10  # seconds
        timeout = 30  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempting Groq API call (attempt {attempt + 1}/{max_retries + 1})")
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                # Handle different status codes appropriately
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        text_response = response_data["choices"][0]["message"]["content"]
                        
                        # Extract and format usage data
                        usage_data = None
                        if "usage" in response_data:
                            usage_data = f"Model: {config.selected_model}\n"
                            usage_data += f"Prompt tokens: {response_data['usage']['prompt_tokens']}\n"
                            usage_data += f"Completion tokens: {response_data['usage']['completion_tokens']}\n"
                            usage_data += f"Total tokens: {response_data['usage']['total_tokens']}"
                        
                        logger.info("Groq API call successful")
                        return text_response, usage_data
                    else:
                        logger.error("Groq API returned empty response")
                        return "Error: No response generated by the AI model", None
                        
                elif response.status_code in [500, 502, 503, 504]:  # Server errors - retry
                    logger.warning(f"Groq API server error (status {response.status_code}): {response.text[:200]}...")
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        return "🚨 Groq API is temporarily unavailable (server error). Please try again in a few moments.", None
                        
                elif response.status_code == 429:  # Rate limiting - retry with longer delay
                    logger.warning(f"Groq API rate limit exceeded: {response.text[:200]}...")
                    if attempt < max_retries:
                        delay = min(base_delay * (3 ** attempt), max_delay)
                        logger.info(f"Rate limited, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        return "⏱️ Rate limit exceeded. Please wait a moment and try again.", None
                        
                elif response.status_code in [401, 403]:  # Authentication errors - don't retry
                    logger.error(f"Groq API authentication error (status {response.status_code}): {response.text[:200]}...")
                    return "🔑 Authentication error. Please check your Groq API key and try again.", None
                    
                elif response.status_code == 400:  # Bad request - don't retry
                    logger.error(f"Groq API bad request (status {response.status_code}): {response.text[:200]}...")
                    return "❌ Invalid request. Please check your input and try again.", None
                    
                else:  # Other errors - retry once
                    logger.error(f"Groq API unexpected error (status {response.status_code}): {response.text[:200]}...")
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.info(f"Unexpected error, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        return f"❌ Groq API error (status {response.status_code}). Please try again later.", None
                
            except requests.exceptions.Timeout:
                logger.warning(f"Groq API request timeout (attempt {attempt + 1})")
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Request timeout, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return "⏰ Request timed out. The Groq API is taking too long to respond. Please try again.", None
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Groq API connection error (attempt {attempt + 1})")
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Connection error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return "🌐 Unable to connect to Groq API. Please check your internet connection and try again.", None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Groq API request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Request error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return f"❌ Error communicating with Groq API: {str(e)}", None
                    
            except Exception as e:
                logger.error(f"Unexpected error during Groq API call (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Unexpected error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return f"❌ Unexpected error: {str(e)}", None
        
        # This should never be reached, but just in case
        return "❌ Maximum retry attempts exceeded. Please try again later.", None

class ElevenLabsAPI:
    @staticmethod
    def convert_speech_to_text(audio_path, config):
        """Convert speech to text using ElevenLabs API"""
        if not config.elevenlabs_api_key:
            return "Error: ElevenLabs API key not set"
        
        try:
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            
            headers = {
                "xi-api-key": config.elevenlabs_api_key
            }
            
            with open(audio_path, "rb") as audio_file:
                files = {
                    "audio": ("audio.mp3", audio_file),
                    "model_id": (None, "whisper-1")
                }
                
                response = requests.post(url, headers=headers, files=files)
                response.raise_for_status()
                
                result = response.json()
                return result.get("text", "No text transcription available")
                
        except Exception as e:
            logger.error(f"ElevenLabs speech-to-text error: {str(e)}")
            return f"Error in speech-to-text conversion: {str(e)}"

    @staticmethod
    def convert_text_to_speech(text, config):
        """Convert text to speech using ElevenLabs API"""
        if not config.elevenlabs_api_key:
            logger.error("ElevenLabs API key not set")
            return None
            
        # Validate voice selection
        if config.voice_id not in config.voice_options:
            logger.error(f"Invalid voice ID selected: {config.voice_id}")
            return None
        
        # Validate text is not empty
        if not text or len(text.strip()) == 0:
            logger.error("Cannot generate speech from empty text")
            return None
            
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}"
            logger.info(f"Using ElevenLabs voice: {config.voice_id} ({config.voice_options.get(config.voice_id, 'Unknown')})")
            logger.debug(f"Generating speech for text: {text[:100]}...")
            
            headers = {
                "xi-api-key": config.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            logger.info(f"Sending request to ElevenLabs API with voice_id: {config.voice_id}")
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
            response.raise_for_status()
            
            # Create a temporary file to store the audio
            temp_audio_file = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp3")
            with open(temp_audio_file, "wb") as f:
                f.write(response.content)
            
            # Verify the file was created and has content
            if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                logger.info(f"Audio file created successfully: {temp_audio_file} ({os.path.getsize(temp_audio_file)} bytes)")
                return temp_audio_file
            else:
                logger.error(f"Failed to create audio file or file is empty: {temp_audio_file}")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ElevenLabs API request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"ElevenLabs text-to-speech error: {str(e)}")
            return None

# Main application function
def process_healthcare_query(
    text_input, 
    audio_input,
    medical_history=None,
    patient_age=None,
    patient_gender=None
):
    """Process a healthcare query with text and audio input"""
    
    # Validate configuration before processing
    if not config.is_configured():
        return "⚠️ Please configure all required settings (Model, Voice, Language, and Output Format) before sending queries.", None, None
    
    # Process input from text field
    query_text = text_input if text_input else ""
    
    # Process audio input if available
    audio_path = None
    if audio_input is not None:
        audio_path = save_uploaded_file(audio_input)
        transcribed_text = ElevenLabsAPI.convert_speech_to_text(audio_path, config)
        if not transcribed_text.startswith("Error"):
            if query_text:
                query_text += "\n\n"
            query_text += f"Transcribed audio: {transcribed_text}"
    
    # Add additional patient information if provided
    if medical_history or patient_age or patient_gender:
        query_text += "\n\nAdditional patient information:"
        if patient_age:
            query_text += f"\n- Age: {patient_age}"
        if patient_gender:
            query_text += f"\n- Gender: {patient_gender}"
        if medical_history:
            query_text += f"\n- Medical History: {medical_history}"
    
    # Get response from Groq API
    if not query_text:
        return "Please provide either text or audio input.", None, None
    
    # Get response from Groq API with error handling
    try:
        text_response, usage_data = GroqAPI.process_multimodal_input(query_text, config)
    except Exception as e:
        logger.error(f"Error processing input with Groq API: {str(e)}")
        return f"Error processing your request with model {config.selected_model}. Please check your input or try again later.", None, None
    
    # Generate audio response with error handling
    audio_response_path = None
    if not text_response.startswith("Error"):
        logger.info(f"Attempting to generate audio response with voice ID: {config.voice_id}")
        try:
            logger.debug(f"Text for speech generation (first 100 chars): {text_response[:100]}...")
            audio_response_path = ElevenLabsAPI.convert_text_to_speech(text_response, config)
            
            if audio_response_path is None:
                logger.warning(f"Failed to generate audio with voice ID: {config.voice_id}")
            else:
                logger.info(f"Successfully generated audio response at: {audio_response_path}")
                
                if not os.path.exists(audio_response_path):
                    logger.error(f"Audio file does not exist: {audio_response_path}")
                    audio_response_path = None
                elif os.path.getsize(audio_response_path) == 0:
                    logger.error(f"Audio file is empty: {audio_response_path}")
                    audio_response_path = None
        except Exception as e:
            logger.error(f"Error generating speech with ElevenLabs API: {str(e)}")
            audio_response_path = None
    else:
        logger.warning(f"Not generating audio for error response: {text_response[:50]}...")
    
    # Log the final outcome
    if audio_response_path:
        logger.info(f"Returning audio response path: {audio_response_path}")
    else:
        logger.warning("No audio response was generated")
        
    return text_response, audio_response_path, usage_data

def set_api_keys(groq_key, elevenlabs_key):
    """Set API keys for Groq and ElevenLabs"""
    result = []
    groq_key_processed = False
    elevenlabs_key_processed = False
    
    if groq_key is not None:
        try:
            groq_key_processed = True
            groq_key_str = str(groq_key).strip()
            if groq_key_str:
                config.groq_api_key = groq_key_str
                logger.info("Groq API key set successfully")
                result.append("✅ Groq API key set successfully")
            else:
                logger.warning("Empty Groq API key provided")
                result.append("❌ Groq API key is empty or contains only whitespace")
        except Exception as e:
            logger.error(f"Error setting Groq API key: {str(e)}")
            result.append(f"❌ Error setting Groq API key: {str(e)}")
    
    if elevenlabs_key is not None:
        try:
            elevenlabs_key_processed = True
            elevenlabs_key_str = str(elevenlabs_key).strip()
            if elevenlabs_key_str:
                config.elevenlabs_api_key = elevenlabs_key_str
                logger.info("ElevenLabs API key set successfully")
                result.append("✅ ElevenLabs API key set successfully")
            else:
                logger.warning("Empty ElevenLabs API key provided")
                result.append("❌ ElevenLabs API key is empty or contains only whitespace")
        except Exception as e:
            logger.error(f"Error setting ElevenLabs API key: {str(e)}")
            result.append(f"❌ Error setting ElevenLabs API key: {str(e)}")
    
    if not groq_key_processed and not elevenlabs_key_processed:
        logger.warning("No API keys were provided to set_api_keys function")
        return "Please provide at least one API key"
    
    return "\n".join(result)

def update_configuration(model, voice, language, output_format):
    """Update the configuration settings"""
    # Validate model selection
    if not model:
        return "Error: No model selected. Please select a model."
    if model not in config.available_models:
        return f"Error: Invalid model '{model}'. Please select from available models."
    
    # Find the voice_id corresponding to the selected voice name
    voice_id = None
    for vid, vname in config.voice_options.items():
        if vname == voice:
            voice_id = vid
            break
    
    # Validate voice selection
    if voice_id is None:
        return f"Error: Invalid voice selected: {voice}. Please choose from available voices."
    
    # Validate language selection
    if language not in config.available_languages:
        return f"Error: Invalid language selected: {language}. Please choose from available languages."
    
    # Validate output format
    if output_format not in config.output_formats:
        return f"Error: Invalid output format selected: {output_format}. Please choose from available formats."
    
    # Update configuration if all validations pass
    config.selected_model = model
    config.voice_id = voice_id
    config.language = language
    config.output_format = output_format
    
    logger.info(f"Configuration updated - Model: {model}, Voice: {voice}, Language: {language}, Format: {output_format}")
    
    return f"Configuration updated successfully:\n- Model: {model}\n- Voice: {voice}\n- Language: {config.available_languages[language]}\n- Output Format: {output_format.replace('_', ' ')}"

# Define the Gradio interface
def create_gradio_interface():
    # Define theme
    theme = Soft(
        primary_hue="teal",
        secondary_hue="blue",
    )
    
    # Create Blocks interface
    with gr.Blocks(title="Healthcare Multimodal Chatbot", theme=theme) as app:
        gr.Markdown("# 🏥 Healthcare Chatbot")
        gr.Markdown("Powered by Groq Cloud API with multiple language models and ElevenLabs for voice processing")
        
        # API keys tab
        with gr.Tab("API Setup"):
            gr.Markdown("### Set your API keys")
            gr.Markdown("Enter one or both API keys below. Leave fields empty if you don't want to use a service.")
            with gr.Row():
                groq_api_key = gr.Textbox(
                    label="Groq API Key", 
                    type="password",
                    placeholder="Enter Groq API key (required for text processing)",
                    value="" if config.groq_api_key is None else config.groq_api_key
                )
                elevenlabs_api_key = gr.Textbox(
                    label="ElevenLabs API Key", 
                    type="password",
                    placeholder="Enter ElevenLabs API key (required for voice features)",
                    value="" if config.elevenlabs_api_key is None else config.elevenlabs_api_key
                )
            
            api_key_btn = gr.Button("Save API Keys", variant="primary")
            api_key_status = gr.Textbox(
                label="Status", 
                interactive=False,
                placeholder="Status will appear here after saving"
            )
            
            def handle_api_keys(groq_key, elevenlabs_key):
                if groq_key is None or groq_key == "":
                    groq_key = None
                else:
                    groq_key = str(groq_key).strip()
                    if not groq_key:
                        groq_key = None
                
                if elevenlabs_key is None or elevenlabs_key == "":
                    elevenlabs_key = None
                else:
                    elevenlabs_key = str(elevenlabs_key).strip()
                    if not elevenlabs_key:
                        elevenlabs_key = None
                
                return set_api_keys(groq_key, elevenlabs_key)
            
            api_key_btn.click(
                handle_api_keys,
                inputs=[groq_api_key, elevenlabs_api_key],
                outputs=api_key_status
            )
        
        # Configuration tab
        with gr.Tab("Configuration"):
            gr.Markdown("### Configure the Chatbot")
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=config.available_models,
                    value=config.selected_model,
                    label="Language Model",
                    info="Select the AI model to process your healthcare queries",
                    interactive=True
                )
                
                voice_dropdown = gr.Dropdown(
                    choices=list(config.voice_options.values()),
                    value=config.voice_options.get(config.voice_id),
                    label="Voice Selection",
                    type="value",
                    info="Select a voice for audio responses",
                    interactive=True
                )
            
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=config.available_languages,
                    value=config.language,
                    label="Language",
                    type="value",
                    info="Select the response language",
                    interactive=True
                )
                
                output_format_dropdown = gr.Dropdown(
                    choices=config.output_formats,
                    value=None,
                    label="Output Format",
                    info="Select the format for the AI's response",
                    interactive=True
                )
            
            config_btn = gr.Button("Update Configuration", variant="primary")
            config_status = gr.Textbox(label="Configuration Status", interactive=False)
            
            config_btn.click(
                update_configuration,
                inputs=[model_dropdown, voice_dropdown, language_dropdown, output_format_dropdown],
                outputs=config_status
            )
        
        # Main chat interface
        with gr.Tab("Healthcare Chat"):
            gr.Markdown("### Describe your health concern")
            
            # Add status information
            with gr.Row():
                gr.Markdown("""
                ℹ️ **Important Notes:**
                - If you encounter "503 Service Unavailable" errors, the Groq API is temporarily overloaded. The system will automatically retry your request.
                - For persistent issues, try again in a few minutes as API services may be experiencing high traffic.
                - All requests include automatic retry with exponential backoff for better reliability.
                """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text Input (Optional)",
                        placeholder="Describe your symptoms or ask a health-related question...",
                        lines=3
                    )
                    
                    audio_input = gr.Audio(
                        label="Voice Input (Optional)",
                        type="filepath",
                        format="mp3",
                        sources=["upload"]
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Additional Information (Optional)")
                    patient_age = gr.Number(label="Patient Age", precision=0)
                    patient_gender = gr.Radio(
                        ["Male", "Female", "Other", "Prefer not to say"],
                        label="Patient Gender"
                    )
                    medical_history = gr.Textbox(
                        label="Relevant Medical History",
                        lines=3,
                        placeholder="List any relevant conditions, medications, or allergies..."
                    )
            
            submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=3):
                    text_output = gr.Textbox(
                        label="Healthcare Assistant Response",
                        lines=10,
                        interactive=False
                    )
                    
                    audio_output = gr.Audio(
                        label="Voice Response",
                        interactive=False,
                        visible=True,
                        format="mp3",
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    usage_info = gr.Textbox(
                        label="Processing Information",
                        interactive=False
                    )
            
            submit_btn.click(
                process_healthcare_query,
                inputs=[
                    text_input,
                    audio_input,
                    medical_history,
                    patient_age,
                    patient_gender
                ],
                outputs=[text_output, audio_output, usage_info]
            )
        
        # Help and information tab
        with gr.Tab("Help & Information"):
            gr.Markdown("""
            ## How to Use This Healthcare Chatbot
            
            ### Input Options
            - **Text**: Type your health concern, symptoms, or question.
            - **Voice**: Speak your symptoms or questions.
            - **Additional Information**: Provide optional details like age, gender, and medical history.
            
            ### Privacy & Compliance
            - All data is processed according to HIPAA guidelines.
            - No user data is stored permanently.
            - Temporary files are deleted after processing.
            
            ### Important Disclaimers
            - This chatbot is for informational purposes only and is not a substitute for professional medical advice.
            - Always consult with a qualified healthcare professional for medical advice.
            - In case of emergency, call emergency services immediately.
            
            ### Supported Languages
            This chatbot supports multiple languages including English, Spanish, French, German, Chinese, Hindi, Arabic, Portuguese, Russian, and Japanese. 
            
            ### Supported Models
            The chatbot supports various AI models including:
            - LLaMA 3 (70B and 8B variants)
            
            ### Output Formats
            - **Medical Advice**: General healthcare guidance
            - **Concise Summary**: Brief overview of the situation
            - **Step-by-Step Instructions**: Detailed procedural guidance
            - **Technical Explanation**: More detailed medical information
            """)
    
    return app

# Run the application
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=False)