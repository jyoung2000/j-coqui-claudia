#!/usr/bin/env python3
"""
Enhanced Coqui TTS Gradio Interface with VCTK 109 Voices Support
Comprehensive web interface for text-to-speech, voice cloning, and audio management
"""

import os
import sys
import json
import logging
import tempfile
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Set environment variables before importing numba-dependent libraries
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create cache directory
cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.makedirs(cache_dir, exist_ok=True)

import gradio as gr
import torch
import numpy as np

# Add TTS to path
sys.path.insert(0, '/app')

# Import TTS with error handling
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: TTS not fully available: {e}")
    TTS_AVAILABLE = False

# Import audio libraries
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Audio libraries not available: {e}")
    AUDIO_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv('PORT', 2201))
MODELS_DIR = os.getenv('TTS_MODELS_DIR', '/app/models')
VOICES_DIR = os.getenv('TTS_VOICES_DIR', '/app/voices')
OUTPUTS_DIR = os.getenv('TTS_OUTPUTS_DIR', '/app/outputs')
UPLOADS_DIR = os.getenv('TTS_UPLOADS_DIR', '/app/uploads')

# Ensure directories exist
for directory in [MODELS_DIR, VOICES_DIR, OUTPUTS_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global variables
current_model = None
model_manager = None
available_models = {}
voice_profiles = {}

# VCTK 109 Complete Speaker Database with Full Details
VCTK_SPEAKERS = {
    # Scottish Speakers
    "p225": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 22, Edinburgh",
    "p234": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 22, Fife", 
    "p240": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 32, Dundee",
    "p241": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 21, Argyll",
    "p246": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 22, Yorkshire",
    "p247": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 22, Orkney",
    "p249": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 22, Fife",
    "p251": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 26, Edinburgh",
    "p252": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 22, Edinburgh",
    "p255": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 19, Birmingham",
    "p260": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 21, Orkney",
    "p262": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 23, Edinburgh",
    "p263": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 22, Fife",
    "p265": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 23, Argyll",
    "p271": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 19, Edinburgh",
    "p272": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 25, Edinburgh",
    "p275": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 23, Midlothian",
    "p281": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 29, Edinburgh",
    "p283": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 19, Fife",
    "p285": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Male, Scottish, Age 23, Edinburgh",
    "p307": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Female, Scottish, Age 27, Fife",
    
    # English Speakers
    "p226": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Surrey",
    "p227": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 38, Cumbria",
    "p228": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Southern England",
    "p229": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Southern England",
    "p230": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Stockton-on-Tees",
    "p231": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Southern England",
    "p232": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Southern England",
    "p233": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Staffordshire",
    "p236": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Manchester",
    "p237": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Yorkshire",
    "p238": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Potters Bar",
    "p239": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Essex",
    "p243": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, London",
    "p244": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Leeds",
    "p248": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Indian Heritage",
    "p250": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Hertfordshire",
    "p254": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 21, Surrey",
    "p256": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 24, Birmingham",
    "p257": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 24, Surrey",
    "p258": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Southern England",
    "p259": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Nottingham",
    "p264": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, London",
    "p267": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Yorkshire",
    "p268": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Midlands",
    "p269": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Hertfordshire",
    "p270": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 21, Gloucester",
    "p273": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Lincolnshire",
    "p274": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Cambridgeshire",
    "p276": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 24, Lancashire",
    "p277": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Midlands",
    "p278": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Surrey",
    "p279": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Oxfordshire",
    "p280": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 29, Nottingham",
    "p282": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Newcastle",
    "p284": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, London",
    "p286": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Newcastle",
    "p287": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, Yorkshire",
    "p292": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 23, London",
    "p294": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, West Midlands",
    "p297": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 25, Oxfordshire",
    "p301": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 21, Nottingham",
    "p302": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 26, West Midlands",
    "p303": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 26, Manchester",
    "p304": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 19, London",
    "p305": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Bedfordshire",
    "p306": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Hertfordshire",
    "p308": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Stockport",
    "p310": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 21, Yorkshire",
    "p311": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 21, Stockport",
    "p312": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Cornwall",
    "p313": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 26, West Sussex",
    "p316": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 28, Yorkshire",
    "p317": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Newcastle",
    "p318": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 22, Yorkshire",
    "p323": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 21, Yorkshire",
    "p326": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 26, Yorkshire",
    "p329": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Hertfordshire",
    "p330": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Liverpool",
    "p333": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, West Midlands",
    "p334": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 18, Manchester",
    "p335": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 23, Stockport",
    "p336": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 19, Birmingham",
    "p339": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 21, Cambridge",
    "p340": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 21, Norfolk",
    "p341": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 18, Plymouth",
    "p343": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 20, Yorkshire",
    "p345": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 20, Leeds",
    "p347": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 20, Yorkshire",
    "p360": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 19, Warwick",
    "p361": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 19, Leeds",
    "p362": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 22, Hertfordshire",
    "p363": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Male, English, Age 16, West Midlands",
    "p376": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Female, English, Age 19, Birmingham",
    
    # Irish Speakers
    "p245": "🇮🇪 Male, Irish, Age 25, Ireland",
    "p266": "🇮🇪 Female, Irish, Age 22, Derry",
    "p288": "🇮🇪 Female, Irish, Age 22, Dublin",
    "p295": "🇮🇪 Female, Irish, Age 23, Dublin",
    "p298": "🇮🇪 Male, Irish, Age 24, Dublin",
    "p300": "🇮🇪 Female, Irish, Age 19, Dublin",
    "p314": "🇮🇪 Female, Irish, Age 26, Dublin",
    "p364": "🇮🇪 Male, Irish, Age 18, Dublin",
    
    # Northern Irish Speakers
    "p261": "🇬🇧 Female, Northern Irish, Age 23, Belfast",
    "p293": "🇬🇧 Female, Northern Irish, Age 22, Belfast",
    "p351": "🇬🇧 Male, Northern Irish, Age 21, Belfast",
    
    # Welsh Speakers
    "p253": "🏴󠁧󠁢󠁷󠁬󠁳󠁿 Female, Welsh, Age 22, Cardiff",
    
    # American Speakers
    "s5": "🇺🇸 Female, American, Age 22, New York",
    
    # Australian Speakers
    "p374": "🇦🇺 Male, Australian, Age 28, Victoria"
}

# Create categorized speaker lists for better organization
SPEAKER_CATEGORIES = {
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish": [k for k, v in VCTK_SPEAKERS.items() if "Scottish" in v],
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English": [k for k, v in VCTK_SPEAKERS.items() if "English" in v],
    "🇮🇪 Irish": [k for k, v in VCTK_SPEAKERS.items() if "Irish" in v and "Northern" not in v],
    "🇬🇧 Northern Irish": [k for k, v in VCTK_SPEAKERS.items() if "Northern Irish" in v],
    "🏴󠁧󠁢󠁷󠁬󠁳󠁿 Welsh": [k for k, v in VCTK_SPEAKERS.items() if "Welsh" in v],
    "🇺🇸 American": [k for k, v in VCTK_SPEAKERS.items() if "American" in v],
    "🇦🇺 Australian": [k for k, v in VCTK_SPEAKERS.items() if "Australian" in v]
}

def initialize_models():
    """Initialize TTS models and model manager"""
    global model_manager, available_models
    
    if not TTS_AVAILABLE:
        logger.error("TTS libraries not available")
        return
    
    try:
        model_manager = ModelManager()
        models_dict = model_manager.list_models()
        
        available_models = {
            'tts_models': models_dict.get('tts_models', {}),
            'vocoder_models': models_dict.get('vocoder_models', {}),
            'voice_conversion_models': models_dict.get('voice_conversion_models', {})
        }
        
        logger.info(f"Initialized with {len(available_models['tts_models'])} TTS models")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        available_models = {'tts_models': {}, 'vocoder_models': {}, 'voice_conversion_models': {}}

def get_model_list():
    """Get formatted list of available models"""
    model_list = []
    
    # Add TTS models
    for lang_family, models in available_models.get('tts_models', {}).items():
        for dataset, model_types in models.items():
            for model_type in model_types:
                model_name = f"tts_models/{lang_family}/{dataset}/{model_type}"
                model_list.append(model_name)
    
    return sorted(model_list)

def load_model(model_name: str):
    """Load a specific TTS model"""
    global current_model
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {model_name} on {device}")
        
        current_model = TTS(model_name).to(device)
        logger.info(f"Successfully loaded {model_name}")
        
        return current_model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        
        # Try fallback model
        try:
            current_model = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
            logger.info("Loaded fallback model")
            return current_model
        except:
            return None

def get_current_model():
    """Get or initialize current model"""
    global current_model
    
    if current_model is None:
        # Default to VCTK model for multiple speakers
        return load_model("tts_models/en/vctk/vits")
    
    return current_model

def load_voice_profiles():
    """Load saved voice profiles"""
    global voice_profiles
    
    voices_file = os.path.join(VOICES_DIR, 'voice_profiles.json')
    try:
        if os.path.exists(voices_file):
            with open(voices_file, 'r') as f:
                voice_profiles = json.load(f)
        else:
            voice_profiles = {}
    except Exception as e:
        logger.error(f"Error loading voice profiles: {e}")
        voice_profiles = {}

def save_voice_profiles():
    """Save voice profiles to disk"""
    voices_file = os.path.join(VOICES_DIR, 'voice_profiles.json')
    try:
        with open(voices_file, 'w') as f:
            json.dump(voice_profiles, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving voice profiles: {e}")

def synthesize_speech(
    text: str,
    model_name: str,
    speaker: str,
    language: str,
    speed: float,
    pitch: float,
    emotion: str,
    voice_file=None
) -> Tuple[str, str]:
    """Synthesize speech with given parameters"""
    
    if not text:
        return None, "Please enter text to synthesize"
    
    try:
        # Load or switch model if needed
        if model_name and (current_model is None or 
                          getattr(current_model, 'model_name', '') != model_name):
            model = load_model(model_name)
        else:
            model = get_current_model()
        
        if model is None:
            return None, "No TTS model available"
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(OUTPUTS_DIR, filename)
        
        # Handle different synthesis cases
        if voice_file is not None:
            # Voice cloning from uploaded file
            temp_path = f"/tmp/upload_{uuid.uuid4().hex[:8]}.wav"
            
            if hasattr(voice_file, 'name'):
                # File upload from Gradio
                import shutil
                shutil.copy(voice_file.name, temp_path)
            else:
                # Direct audio data
                sf.write(temp_path, voice_file[1], voice_file[0])
            
            model.tts_to_file(
                text=text,
                speaker_wav=temp_path,
                language=language,
                file_path=output_path
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        elif speaker and speaker != "None":
            # Use specific speaker
            if speaker.startswith("VCTK_"):
                # VCTK speaker
                speaker_id = speaker
            else:
                # Cloned voice profile
                if speaker in voice_profiles:
                    speaker_wav = voice_profiles[speaker]['audio_path']
                    model.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=output_path
                    )
                    return output_path, "Success!"
                else:
                    speaker_id = speaker
            
            # Multi-speaker model synthesis
            model.tts_to_file(
                text=text,
                speaker=speaker_id,
                language=language,
                file_path=output_path
            )
        else:
            # Single speaker synthesis
            model.tts_to_file(
                text=text,
                language=language,
                file_path=output_path
            )
        
        return output_path, "Success!"
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return None, f"Error: {str(e)}"

def clone_voice(audio_file, voice_name: str) -> str:
    """Clone a voice from uploaded audio"""
    
    if audio_file is None:
        return "Please upload an audio file"
    
    if not voice_name:
        voice_name = f"voice_{uuid.uuid4().hex[:8]}"
    
    try:
        # Save uploaded audio
        filename = f"{voice_name}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = os.path.join(VOICES_DIR, filename)
        
        # Handle different input types
        if hasattr(audio_file, 'name'):
            # File upload
            import shutil
            shutil.copy(audio_file.name, audio_path)
        else:
            # Direct audio (sr, data)
            sf.write(audio_path, audio_file[1], audio_file[0])
        
        # Process audio for better compatibility
        if AUDIO_LIBS_AVAILABLE:
            audio, sr = librosa.load(audio_path, sr=22050)
            sf.write(audio_path, audio, sr)
            duration = len(audio) / sr
        else:
            duration = 0
        
        # Save voice profile
        voice_profiles[voice_name] = {
            'name': voice_name,
            'audio_path': audio_path,
            'created_at': datetime.now().isoformat(),
            'duration': duration
        }
        
        save_voice_profiles()
        
        return f"Voice '{voice_name}' cloned successfully!"
        
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        return f"Error: {str(e)}"

def get_speaker_list(model_name: str) -> List[str]:
    """Get list of speakers for a model"""
    speakers = ["None"]
    
    # Add VCTK speakers if VCTK model selected
    if model_name and "vctk" in model_name.lower():
        speakers.extend(sorted(VCTK_SPEAKERS.keys()))
    
    # Add cloned voices
    speakers.extend(sorted(voice_profiles.keys()))
    
    return speakers

def get_categorized_speakers() -> Dict[str, List[str]]:
    """Get speakers organized by category for better UI organization"""
    categorized = {}
    
    for category, speaker_ids in SPEAKER_CATEGORIES.items():
        categorized[category] = [(sid, VCTK_SPEAKERS[sid]) for sid in speaker_ids]
    
    # Add custom voices if any
    if voice_profiles:
        categorized["🎭 Custom Voices"] = [(name, f"Cloned voice (created {profile['created_at'][:10]})") 
                                         for name, profile in voice_profiles.items()]
    
    return categorized

def delete_voice(voice_name: str) -> str:
    """Delete a cloned voice"""
    if voice_name not in voice_profiles:
        return f"Voice '{voice_name}' not found"
    
    try:
        # Delete audio file
        audio_path = voice_profiles[voice_name]['audio_path']
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Remove from profiles
        del voice_profiles[voice_name]
        save_voice_profiles()
        
        return f"Voice '{voice_name}' deleted successfully"
        
    except Exception as e:
        return f"Error deleting voice: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="🐸 Coqui TTS Studio", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🐸 Coqui TTS Studio
        ### Advanced Text-to-Speech with Voice Cloning & VCTK 109 Voices
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Text to Speech Tab
            with gr.TabItem("🎤 Text to Speech"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="📝 Text to Synthesize",
                            placeholder="Enter your text here... Try: 'Hello, this is a demonstration of Coqui TTS with VCTK voices!'",
                            lines=4,
                            value="Hello, this is a demonstration of Coqui TTS with VCTK voices!"
                        )
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="🧠 TTS Model",
                                choices=get_model_list(),
                                value="tts_models/en/vctk/vits",
                                info="VCTK model supports 109 different voices"
                            )
                            
                            language_dropdown = gr.Dropdown(
                                label="🌍 Language",
                                choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", 
                                        "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
                                value="en"
                            )
                        
                        # VCTK Voice Selection
                        gr.Markdown("### 🎭 Choose Your Voice")
                        
                        with gr.Row():
                            voice_category = gr.Dropdown(
                                label="Voice Category",
                                choices=list(SPEAKER_CATEGORIES.keys()),
                                value="🏴󠁧󠁢󠁥󠁮󠁧󠁿 English",
                                info="Select accent/region"
                            )
                            
                            speaker_dropdown = gr.Dropdown(
                                label="Specific Voice",
                                choices=SPEAKER_CATEGORIES["🏴󠁧󠁢󠁥󠁮󠁧󠁿 English"],
                                value="p225",
                                info="Choose individual speaker"
                            )
                        
                        # Voice details display
                        voice_details = gr.HTML(
                            value=f"<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'><b>p225:</b> {VCTK_SPEAKERS['p225']}</div>"
                        )
                        
                        with gr.Row():
                            speed_slider = gr.Slider(
                                label="🎵 Speed",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1
                            )
                            
                            pitch_slider = gr.Slider(
                                label="🎼 Pitch",
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=1
                            )
                        
                        synthesize_btn = gr.Button("🎯 Generate Speech", variant="primary", size="lg")
                        
                        status_text = gr.Textbox(label="📊 Status", interactive=False)
                        audio_output = gr.Audio(label="🔊 Generated Audio", type="filepath")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 Quick Examples")
                        
                        example_btn1 = gr.Button("👋 Hello Example", size="sm")
                        example_btn2 = gr.Button("📚 Educational Text", size="sm") 
                        example_btn3 = gr.Button("🎭 Shakespeare Quote", size="sm")
                        example_btn4 = gr.Button("📰 News Report", size="sm")
                        
                        gr.Markdown("### 🧬 Quick Voice Clone")
                        quick_voice_input = gr.Audio(
                            label="Upload Voice Sample",
                            type="filepath",
                            sources=["upload", "microphone"],
                            info="Upload 10-30 seconds of clear speech"
                        )
                        quick_clone_btn = gr.Button("🎭 Clone & Use Voice", variant="secondary")
                        
                        gr.Markdown("### 🎲 Random Voice")
                        random_voice_btn = gr.Button("🎲 Try Random VCTK Voice", variant="secondary")
            
            # Voice Cloning Tab
            with gr.TabItem("🎭 Voice Cloning"):
                with gr.Row():
                    with gr.Column():
                        clone_audio_input = gr.Audio(
                            label="Upload Voice Sample (10-30 seconds recommended)",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        clone_name_input = gr.Textbox(
                            label="Voice Profile Name",
                            placeholder="Enter a name for this voice..."
                        )
                        
                        clone_btn = gr.Button("🎭 Clone Voice", variant="primary")
                        clone_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### 📚 Saved Voice Profiles")
                        
                        voice_list = gr.Dataframe(
                            headers=["Name", "Created", "Duration"],
                            label="Cloned Voices"
                        )
                        
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh")
                            delete_name = gr.Textbox(
                                placeholder="Voice name to delete",
                                scale=2
                            )
                            delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
            
            # VCTK Voice Gallery Tab  
            with gr.TabItem("🎭 VCTK Voice Gallery"):
                gr.Markdown("""
                ### 🎭 Complete VCTK 109 Voice Collection
                Explore all available voices organized by accent and region. Click any voice to hear a sample!
                """)
                
                # Create accordion for each category
                for category, speaker_ids in SPEAKER_CATEGORIES.items():
                    with gr.Accordion(f"{category} ({len(speaker_ids)} voices)", open=False):
                        # Create a dataframe for this category
                        speaker_data = []
                        for speaker_id in speaker_ids[:15]:  # Limit to first 15 to avoid overwhelming UI
                            speaker_data.append([
                                speaker_id,
                                VCTK_SPEAKERS[speaker_id].replace("🏴󠁧󠁢󠁳󠁣󠁴󠁿 ", "").replace("🏴󠁧󠁢󠁥󠁮󠁧󠁿 ", "").replace("🇮🇪 ", "").replace("🇬🇧 ", "").replace("🏴󠁧󠁢󠁷󠁬󠁳󠁿 ", "").replace("🇺🇸 ", "").replace("🇦🇺 ", ""),
                                "🎯 Select"
                            ])
                        
                        category_df = gr.Dataframe(
                            headers=["Speaker ID", "Details", "Action"],
                            value=speaker_data,
                            interactive=False,
                            wrap=True
                        )
                        
                        if len(speaker_ids) > 15:
                            gr.Markdown(f"*Showing first 15 of {len(speaker_ids)} voices. Use the main interface to access all voices.*")
            
            # Batch Processing Tab
            with gr.TabItem("📦 Batch Processing"):
                gr.Markdown("""
                ### Batch Text-to-Speech Processing
                Process multiple texts at once with the same voice settings.
                """)
                
                batch_text = gr.Textbox(
                    label="Batch Text (one per line)",
                    placeholder="Line 1: First text to synthesize\nLine 2: Second text\n...",
                    lines=10
                )
                
                with gr.Row():
                    batch_model = gr.Dropdown(
                        label="Model",
                        choices=get_model_list(),
                        value="tts_models/en/vctk/vits"
                    )
                    
                    batch_speaker = gr.Dropdown(
                        label="Speaker",
                        choices=get_speaker_list("tts_models/en/vctk/vits"),
                        value="VCTK_p225"
                    )
                
                batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                batch_status = gr.Textbox(label="Processing Status", lines=5)
                batch_files = gr.File(label="Generated Files", file_count="multiple")
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Model Information")
                
                with gr.Row():
                    device_info = gr.Textbox(
                        label="Device",
                        value=f"{'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}",
                        interactive=False
                    )
                    
                    model_info = gr.JSON(
                        label="Available Models",
                        value=available_models
                    )
                
                gr.Markdown("### API Configuration")
                api_key_input = gr.Textbox(
                    label="API Key (for external access)",
                    type="password",
                    placeholder="Enter API key..."
                )
                
                save_settings_btn = gr.Button("💾 Save Settings")
                settings_status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        def update_speaker_list(model_name):
            speakers = get_speaker_list(model_name)
            return gr.Dropdown(choices=speakers, value=speakers[0] if speakers else "None")
        
        def update_speakers_by_category(category):
            """Update speaker dropdown based on selected category"""
            if category in SPEAKER_CATEGORIES:
                speakers = SPEAKER_CATEGORIES[category]
                return gr.Dropdown(choices=speakers, value=speakers[0] if speakers else None)
            return gr.Dropdown(choices=[], value=None)
        
        def update_voice_details(speaker_id):
            """Update voice details display"""
            if speaker_id in VCTK_SPEAKERS:
                details = VCTK_SPEAKERS[speaker_id]
                return f"<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'><b>{speaker_id}:</b> {details}</div>"
            elif speaker_id in voice_profiles:
                profile = voice_profiles[speaker_id]
                return f"<div style='padding: 10px; background: #e6f3ff; border-radius: 5px;'><b>{speaker_id}:</b> Custom cloned voice (created {profile['created_at'][:10]})</div>"
            return "<div style='padding: 10px; background: #f8f8f8; border-radius: 5px;'>Select a voice to see details</div>"
        
        def get_random_voice():
            """Select a random VCTK voice"""
            import random
            random_speaker = random.choice(list(VCTK_SPEAKERS.keys()))
            # Find which category this speaker belongs to
            for category, speakers in SPEAKER_CATEGORIES.items():
                if random_speaker in speakers:
                    return category, random_speaker
            return list(SPEAKER_CATEGORIES.keys())[0], random_speaker
        
        def quick_clone_and_use(audio_file):
            if audio_file is None:
                return None, "Please upload an audio file", gr.Dropdown()
            
            voice_name = f"quick_voice_{uuid.uuid4().hex[:6]}"
            status = clone_voice(audio_file, voice_name)
            
            if "successfully" in status:
                speakers = get_speaker_list(model_dropdown.value)
                return audio_file, status, gr.Dropdown(choices=speakers, value=voice_name)
            return None, status, gr.Dropdown()
        
        def set_example_text(example_type):
            """Set example text based on type"""
            examples = {
                "hello": "Hello! Welcome to Coqui TTS with VCTK voices. This advanced text-to-speech system can generate natural-sounding speech in multiple accents and voices.",
                "educational": "The VCTK dataset contains speech from 109 speakers with various English accents. Each speaker's voice has unique characteristics that make the synthesized speech sound natural and engaging.",
                "shakespeare": "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
                "news": "Breaking news: Researchers have developed an advanced text-to-speech system that can synthesize speech using over one hundred different voices, each with distinct accents and characteristics from across the United Kingdom and beyond."
            }
            return examples.get(example_type, "")
        
        def refresh_voice_list():
            data = []
            for name, profile in voice_profiles.items():
                data.append([
                    name,
                    profile['created_at'][:19],
                    f"{profile.get('duration', 0):.1f}s"
                ])
            return data
        
        def process_batch(texts, model, speaker):
            if not texts:
                return "No text provided", None
            
            lines = texts.strip().split('\n')
            files = []
            status_lines = []
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                status_lines.append(f"Processing {i+1}/{len(lines)}: {line[:50]}...")
                
                output_path, msg = synthesize_speech(
                    line.strip(), model, speaker, "en", 1.0, 0, "Neutral", None
                )
                
                if output_path:
                    files.append(output_path)
                    status_lines[-1] += " ✓"
                else:
                    status_lines[-1] += f" ✗ ({msg})"
            
            return "\n".join(status_lines), files
        
        # Connect event handlers
        model_dropdown.change(
            update_speaker_list,
            inputs=[model_dropdown],
            outputs=[speaker_dropdown]
        )
        
        # Voice category and speaker selection
        voice_category.change(
            update_speakers_by_category,
            inputs=[voice_category],
            outputs=[speaker_dropdown]
        )
        
        speaker_dropdown.change(
            update_voice_details,
            inputs=[speaker_dropdown],
            outputs=[voice_details]
        )
        
        # Random voice selection
        random_voice_btn.click(
            get_random_voice,
            outputs=[voice_category, speaker_dropdown]
        )
        
        # Example text buttons
        example_btn1.click(
            lambda: set_example_text("hello"),
            outputs=[text_input]
        )
        
        example_btn2.click(
            lambda: set_example_text("educational"),
            outputs=[text_input]
        )
        
        example_btn3.click(
            lambda: set_example_text("shakespeare"),
            outputs=[text_input]
        )
        
        example_btn4.click(
            lambda: set_example_text("news"),
            outputs=[text_input]
        )
        
        synthesize_btn.click(
            synthesize_speech,
            inputs=[
                text_input, model_dropdown, speaker_dropdown,
                language_dropdown, speed_slider, pitch_slider,
                emotion_dropdown, quick_voice_input
            ],
            outputs=[audio_output, status_text]
        )
        
        quick_clone_btn.click(
            quick_clone_and_use,
            inputs=[quick_voice_input],
            outputs=[quick_voice_input, status_text, speaker_dropdown]
        )
        
        clone_btn.click(
            clone_voice,
            inputs=[clone_audio_input, clone_name_input],
            outputs=[clone_status]
        ).then(
            refresh_voice_list,
            outputs=[voice_list]
        )
        
        refresh_btn.click(
            refresh_voice_list,
            outputs=[voice_list]
        )
        
        delete_btn.click(
            delete_voice,
            inputs=[delete_name],
            outputs=[clone_status]
        ).then(
            refresh_voice_list,
            outputs=[voice_list]
        )
        
        batch_btn.click(
            process_batch,
            inputs=[batch_text, batch_model, batch_speaker],
            outputs=[batch_status, batch_files]
        )
        
        # Load initial data
        demo.load(refresh_voice_list, outputs=[voice_list])
    
    return demo

# Initialize on module load
logger.info("Initializing Coqui TTS Gradio Interface...")
initialize_models()
load_voice_profiles()

# Create and export the interface
interface = create_interface()

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True
    )
