"""
Interactive Alice in Wonderland Storytelling - Streamlit App
===========================================================

This Streamlit app provides a web interface for the Interactive Alice Storytelling system.
All core logic from the Jupyter notebook is preserved exactly as-is.

Author: ArenGolazizian
"""

import streamlit as st
import os
import requests
import json
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from datetime import datetime
import time
import urllib.request

# Configure Streamlit page
st.set_page_config(
    page_title="Alice in Wonderland: Interactive Storytelling",
    page_icon="🐰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Alice theme
st.markdown("""
<style>
    /* Force the main app background */
    .stApp {
        background: #6a4c93 !important;
    }
    
    .main {
        background: #6a4c93 !important;
    }
    
    /* Main content area with purple gradient - more specific selectors */
    .main .block-container,
    .stApp > .main .block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 50%, #8e44ad 100%) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        margin: 1rem auto !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
        max-width: none !important;
    }
    
    /* Also target the main content wrapper */
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stAppViewContainer"] {
        background: #6a4c93 !important;
    }
    
    /* Style the top toolbar area with Rerun/Deploy buttons */
    [data-testid="stToolbar"] {
        background: #6a4c93 !important;
        border: none !important;
        border-bottom: none !important;
    }
    
    /* Style the toolbar buttons */
    [data-testid="stToolbar"] button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stToolbar"] button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Style the toolbar menu button (three dots) */
    [data-testid="stToolbar"] [data-testid="stActionButton"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stToolbar"] [data-testid="stActionButton"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    .story-container {
        background: white !important;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 5px solid #4CAF50;
        color: #2c3e50 !important;
    }
    
    /* Ensure all text within story container is dark and readable */
    .story-container h1,
    .story-container h2,
    .story-container h3,
    .story-container p,
    .story-container li,
    .story-container strong {
        color: #2c3e50 !important;
    }
    
    .story-container strong {
        color: #1976D2 !important;
    }
    
    .choice-button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        padding: 15px 25px;
        border: none;
        border-radius: 25px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s;
        font-size: 16px;
        width: 100%;
        text-align: left;
    }
    
    .choice-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .context-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        color: #2c3e50 !important;
    }
    
    .context-box strong {
        color: #1976D2 !important;
    }
    
    .context-box em {
        color: #546e7a !important;
    }
    
    .welcome-header {
        text-align: center;
        color: white;
        font-family: 'Times New Roman', serif;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .loading-container {
        text-align: center;
        padding: 40px;
        color: #666;
    }
    
    /* Story history styling */
    .story-history-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #4CAF50;
        font-size: 0.85em;
    }
    
    .story-history-choice {
        color: #E8F5E8;
        font-weight: 500;
        margin-bottom: 5px;
    }
    
    .story-history-text {
        color: #B8E6B8;
        font-style: italic;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)

# === CONFIGURATION ===
@dataclass
class Config:
    """Configuration class - EXACTLY as in notebook"""
    OPENROUTER_API_KEY = "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"
    MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800

config = Config()

# === INITIALIZATION FUNCTIONS ===
@st.cache_data
def download_story_text(url: str, filename: str) -> str:
    """Downloads Alice in Wonderland - EXACTLY as in notebook"""
    try:
        urllib.request.urlretrieve(url, filename)
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        st.error(f"Error downloading {filename}: {e}")
        return ""

@st.cache_data
def clean_alice_text(text: str) -> str:
    """Cleans Alice text - EXACTLY as in notebook"""
    story_start = text.find("CHAPTER I.\nDown the Rabbit-Hole")
    story_end = text.find("THE END") + len("THE END")
    story_text = text[story_start:story_end]
    
    # Remove asterisk separators
    story_text = re.sub(r'\n\s*\*+\s*\*+\s*\*+.*?\*+\s*\*+\s*\*+\s*\n', '\n\n', story_text)
    story_text = re.sub(r'\n\s*\*\s*\*\s*\*.*?\*\s*\*\s*\*\s*\n', '\n\n', story_text)
    
    # Convert _italics_ to regular text
    story_text = re.sub(r'_([^_]+)_', r'\1', story_text)
    
    # Fix typography
    story_text = story_text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    story_text = re.sub(r'\r\n', '\n', story_text)
    story_text = re.sub(r'[ \t]+', ' ', story_text)
    story_text = re.sub(r'\n{3,}', '\n\n', story_text)
    
    return story_text.strip()

# === CHUNKING FUNCTIONS ===
def create_chunk(text: str, chapter: str, chunk_id: int) -> Dict:
    """Create chunk dictionary - EXACTLY as in notebook"""
    return {
        'text': text.strip(),
        'chapter': chapter,
        'chunk_id': chunk_id
    }

def split_long_paragraph(paragraph: str, target_size: int = config.CHUNK_SIZE) -> List[str]:
    """Split oversized paragraphs - EXACTLY as in notebook"""
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence) > target_size:
            parts = re.split(r'[,;:]\s+', sentence)
            for part in parts:
                if len(current_chunk + part) > target_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = part + " "
                else:
                    current_chunk += part + " "
        else:
            if len(current_chunk + sentence) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

@st.cache_data
def chunk_alice_text(text: str) -> List[Dict]:
    """Chunk Alice text - EXACTLY as in notebook"""
    chapters = re.split(r'\n\n(CHAPTER [IVXLC]+\.)', text)
    chunks = []
    chunk_counter = 0
    
    for i in range(1, len(chapters), 2):
        if i + 1 < len(chapters):
            chapter_title = chapters[i].strip()
            chapter_text = chapters[i + 1].strip()
            
            paragraphs = re.split(r'\n\s*\n', chapter_text)
            
            for para in paragraphs:
                if len(para.strip()) < 50:
                    continue
                
                if len(para) <= config.CHUNK_SIZE:
                    chunks.append(create_chunk(para, chapter_title, chunk_counter))
                    chunk_counter += 1
                else:
                    sub_chunks = split_long_paragraph(para, config.CHUNK_SIZE)
                    for sub_chunk in sub_chunks:
                        chunks.append(create_chunk(sub_chunk, chapter_title, chunk_counter))
                        chunk_counter += 1
    
    return chunks

# === KNOWLEDGE BASE CLASS ===
class AliceKnowledgeBase:
    """Alice Knowledge Base - EXACTLY as in notebook"""
    
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks - EXACTLY as in notebook"""
        embedding_model = st.session_state.embedding_model
        query_embedding = embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = self.chunks[idx].copy()
            chunk['relevance_score'] = float(score)
            chunk['rank'] = i + 1
            results.append(chunk)
        
        return results

# === LLM CLIENT ===
class OpenRouterClient:
    """OpenRouter Client - EXACTLY as in notebook"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Alice Interactive Storytelling"
        }
    
    def generate_response(self, messages: List[Dict], max_tokens: int = 800,
                         temperature: float = 0.7) -> str:
        """Generate response - EXACTLY as in notebook"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep(1)
        
        return None

# === PROMPT ENGINE ===
class StoryPromptEngine:
    """Story Prompt Engine - EXACTLY as in notebook"""
    
    def __init__(self):
        self.system_prompt = """You are an expert interactive storyteller creating adventures inspired by Alice in Wonderland.

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Write 2-3 paragraphs of story
- End with EXACTLY this separator: "CHOICES:"
- Follow with EXACTLY 3 choices in this format:
  1. [Choice text]
  2. [Choice text]
  3. [Choice text]
- NO extra text after the choices
- NO variations in formatting

EXAMPLE OUTPUT:
Alice tumbled through the rabbit hole, her blue dress billowing around her like flower petals. The walls were lined with curious objects - maps, clocks, and jars of marmalade that seemed to float in mid-air.

She landed softly on a pile of dried leaves, brushing dust from her apron. Before her stretched three curious doorways, each more peculiar than the last.

CHOICES:
1. Enter the door marked "EAT ME" with a golden key hanging beside it
2. Follow the sound of distant music coming from the red door
3. Investigate the tiny door that seems to be breathing softly

STRICT RULES:
- Use "CHOICES:" as separator (nothing else)
- Number choices 1, 2, 3 (no other format)
- End immediately after choice 3
- Keep each choice under 15 words"""
    
    def create_story_prompt(self, user_choice: str, alice_context: List[Dict],
                           previous_story: str = "") -> List[Dict]:
        """Create story prompt - EXACTLY as in notebook"""
        context_text = self._format_alice_context(alice_context)
        
        user_message = f"""ALICE IN WONDERLAND CONTEXT:
{context_text}

PREVIOUS STORY:
{previous_story}

USER CHOICE: {user_choice}

Generate the next story segment inspired by the Alice context above."""
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    def _format_alice_context(self, contexts: List[Dict]) -> str:
        """Format Alice context - EXACTLY as in notebook"""
        if not contexts:
            return "(No specific Alice context - use general Wonderland themes)"
        
        formatted_contexts = []
        for i, ctx in enumerate(contexts[:3], 1):
            chapter = ctx.get('chapter', 'Unknown Chapter')
            text = ctx.get('text', '')
            score = ctx.get('relevance_score', 0)
            
            formatted_contexts.append(f"""
Context {i} (Relevance: {score:.3f}) - {chapter}:
{text}
""")
        
        return "\n".join(formatted_contexts)

# === STORY GENERATOR ===
class AliceStoryGenerator:
    """Story Generator - EXACTLY as in notebook"""
    
    def __init__(self, knowledge_base: AliceKnowledgeBase, llm_client: OpenRouterClient):
        self.kb = knowledge_base
        self.llm = llm_client
        self.prompts = StoryPromptEngine()
        self.story_history = []
    
    def generate_story_segment(self, user_input: str) -> Dict[str, Any]:
        """Generate story segment - EXACTLY as in notebook"""
        try:
            # Get Alice contexts
            contexts = self.kb.search(user_input, top_k=3)
            
            # Get recent story history
            recent_history = self._get_recent_story_history(max_segments=2)
            
            # Create prompt
            messages = self.prompts.create_story_prompt(
                user_choice=user_input,
                alice_context=contexts,
                previous_story=recent_history
            )
            
            # Generate response
            raw_response = self.llm.generate_response(messages)
            
            if not raw_response:
                return {
                    'success': False,
                    'error': 'LLM failed to generate response'
                }
            
            # Parse response
            parsed = self._parse_story_response(raw_response)
            
            if not parsed['success']:
                return parsed
            
            # Store in history
            self.story_history.append({
                'user_input': user_input,
                'story_text': parsed['story_text'],
                'choices': parsed['choices'],
                'contexts_used': contexts,
                'timestamp': time.time()
            })
            
            return {
                'success': True,
                'story_text': parsed['story_text'],
                'choices': parsed['choices'],
                'contexts_used': contexts
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Generation failed: {str(e)}'
            }
    
    def _get_recent_story_history(self, max_segments: int = 2) -> str:
        """Get recent story history - EXACTLY as in notebook"""
        if not self.story_history:
            return ""
        
        recent_segments = self.story_history[-max_segments:]
        history_parts = []
        
        for segment in recent_segments:
            history_parts.append(f"Previous choice: {segment['user_input']}")
            history_parts.append(segment['story_text'])
        
        return "\n\n".join(history_parts)
    
    def _parse_story_response(self, response: str) -> Dict[str, Any]:
        """Parse story response - EXACTLY as in notebook"""
        try:
            if "CHOICES:" not in response:
                return {
                    'success': False,
                    'error': 'Response missing CHOICES: separator'
                }
            
            # Split at choices
            parts = response.split("CHOICES:")
            if len(parts) != 2:
                return {
                    'success': False,
                    'error': 'Invalid CHOICES: separator usage'
                }
            
            story_text = parts[0].strip()
            choices_text = parts[1].strip()
            
            # Extract numbered choices
            choice_pattern = r'^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)'
            matches = re.findall(choice_pattern, choices_text, re.MULTILINE | re.DOTALL)
            
            if len(matches) != 3:
                return {
                    'success': False,
                    'error': f'Expected 3 choices, found {len(matches)}'
                }
            
            choices = [match[1].strip() for match in matches]
            
            return {
                'success': True,
                'story_text': story_text,
                'choices': choices
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Parsing error: {str(e)}'
            }

# === STREAMLIT INITIALIZATION ===
@st.cache_resource
def initialize_embedding_model():
    """Initialize embedding model - cached for performance"""
    return SentenceTransformer(config.EMBEDDING_MODEL)

@st.cache_data
def initialize_alice_data():
    """Initialize Alice data - cached for performance"""
    # Download and process Alice text
    alice_url = "https://www.gutenberg.org/files/11/11-0.txt"
    alice_filename = "alice_in_wonderland.txt"
    
    alice_text = download_story_text(alice_url, alice_filename)
    if not alice_text:
        st.error("Failed to download Alice in Wonderland text")
        st.stop()
    
    cleaned_alice = clean_alice_text(alice_text)
    alice_chunks = chunk_alice_text(cleaned_alice)
    
    return alice_chunks

@st.cache_data
def create_embeddings(alice_chunks):
    """Create embeddings - cached for performance"""
    embedding_model = st.session_state.embedding_model
    chunk_texts = [chunk['text'] for chunk in alice_chunks]
    embeddings = embedding_model.encode(chunk_texts)
    return embeddings

# === STREAMLIT APP LAYOUT ===
def main():
    """Main Streamlit app"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'story_generator' not in st.session_state:
        st.session_state.story_generator = None
    
    if 'current_story' not in st.session_state:
        st.session_state.current_story = ""
    
    if 'current_choices' not in st.session_state:
        st.session_state.current_choices = []
    
    if 'turn_number' not in st.session_state:
        st.session_state.turn_number = 0
    
    # Header
    st.markdown("""
    <div class="welcome-header">
        <h1>🐰 Alice in Wonderland: Interactive Storytelling</h1>
        <h3>✨ A RAG-Enhanced Adventure ✨</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls and info
    with st.sidebar:
        st.header("🎮 Adventure Controls")
        
        if st.button("🔄 Reset Adventure", key="reset"):
            # Reset everything
            st.session_state.story_generator.story_history = []
            st.session_state.turn_number = 0
            st.session_state.current_story = ""
            st.session_state.current_choices = []
            st.rerun()
        
        show_contexts = st.checkbox("🔍 Show Alice Contexts", value=False)
        
        st.markdown("---")
        st.header("📚 Story History")
        
        if st.session_state.story_generator and st.session_state.story_generator.story_history:
            # Show recent story segments
            history = st.session_state.story_generator.story_history
            recent_history = history[-3:] if len(history) > 3 else history  # Show last 3 segments
            
            for i, segment in enumerate(reversed(recent_history), 1):
                choice = segment.get('user_input', 'Unknown choice')
                story_preview = segment.get('story_text', '')[:100] + "..." if len(segment.get('story_text', '')) > 100 else segment.get('story_text', '')
                
                st.markdown(f"""
                <div class="story-history-item">
                    <div class="story-history-choice">🎯 {choice}</div>
                    <div class="story-history-text">{story_preview}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show total count
            total_segments = len(history)
            if total_segments > 3:
                st.markdown(f"<div style='text-align: center; color: #B8E6B8; font-size: 0.8em; margin-top: 10px;'>...and {total_segments - 3} more segments</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align: center; color: #E8F5E8; font-weight: 500; margin-top: 15px;'>📊 Total: {total_segments} segments</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #B8E6B8; text-align: center; font-style: italic;'>Your adventure story will appear here as you make choices...</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This interactive storytelling app uses:
        - **RAG (Retrieval-Augmented Generation)**: Finds relevant Alice passages
        - **DeepSeek R1**: Advanced language model for story generation
        - **FAISS**: Vector database for similarity search
        - **Streamlit**: Web interface framework
        
        Your choices shape the story, while the original Alice text provides inspiration!
        """)
    
    # Initialize the system if not done
    if not st.session_state.initialized:
        # Show Alice-themed loading screen
        init_placeholder = st.empty()
        init_placeholder.markdown("""
        <div class="loading-container">
            <h1>� Entering Wonderland... 🌟</h1>
            <p style="font-size: 18px;">Preparing your magical adventure:</p>
            <div style="margin: 30px 0;">
                <p>🤖 Loading the Cheshire Cat's wisdom (AI models)...</p>
                <p>📚 Gathering Alice's story fragments...</p>
                <p>🔮 Enchanting the memory palace (vector database)...</p>
                <p>🎭 Rehearsing the story engine...</p>
            </div>
            <div style="font-size: 4em; margin: 20px 0;">🐰</div>
            <p><em>"Take more tea," the March Hare said earnestly, "it's very easy to take more than nothing."</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Initialize embedding model
            st.session_state.embedding_model = initialize_embedding_model()
            
            # Load Alice data
            alice_chunks = initialize_alice_data()
            
            # Create embeddings
            embeddings = create_embeddings(alice_chunks)
            
            # Create knowledge base
            alice_kb = AliceKnowledgeBase(alice_chunks, embeddings)
            
            # Create LLM client
            llm_client = OpenRouterClient(config.OPENROUTER_API_KEY, config.MODEL_NAME)
            
            # Create story generator
            st.session_state.story_generator = AliceStoryGenerator(alice_kb, llm_client)
            
            st.session_state.initialized = True
            
            # Clear loading message and show success
            init_placeholder.empty()
            st.success("✨ Welcome to Wonderland! Your adventure awaits... ✨")
            time.sleep(1)  # Brief pause for effect
            st.rerun()
            
        except Exception as e:
            init_placeholder.empty()
            st.error(f"🚨 The Queen of Hearts blocked our way: {str(e)}")
            st.stop()
    
    # Main content area
    if st.session_state.turn_number == 0:
        # Welcome screen
        st.markdown("""
        <div class="story-container">
            <h2 style="color: #2c3e50 !important;">🌟 Welcome to Your Alice Adventure!</h2>
            <p style="font-size: 18px; line-height: 1.6; color: #2c3e50 !important;">
                Dive down the rabbit hole into a world where your choices shape the story! 
                This adventure combines the classic Alice in Wonderland with cutting-edge AI 
                to create a unique, personalized journey through Wonderland.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #2c3e50 !important;">
                🎯 <strong style="color: #1976D2 !important;">How it works:</strong><br>
                • Choose your path from three options<br>
                • The AI finds relevant passages from the original Alice story<br>
                • Your personalized adventure unfolds, inspired by Carroll's masterpiece<br>
                • Every choice leads to new possibilities!
            </p>
            <p style="text-align: center; margin-top: 30px; color: #2c3e50 !important;">
                <strong style="color: #1976D2 !important;">Ready to begin your journey? Choose your starting adventure:</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Starting choices
        starting_choices = [
            "🐰 Follow the White Rabbit down the rabbit hole",
            "🏰 Explore the mysterious garden behind the small door", 
            "🎩 Join the Mad Hatter's tea party in progress"
        ]
        
        for i, choice in enumerate(starting_choices):
            if st.button(choice, key=f"start_{i}", use_container_width=True):
                handle_choice(choice)
    
    else:
        # Display current story
        if st.session_state.current_story:
            st.markdown(f"""
            <div class="story-container">
                <h3 style="color: #2c3e50 !important;">📖 Chapter {st.session_state.turn_number}</h3>
                <div style="font-size: 16px; line-height: 1.8; text-align: justify; color: #2c3e50 !important;">
                    {st.session_state.current_story.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display context if enabled
            if show_contexts and st.session_state.story_generator.story_history:
                last_segment = st.session_state.story_generator.story_history[-1]
                contexts = last_segment.get('contexts_used', [])
                
                if contexts:
                    st.markdown("### 🔍 Alice Contexts Used")
                    for i, ctx in enumerate(contexts[:3], 1):
                        score = ctx.get('relevance_score', 0)
                        chapter = ctx.get('chapter', 'Unknown')
                        text = ctx.get('text', '')[:200] + "..."
                        
                        st.markdown(f"""
                        <div class="context-box">
                            <strong style="color: #1976D2 !important;">Context {i}</strong> 
                            <span style="color: #2c3e50 !important;">(Relevance: {score:.3f}) - {chapter}</span><br>
                            <em style="color: #546e7a !important;">{text}</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display choices
            if st.session_state.current_choices:
                st.markdown("### 🎯 What do you choose next?")
                
                for i, choice in enumerate(st.session_state.current_choices):
                    if st.button(f"{i+1}. {choice}", key=f"choice_{i}", use_container_width=True):
                        handle_choice(choice)

def handle_choice(choice_text: str):
    """Handle user choice selection"""
    
    # Show loading message
    with st.spinner("🎭 Crafting your adventure..."):
        try:
            # Generate story segment
            result = st.session_state.story_generator.generate_story_segment(choice_text)
            
            if result['success']:
                # Update session state
                st.session_state.current_story = result['story_text']
                st.session_state.current_choices = result['choices']
                st.session_state.turn_number += 1
                
                # Rerun to show new content
                st.rerun()
            else:
                st.error(f"🚨 Something went wrong: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"🚨 Adventure interrupted: {str(e)}")

if __name__ == "__main__":
    main()
