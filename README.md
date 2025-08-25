# Alice in Wonderland: Interactive Storytelling 🐰

A sophisticated RAG-enhanced interactive storytelling application that creates personalized Alice in Wonderland adventures. Your choices shape the story while the original Lewis Carroll text provides authentic inspiration, all presented through a beautiful Alice-themed user interface.

## ✨ Features

- **Interactive Storytelling**: Your choices determine the adventure path through an intuitive web interface
- **RAG Technology**: Retrieves relevant passages from the original Alice in Wonderland text
- **AI-Powered**: Uses DeepSeek R1 model for intelligent story generation
- **Beautiful Alice Theme**: Custom purple gradient interface with elegant styling
- **Story History**: Track your adventure journey in the sidebar with visual story progression
- **Context Visualization**: Optional display of which Alice passages inspired each story segment
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Authentic Alice Content**: All stories inspired by Carroll's original text

## 🚀 Quick Start

#### Prerequisites
- Python 3.8 or higher
- Internet connection (for downloading Alice text and API calls)

#### Installation

1. **Clone or download this project**
   ```bash
   git clone https://github.com/yourusername/alice-storytelling.git
   cd alice-storytelling
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** to the URL shown (usually `http://localhost:8501`)

## 🎮 How to Use

1. **Start Your Adventure**: Choose from three starting scenarios on the welcome screen
2. **Make Choices**: At each turn, select from three AI-generated options using the styled buttons
3. **Watch Your Story Unfold**: The AI creates unique narrative based on your choices, displayed in elegant white containers
4. **Track Your Journey**: View your story history in the left sidebar, showing recent choices and story previews
5. **Explore Contexts**: Toggle "Show Alice Contexts" to see which original passages inspired each turn
6. **Reset Anytime**: Use the "Reset Adventure" button to start over with a fresh story


## 🔧 Technical Architecture

### Core Components
- **Knowledge Base**: FAISS vector database storing Alice text chunks
- **Embedding Model**: SentenceTransformers for semantic similarity
- **LLM Integration**: OpenRouter API with DeepSeek R1 model
- **Prompt Engineering**: Carefully crafted prompts for consistent output
- **Web Interface**: Streamlit for interactive user experience

### Data Flow
1. User makes a choice
2. System searches Alice knowledge base for relevant passages
3. AI generates story continuation using retrieved context
4. Story and new choices are presented to user
5. Process repeats, building narrative history

## 📊 System Requirements

- **Memory**: ~2GB RAM (for embedding model and Alice text)
- **Storage**: ~500MB (for cached models and data)
- **Network**: Required for initial setup and API calls

## 📝 Application Architecture

The Streamlit app (`streamlit_app.py`) contains the complete, production-ready implementation:

- `Config` class: Configuration management with environment variables
- `AliceKnowledgeBase`: Vector search and retrieval using FAISS
- `OpenRouterClient`: LLM API integration with DeepSeek R1
- `StoryPromptEngine`: Advanced prompt engineering for story generation
- `AliceStoryGenerator`: Complete story generation and choice logic
- **Web Interface**: Beautiful Alice-themed UI with story history and responsive design

The included Jupyter notebook (`Interactive_Storytelling_final.ipynb`) serves as development documentation, showing the step-by-step implementation process.

## 📄 License

This project uses the public domain Alice in Wonderland text from Project Gutenberg.

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest new features
- Improve the documentation
- Enhance the user interface

---

**Enjoy your journey through Wonderland!** 🎩✨
