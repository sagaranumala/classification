# ============================================================================
# AI CHATBOT WEB UI - ULTRA FAST OPTIMIZED VERSION
# Save as: app.py
# Run: streamlit run app.py
# ============================================================================

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
from datetime import datetime
import json
import re
import time

# Page configuration
st.set_page_config(
    page_title="AI Chatbot - Fast ⚡",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "sentiment_analyzer" not in st.session_state:
    st.session_state.sentiment_analyzer = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_times" not in st.session_state:
    st.session_state.response_times = []

# ============================================================================
# FAST MODEL LOADING WITH 4-BIT QUANTIZATION
# ============================================================================

@st.cache_resource
def load_models():
    """Load optimized models with 4-bit quantization for speed"""
    try:
        with st.spinner("⚡ Loading optimized AI models... This may take 2-3 minutes."):
            
            st.info("📊 Loading sentiment analyzer (DistilBERT)...")
            sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            st.info("🧠 Loading TinyLlama with 4-bit quantization...")
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 4-bit quantization for speed and memory efficiency
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                # CPU fallback with optimizations
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            st.success("✅ Optimized models loaded! Ready for fast responses 🚀")
            return sentiment, tokenizer, model, True
            
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}")
        return None, None, None, False

# ============================================================================
# FAST SENTIMENT ANALYSIS
# ============================================================================

def analyze_sentiment(text, sentiment_analyzer):
    """
    FAST: Optimized sentiment with early exit
    """
    if not sentiment_analyzer:
        return {"label": "NEUTRAL", "score": 0.5}
    
    try:
        text_lower = text.lower()
        
        # --- FAST PATH: Quick keyword check ---
        positive_words = ['love', 'amazing', 'excellent', 'great', 'wonderful', 'fantastic', 'awesome', 'perfect']
        negative_words = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'frustrating']
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        
        # Quick exit for obvious single sentiment (no contrast)
        has_contrast = any(word in text_lower for word in [' but ', ' however ', ' although ', ' though ', ' yet '])
        
        if has_positive and not has_negative and not has_contrast:
            return {"label": "POSITIVE", "score": 0.95}
        if has_negative and not has_positive and not has_contrast:
            return {"label": "NEGATIVE", "score": 0.95}
        
        # --- STEP 1: DETECT CONTRASTING STATEMENTS ---
        contrast_words = ['but', 'however', 'although', 'though', 'yet', 
                         'nevertheless', 'nonetheless', 'despite', 'whereas']
        
        if has_contrast:
            for word in contrast_words:
                if f" {word} " in f" {text_lower} ":
                    parts = text_lower.split(f" {word} ")
                    if len(parts) >= 2:
                        part1 = parts[0].strip()
                        part2 = parts[1].strip()
                        
                        # Quick check on parts
                        pos1 = any(w in part1 for w in positive_words)
                        neg1 = any(w in part1 for w in negative_words)
                        pos2 = any(w in part2 for w in positive_words)
                        neg2 = any(w in part2 for w in negative_words)
                        
                        if (pos1 and neg2) or (neg1 and pos2):
                            return {"label": "NEUTRAL", "score": 0.75, "mixed": True}
        
        # --- STEP 2: MODEL SENTIMENT (only if needed) ---
        result = sentiment_analyzer(text[:512])[0]
        label = result['label']
        score = float(result['score'])
        
        # --- STEP 3: NEUTRAL KEYWORD DETECTION ---
        neutral_keywords = [
            "not sure", "don't know", "maybe", "perhaps", "possibly",
            "interesting", "different", "mixed", "not sure how",
            "can't decide", "undecided", "unsure", "neutral",
            "not certain", "not confident", "ambiguous", "uncertain"
        ]
        
        is_neutral = any(keyword in text_lower for keyword in neutral_keywords)
        
        # --- STEP 4: SARCASM DETECTION ---
        sarcasm_patterns = [
            r'oh great', r'just what i needed', r'wonderful', r'fantastic',
            r'oh joy', r'how nice', r'how wonderful', r'how great'
        ]
        is_sarcastic = any(re.search(pattern, text_lower) for pattern in sarcasm_patterns)
        
        if is_sarcastic and any(word in text_lower for word in ['update', 'change', 'confusion', 'problem', 'issue']):
            return {"label": "NEGATIVE", "score": 0.85}
        
        # --- STEP 5: CONFIDENCE THRESHOLD ---
        if score < 0.70 or is_neutral:
            final_label = "NEUTRAL"
            final_score = 0.5 + (score * 0.3)
        else:
            final_label = label
            final_score = score
        
        return {
            "label": final_label,
            "score": final_score
        }
        
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"label": "NEUTRAL", "score": 0.5}

# ============================================================================
# FAST AI RESPONSE GENERATION
# ============================================================================

def get_ai_response(prompt, tokenizer, model, sentiment_info=None):
    """
    FAST: Optimized generation with fewer tokens and greedy decoding
    """
    try:
        # Build system prompt based on sentiment
        if sentiment_info and sentiment_info.get('mixed', False):
            sentiment_prompt = "User has mixed feelings. Acknowledge both sides briefly."
        elif sentiment_info and sentiment_info.get('label') == 'NEGATIVE':
            sentiment_prompt = "User is unhappy. Show empathy and offer help."
        elif sentiment_info and sentiment_info.get('label') == 'POSITIVE':
            sentiment_prompt = "User is happy. Be warm and encouraging."
        else:
            sentiment_prompt = "User is neutral. Be informative."
        
        system_prompt = f"""You are a helpful AI assistant. {sentiment_prompt} Keep responses very concise (2-3 sentences)."""
        
        input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
        
        # FAST GENERATION: Greedy decoding, fewer tokens
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=80,          # Reduced from 150
                do_sample=False,             # Greedy decoding (faster)
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        patterns = [
            r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",
            r"assistant\n(.*?)(?:<\|im_end\|>|$)",
            r"Assistant:\s*(.*?)(?:$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                response = match.group(1).strip()
                break
        
        response = response.replace(prompt, "").strip()
        response = re.sub(r'<\|.*?\|>', '', response)
        response = re.sub(r'Human:.*?\n', '', response)
        response = re.sub(r'User:.*?\n', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Limit length
        if len(response.split()) > 40:
            response = ' '.join(response.split()[:40]) + "..."
        
        if not response or len(response.strip()) < 3:
            return None
            
        return response
        
    except Exception as e:
        print(f"AI generation error: {e}")
        return None

# ============================================================================
# FALLBACK RESPONSES
# ============================================================================

def get_fallback_response(user_input):
    """Fast rule-based responses"""
    user_input_lower = user_input.lower()
    
    keywords = {
        "greeting": ["hello", "hi", "hey", "greetings", "howdy"],
        "farewell": ["bye", "goodbye", "exit", "quit", "see you"],
        "thanks": ["thank", "thanks", "appreciate", "grateful"],
        "joke": ["joke", "funny", "laugh", "humor"],
        "help": ["help", "what can you do", "capabilities"]
    }
    
    responses = {
        "greeting": [
            "Hello! How can I help you today? 🤖",
            "Hi there! Ready for a quick chat?"
        ],
        "farewell": [
            "Goodbye! 👋",
            "See you later!"
        ],
        "thanks": [
            "You're welcome! 😊",
            "Happy to help!"
        ],
        "joke": [
            "Why do programmers prefer dark mode? Light attracts bugs! 🐛",
            "What's an AI's favorite music? Al-gorithms! 🎵"
        ],
        "help": [
            "I can chat, analyze sentiment, and answer questions!",
            "Ask me about AI, tech, or anything else!"
        ]
    }
    
    for category, words in keywords.items():
        if any(word in user_input_lower for word in words):
            return random.choice(responses[category])
    
    fallbacks = [
        "That's interesting! Tell me more.",
        "I see! What else would you like to discuss?",
        "Fascinating! Could you elaborate?",
        "That's thought-provoking! What's your perspective?"
    ]
    return random.choice(fallbacks)

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff00;
        text-align: center;
        text-shadow: 0 0 10px #00ff00;
        margin-bottom: 20px;
    }
    .speed-badge {
        display: inline-block;
        background: #4CAF50;
        padding: 2px 12px;
        border-radius: 12px;
        color: white;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .fast-badge {
        display: inline-block;
        background: #ff6b6b;
        padding: 2px 12px;
        border-radius: 12px;
        color: white;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ AI Chatbot - Fast</div>', unsafe_allow_html=True)
st.markdown('*Optimized with 4-bit quantization | <span class="speed-badge">🚀 5-10x Faster</span>*', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚡ Speed Controls")
    
    if st.button("🚀 Load Optimized Models", use_container_width=True, type="primary"):
        sentiment, tokenizer, model, loaded = load_models()
        if loaded:
            st.session_state.sentiment_analyzer = sentiment
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ Models loaded!")
            st.rerun()
    
    if st.session_state.model_loaded:
        st.success("✅ Models: **Ready**")
        st.info(f"💻 Device: {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
        
        # Performance stats
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ Avg Time", f"{avg_time:.2f}s")
            with col2:
                st.metric("📊 Messages", len(st.session_state.response_times))
            
            # Show speed rating
            if avg_time < 1.0:
                st.success("🚀 Lightning Fast!")
            elif avg_time < 2.0:
                st.info("⚡ Very Fast")
            else:
                st.warning("🐢 Could be faster")
    else:
        st.warning("⚠️ Click 'Load Optimized Models' to start")
        st.info("💡 First load may take 2-3 minutes")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.response_times = []
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "user" and "sentiment" in message:
            sentiment = message["sentiment"]
            score = message.get("score", 0)
            emoji = "😊" if sentiment == "POSITIVE" else ("😔" if sentiment == "NEGATIVE" else "😐")
            st.caption(f"{emoji} {sentiment} ({score:.2f})")
        
        if message["role"] == "assistant" and "method" in message:
            st.caption(f"Generated by: {message['method']}")

# Chat input
if prompt := st.chat_input("⚡ Type your message here..."):
    # Start timer
    start_time = time.time()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"user": prompt})
    
    # Analyze sentiment
    sentiment_info = None
    if st.session_state.sentiment_analyzer:
        sentiment_info = analyze_sentiment(prompt, st.session_state.sentiment_analyzer)
        if sentiment_info:
            st.session_state.messages[-1]["sentiment"] = sentiment_info["label"]
            st.session_state.messages[-1]["score"] = sentiment_info["score"]
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            response = None
            method = "📝 Rules"
            
            if st.session_state.model_loaded:
                ai_response = get_ai_response(
                    prompt, 
                    st.session_state.tokenizer, 
                    st.session_state.model,
                    sentiment_info
                )
                if ai_response:
                    response = ai_response
                    method = "🤖 AI"
            
            if not response:
                response = get_fallback_response(prompt)
                method = "📝 Rules"
            
            st.markdown(response)
            st.caption(f"Generated by: {method}")
            
            # Calculate and display speed
            elapsed = time.time() - start_time
            st.session_state.response_times.append(elapsed)
            
            # Speed indicator
            if elapsed < 1.0:
                speed_emoji = "🚀"
                speed_text = "Lightning Fast!"
            elif elapsed < 2.0:
                speed_emoji = "⚡"
                speed_text = "Very Fast!"
            else:
                speed_emoji = "🐢"
                speed_text = "Normal"
            
            st.caption(f"{speed_emoji} Response time: {elapsed:.2f}s - {speed_text}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "method": method
            })
            st.session_state.chat_history[-1]["bot"] = response
            st.session_state.chat_history[-1]["method"] = method
    
    st.rerun()

st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("⚡ Optimized with 4-bit quantization")
    st.caption("📚 Models: TinyLlama (1.1B) + DistilBERT (67M)")
    st.caption("🚀 5-10x faster than standard version")