# 🏥 Healthcare Multimodal Chatbot

A sophisticated AI-powered healthcare assistant that combines text and voice interactions using advanced language models and speech synthesis technologies.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA3--70B-green.svg)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-TTS-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 🤖 **Advanced AI Capabilities**
- **Multiple Language Models**: Powered by Groq Cloud API with LLaMA 3 (70B parameter model)
- **Multimodal Input**: Supports both text and voice input processing
- **Voice Synthesis**: High-quality speech generation using ElevenLabs API
- **Multilingual Support**: 8+ languages including English, Spanish, French, German, Hindi, Arabic, Portuguese, and Russian

### 🎯 **Healthcare-Specific Features**
- **Medical Context Awareness**: Specialized prompting for healthcare scenarios
- **Patient Information Integration**: Age, gender, and medical history consideration
- **HIPAA Compliance**: Privacy-focused design with no permanent data storage
- **Safety-First Approach**: Always recommends professional medical consultation when appropriate

### 🛡️ **Reliability & Performance**
- **Automatic Retry Logic**: Handles API failures with exponential backoff
- **Error Recovery**: Intelligent handling of 503 Service Unavailable and other API errors
- **Request Timeout Management**: Prevents hanging requests
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

### 🎨 **User Experience**
- **Modern Web Interface**: Built with Gradio for intuitive interaction
- **Responsive Design**: Works across different devices and screen sizes
- **Multiple Output Formats**: Medical advice, concise summaries, step-by-step instructions, technical explanations
- **Real-time Processing**: Fast response times with progress indicators

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized deployment)
- **API Keys**:
  - [Groq API Key](https://console.groq.com/) (Required)
  - [ElevenLabs API Key](https://elevenlabs.io/) (Optional, for voice features)

### 🐳 Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "health chatbot"
   ```

2. **Run with Docker**
   ```bash
   # Windows
   .\run_healthcare_chatbot.bat
   
   # Linux/Mac
   chmod +x run_healthcare_chatbot.sh
   ./run_healthcare_chatbot.sh
   ```

3. **Access the application**
   - Open your browser and navigate to `http://localhost:7860`

### 🐍 Local Python Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (optional)
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   echo "ELEVENLABS_API_KEY=your_elevenlabs_api_key_here" >> .env
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

## 📖 Usage Guide

### 1️⃣ **API Setup**
- Navigate to the "API Setup" tab
- Enter your Groq API key (required for text processing)
- Enter your ElevenLabs API key (optional, for voice features)
- Click "Save API Keys"

### 2️⃣ **Configuration**
- Go to the "Configuration" tab
- Select your preferred:
  - **Language Model**: Choose from available models
  - **Voice**: Pick from 10+ voice options
  - **Language**: Select response language
  - **Output Format**: Choose response structure
- Click "Update Configuration"

### 3️⃣ **Healthcare Chat**
- Use the "Healthcare Chat" tab for interactions
- **Input Options**:
  - Type your health concern in the text box
  - Upload audio files for voice input
  - Provide additional patient information (age, gender, medical history)
- Click "Submit" to get AI-powered healthcare guidance

## 🔧 Technical Architecture

### **Core Components**

```
├── healthcare_chatbot.py    # Main application logic
├── app.py                  # Gradio interface setup
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── run_healthcare_chatbot.bat  # Windows launcher
└── temp_uploads/          # Temporary file storage
```

### **API Integration**

- **Groq Cloud API**: Powers the language model inference
- **ElevenLabs API**: Handles speech-to-text and text-to-speech conversion
- **Request Management**: Automatic retry with exponential backoff for reliability

### **Error Handling Strategy**

| Error Type | Status Code | Action |
|------------|-------------|--------|
| Server Errors | 500, 502, 503, 504 | Automatic retry with backoff |
| Rate Limiting | 429 | Extended delay retry |
| Authentication | 401, 403 | Immediate user feedback |
| Bad Request | 400 | Input validation feedback |
| Timeout | - | Retry with exponential backoff |

## 🛠️ Configuration Options

### **Available Models**
- `llama3-70b-8192`: High-performance LLaMA 3 model with 70B parameters

### **Voice Options**
- **Female Voices**: Rachel, Bella, Elli, Glinda
- **Male Voices**: Domi, Antoni, Josh, Arnold, Adam, Sam

### **Supported Languages**
- English (en), Spanish (es), French (fr), German (de)
- Hindi (hi), Arabic (ar), Portuguese (pt), Russian (ru)

### **Output Formats**
- **Medical Advice**: General healthcare guidance
- **Concise Summary**: Brief overview of the situation
- **Step-by-Step Instructions**: Detailed procedural guidance
- **Technical Explanation**: More detailed medical information

## 📊 System Requirements

### **Minimum Requirements**
- **RAM**: 2GB available memory
- **Storage**: 500MB free space
- **Network**: Stable internet connection for API calls
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

### **Recommended Requirements**
- **RAM**: 4GB+ available memory
- **Storage**: 1GB+ free space
- **Network**: High-speed internet for optimal performance

## 🔒 Security & Privacy

### **Data Protection**
- ✅ No permanent storage of user conversations
- ✅ Temporary files automatically cleaned up
- ✅ API keys stored securely in environment variables
- ✅ HIPAA-compliant design principles

### **Best Practices**
- Use environment variables for API keys
- Run in isolated Docker containers
- Regular security updates
- Audit logs for monitoring

## 🚨 Important Disclaimers

> **⚠️ MEDICAL DISCLAIMER**
> 
> This chatbot is for **informational purposes only** and is **NOT a substitute for professional medical advice, diagnosis, or treatment**. 
> 
> - Always consult qualified healthcare professionals for medical concerns
> - In case of emergency, contact emergency services immediately
> - This tool should not be used for critical medical decisions

## 🐛 Troubleshooting

### **Common Issues**

| Issue | Symptom | Solution |
|-------|---------|----------|
| API Key Error | "Authentication error" message | Verify API keys in setup tab |
| 503 Service Error | "Service Unavailable" | Wait and retry - automatic retry enabled |
| No Audio Output | Missing voice response | Check ElevenLabs API key and voice selection |
| Configuration Error | "Please configure all settings" | Complete all fields in Configuration tab |

### **Docker Issues**
```bash
# Check if Docker is running
docker --version

# View container logs
docker logs healthcare-chatbot

# Restart container
docker restart healthcare-chatbot
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast and reliable LLaMA model inference
- **ElevenLabs** for advanced text-to-speech capabilities
- **Gradio** for the intuitive web interface framework
- **Open Source Community** for various Python packages used

## 📞 Support

For questions, issues, or feature requests:

- 📧 Email: [vishwanathnitinraj796@gmail.com]

---

<div align="center">
  
**Built with ❤️ for better healthcare accessibility**

[⭐ Star this project](https://github.com/Healthcare-chatbot-CALMLY) | [🐛 Report Bug](https://github.com/Healthcare-chatbot-CALMLY/issues) | [💡 Request Feature](https://github.com/Healthcare-chatbot-CALMLY/issues)

</div>
 
