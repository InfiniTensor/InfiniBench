#!/usr/bin/env python3
"""
Prompt Data Definitions
Preset templates and topics for prompt generation
Separated from generation logic for better maintainability
"""

# ==================== Preset Prompt Template Library ====================
PRESET_TEMPLATES = {
    "ai_qa": [
        "Explain the concept of {topic} in simple terms.",
        "What are the main applications of {topic} in today's world?",
        "Describe the history and development of {topic}.",
        "Compare and contrast {topic} with similar technologies.",
        "What are the ethical considerations surrounding {topic}?",
        "How does {topic} impact our daily lives?",
        "What are the future trends in {topic}?",
        "What are the key challenges in {topic} research?",
    ],
    "general_qa": [
        "Tell me about {topic}.",
        "What is {topic}?",
        "Can you explain {topic}?",
        "I need information about {topic}.",
        "Please provide details about {topic}.",
        "Help me understand {topic}.",
    ],
    "technical": [
        "Discuss the technical implementation of {topic}.",
        "What are the algorithms used in {topic}?",
        "Explain the architecture of {topic} systems.",
        "What are the performance considerations for {topic}?",
        "Describe the scalability challenges in {topic}.",
    ],
}

# ==================== Preset Topic Library ====================
PRESET_TOPICS = {
    "ai_ml": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "transformers",
        "large language models",
        "generative AI",
    ],
    "tech": [
        "cloud computing",
        "blockchain technology",
        "quantum computing",
        "Internet of Things",
        "edge computing",
        "distributed systems",
        "cybersecurity",
        "databases",
        "software engineering",
    ],
    "science": [
        "climate change",
        "genetic engineering",
        "space exploration",
        "renewable energy",
        "quantum physics",
        "biotechnology",
        "nanotechnology",
    ],
}

# ==================== Default Configuration ====================
DEFAULT_TEMPLATE_NAME = "ai_qa"
DEFAULT_TOPIC_NAME = "ai_ml"
DEFAULT_CHARS_PER_TOKEN = 4


# ==================== Helper Functions ====================
def get_template_names() -> list:
    """Get all available template names"""
    return list(PRESET_TEMPLATES.keys())


def get_topic_names() -> list:
    """Get all available topic names"""
    return list(PRESET_TOPICS.keys())


def get_template(template_name: str, fallback_name: str = "ai_qa") -> list:
    """Get the template list for the specified name"""
    return PRESET_TEMPLATES.get(template_name, PRESET_TEMPLATES.get(fallback_name, []))


def get_topics(topic_name: str, fallback_name: str = "ai_ml") -> list:
    """Get the topic list for the specified name"""
    return PRESET_TOPICS.get(topic_name, PRESET_TOPICS.get(fallback_name, []))
