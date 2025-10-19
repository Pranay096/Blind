"""
Reasoning Agent with Gemini/Claude/GPT support
"""

import os
import google.generativeai as genai

class ReasoningAgent:
    def __init__(self, api_provider='gemini', api_key=None):
        """
        Initialize reasoning agent with API support.
        
        Args:
            api_provider: 'gemini', 'claude', or 'gpt'
            api_key: API key (if None, reads from environment or uses hardcoded)
        """
        print("🧠 Initializing Reasoning Agent...")
        
        self.api_provider = api_provider.lower()
        
        if self.api_provider == 'gemini':
            self._init_gemini(api_key)
        elif self.api_provider == 'claude':
            self._init_claude(api_key)
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        print("✅ Reasoning Agent ready!")
    
    def _init_gemini(self, api_key=None, model_name='gemini-1.5-flash'):
        """Initialize Gemini API."""
        print("  🌟 Configuring Gemini API...")
        
        # Priority: 1. Passed key, 2. Environment variable, 3. Hardcoded
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        # Hardcoded fallback (your API key)
        if api_key is None:
            api_key = 'AIzaSyCgGplitPcYnbzxgRFGwFPcf_H6uL2xw28'
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Options:\n"
                "1. Set environment variable: export GEMINI_API_KEY='your-key'\n"
                "2. Pass api_key parameter\n"
                "3. Hardcode in reasoning_agent.py"
            )
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.api_key = api_key
            print(f"  ✓ Gemini API configured ({model_name})")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")
    
    def _init_claude(self, api_key=None):
        """Initialize Claude API."""
        print("  🤖 Configuring Claude API...")
        
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            print("  ✓ Claude API configured")
        except ImportError:
            raise ImportError("Install: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Claude: {e}")
    
    def generate_narration(self, vision_data):
        """
        Generate narration from vision data.
        
        Args:
            vision_data: Dict with objects, activities, text, etc.
            
        Returns:
            Natural language narration string
        """
        if self.api_provider == 'gemini':
            return self._generate_gemini(vision_data)
        elif self.api_provider == 'claude':
            return self._generate_claude(vision_data)
        else:
            return "Error: Unsupported API provider"
    
    def _generate_gemini(self, vision_data):
        """Generate narration using Gemini."""
        try:
            prompt = self._build_prompt(vision_data)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )
            
            narration = response.text.strip()
            return narration if narration else "Your surroundings appear clear."
            
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            return self._fallback_narration(vision_data)
    
    def _generate_claude(self, vision_data):
        """Generate narration using Claude."""
        try:
            prompt = self._build_prompt(vision_data)
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            narration = response.content[0].text.strip()
            return narration if narration else "Your surroundings appear clear."
            
        except Exception as e:
            print(f"⚠️ Claude error: {e}")
            return self._fallback_narration(vision_data)
    
    def _build_prompt(self, vision_data):
        """Build structured prompt for AI."""
        objects = vision_data.get('objects', [])
        activities = vision_data.get('activities', [])
        text_detected = vision_data.get('text', '')
        scene_desc = vision_data.get('scene_description', '')
        
        prompt = """You are an AI assistant helping a visually impaired person. 
Generate a brief, natural spoken description (1-2 sentences max) based on:

"""
        
        # Add detected text
        if text_detected:
            prompt += f"VISIBLE TEXT: {text_detected}\n"
        
        # Add human activities
        if activities:
            prompt += f"ACTIVITIES: {', '.join(activities[:3])}\n"
        
        # Add objects with distances
        if objects:
            close = [o for o in objects if o.get('distance', 10) < 2.0]
            far = [o for o in objects if o.get('distance', 10) >= 2.0]
            
            if close:
                prompt += f"NEARBY (<2m): "
                for obj in close[:3]:
                    prompt += f"{obj['label']} at {obj.get('distance', '?')}m, "
                prompt = prompt.rstrip(', ') + "\n"
            
            if far:
                prompt += f"OTHER OBJECTS: "
                for obj in far[:3]:
                    prompt += f"{obj['label']}, "
                prompt = prompt.rstrip(', ') + "\n"
        
        # Add scene
        if scene_desc:
            prompt += f"SCENE: {scene_desc}\n"
        
        prompt += """
Rules:
- Speak naturally and conversationally
- Prioritize safety (mention close objects first)
- Be concise (max 2 sentences)
- Don't repeat "there is/are" too much
- Make it sound like a helpful guide

Generate narration:"""
        
        return prompt
    
    def _fallback_narration(self, vision_data):
        """Simple rule-based fallback if API fails."""
        objects = vision_data.get('objects', [])
        activities = vision_data.get('activities', [])
        text_detected = vision_data.get('text', '')
        
        parts = []
        
        # Text first
        if text_detected:
            parts.append(f"I can see text: {text_detected}")
        
        # Activities
        if activities:
            parts.append(f"Someone is {activities[0]}")
        
        # Close objects
        close = [o for o in objects if o.get('distance', 10) < 2.0]
        if close:
            obj = close[0]
            parts.append(f"A {obj['label']} is {obj.get('distance', '?')} meters away")
        
        # Other objects
        if not close and objects:
            obj = objects[0]
            parts.append(f"I see a {obj['label']}")
        
        if parts:
            return '. '.join(parts[:2]) + '.'
        else:
            return "Your surroundings appear clear."