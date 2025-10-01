import azure.functions as func
import logging
import json
import os
import importlib
import importlib.util
import inspect
import sys
import re
from agents.basic_agent import BasicAgent
import uuid
from openai import AzureOpenAI
from datetime import datetime
import time
from utils.azure_file_storage import AzureFileStorageManager, safe_json_loads
from utils.reflection_engine import ReflectionOrchestrator

# Default GUID to use when no specific user GUID is provided
DEFAULT_USER_GUID = "c0p110t0-aaaa-bbbb-cccc-123456789abc"

def ensure_string_content(message):
    """
    Ensures message content is converted to a string regardless of input type.
    Handles all edge cases including None, undefined, or missing content.
    """
    if message is None:
        return {"role": "user", "content": ""}
        
    if not isinstance(message, dict):
        return {"role": "user", "content": str(message) if message is not None else ""}
    
    message = message.copy()
    
    if 'role' not in message:
        message['role'] = 'user'
    
    if 'content' in message:
        content = message['content']
        message['content'] = str(content) if content is not None else ''
    else:
        message['content'] = ''
    
    return message

def ensure_string_function_args(function_call):
    """
    Ensures function call arguments are properly stringified.
    Handles None and edge cases.
    """
    if not function_call:
        return None
    
    if not hasattr(function_call, 'arguments'):
        return None
        
    if function_call.arguments is None:
        return None
        
    if isinstance(function_call.arguments, (dict, list)):
        return json.dumps(function_call.arguments)
    
    return str(function_call.arguments)

def build_cors_response(origin):
    """
    Builds CORS response headers.
    Safely handles None origin.
    """
    return {
        "Access-Control-Allow-Origin": str(origin) if origin else "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Max-Age": "86400",
    }

def load_agents_from_folder():
    agents_directory = os.path.join(os.path.dirname(__file__), "agents")
    files_in_agents_directory = os.listdir(agents_directory)
    agent_files = [f for f in files_in_agents_directory if f.endswith(".py") and f not in ["__init__.py", "basic_agent.py"]]

    declared_agents = {}
    for file in agent_files:
        try:
            module_name = file[:-3]
            module = importlib.import_module(f'agents.{module_name}')
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasicAgent) and obj is not BasicAgent:
                    agent_instance = obj()
                    declared_agents[agent_instance.name] = agent_instance
        except Exception as e:
            logging.error(f"Error loading agent {file}: {str(e)}")
            continue

    storage_manager = AzureFileStorageManager()
    try:
        agent_files = storage_manager.list_files('agents')
        
        for file in agent_files:
            if not file.name.endswith('_agent.py'):
                continue

            try:
                file_content = storage_manager.read_file('agents', file.name)
                if file_content is None:
                    continue

                temp_dir = "/tmp/agents"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = f"{temp_dir}/{file.name}"

                with open(temp_file, 'w') as f:
                    f.write(file_content)

                if temp_dir not in sys.path:
                    sys.path.append(temp_dir)

                module_name = file.name[:-3]
                spec = importlib.util.spec_from_file_location(module_name, temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BasicAgent) and
                        obj is not BasicAgent):
                        agent_instance = obj()
                        declared_agents[agent_instance.name] = agent_instance

                os.remove(temp_file)

            except Exception as e:
                logging.error(f"Error loading agent {file.name} from Azure File Share: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error loading agents from Azure File Share: {str(e)}")

    # Load multi-agents from multi_agents folder
    try:
        multi_agent_files = storage_manager.list_files('multi_agents')
        
        for file in multi_agent_files:
            if not file.name.endswith('_agent.py'):
                continue

            try:
                file_content = storage_manager.read_file('multi_agents', file.name)
                if file_content is None:
                    continue

                temp_dir = "/tmp/multi_agents"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = f"{temp_dir}/{file.name}"

                with open(temp_file, 'w') as f:
                    f.write(file_content)

                if temp_dir not in sys.path:
                    sys.path.append(temp_dir)

                parent_dir = "/tmp"
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

                module_name = file.name[:-3]
                spec = importlib.util.spec_from_file_location(f"multi_agents.{module_name}", temp_file)
                module = importlib.util.module_from_spec(spec)
                
                import types
                if 'multi_agents' not in sys.modules:
                    multi_agents_module = types.ModuleType('multi_agents')
                    sys.modules['multi_agents'] = multi_agents_module
                
                sys.modules[f"multi_agents.{module_name}"] = module
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BasicAgent) and
                        obj is not BasicAgent):
                        agent_instance = obj()
                        declared_agents[agent_instance.name] = agent_instance
                        logging.info(f"Loaded multi-agent: {agent_instance.name}")

                os.remove(temp_file)

            except Exception as e:
                logging.error(f"Error loading multi-agent {file.name} from Azure File Share: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error loading multi-agents from Azure File Share: {str(e)}")

    return declared_agents

class Assistant:
    def __init__(self, declared_agents):
        self.config = {
            'assistant_name': str(os.environ.get('ASSISTANT_NAME', 'BusinessInsightBot')),
            'characteristic_description': str(os.environ.get('CHARACTERISTIC_DESCRIPTION', 'helpful business assistant with deep reflection capabilities'))
        }

        # Azure OpenAI initialization
        try:
            api_key = os.environ.get('AZURE_OPENAI_API_KEY')
            endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
            api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-01')
            
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API key and endpoint are required")
            
            logging.info(f"Initializing Azure OpenAI with endpoint: {endpoint}, version: {api_version}")
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
        except Exception as e:
            logging.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

        self.known_agents = self.reload_agents(declared_agents)
        self.user_guid = DEFAULT_USER_GUID
        self.shared_memory = None
        self.user_memory = None
        self.storage_manager = AzureFileStorageManager()
        
        # Initialize Reflection Orchestrator
        try:
            self.reflection_orchestrator = ReflectionOrchestrator()
        except Exception as e:
            logging.warning(f"Reflection engine not available: {str(e)}")
            self.reflection_orchestrator = None
        
        self._initialize_context_memory(DEFAULT_USER_GUID)

    def _check_first_message_for_guid(self, conversation_history):
        """Check if the first message contains only a GUID"""
        if not conversation_history or len(conversation_history) == 0:
            return None
            
        first_message = conversation_history[0]
        if first_message.get('role') == 'user':
            content = first_message.get('content')
            if content is None:
                return None
            content = str(content).strip()
            guid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
            if guid_pattern.match(content):
                return content
        return None

    def _initialize_context_memory(self, user_guid=None):
        """Initialize context memory with separate shared and user-specific memories"""
        try:
            context_memory_agent = self.known_agents.get('ContextMemory')
            if not context_memory_agent:
                self.shared_memory = "No shared context memory available."
                self.user_memory = "No specific context memory available."
                return

            try:
                self.storage_manager.set_memory_context(None)
                shared_result = context_memory_agent.perform(full_recall=True)
                self.shared_memory = str(shared_result)[:5000] if shared_result else "No shared context memory available."
            except Exception as e:
                logging.warning(f"Error getting shared memory: {str(e)}")
                self.shared_memory = "Context memory initialization failed."
            
            if not user_guid:
                user_guid = DEFAULT_USER_GUID
            
            try:
                self.storage_manager.set_memory_context(user_guid)
                user_result = context_memory_agent.perform(user_guid=user_guid, full_recall=True)
                self.user_memory = str(user_result)[:5000] if user_result else "No specific context memory available."
            except Exception as e:
                logging.warning(f"Error getting user memory: {str(e)}")
                self.user_memory = "Context memory initialization failed."
                
        except Exception as e:
            logging.warning(f"Error initializing context memory: {str(e)}")
            self.shared_memory = "Context memory initialization failed."
            self.user_memory = "Context memory initialization failed."
    
    def _generate_reflective_context(self, user_guid):
        """Generate reflective context using ReflectionOrchestrator if available"""
        if not self.reflection_orchestrator:
            return ""
        
        try:
            self.storage_manager.set_memory_context(user_guid)
            user_memories = self.storage_manager.read_json()
            
            self.storage_manager.set_memory_context(None)
            shared_memories = self.storage_manager.read_json()
            
            reflections = self.reflection_orchestrator.generate_comprehensive_reflection(
                user_memories, 
                shared_memories
            )
            
            reflection_summary = self.reflection_orchestrator.synthesize_reflection_summary(reflections)
            
            return reflection_summary
        except Exception as e:
            logging.warning(f"Error generating reflective context: {str(e)}")
            return ""
    
    def extract_user_guid(self, text):
        """Try to extract a GUID from user input, but only if it's the entire message"""
        if text is None:
            return None
            
        text_str = str(text).strip()
        
        guid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        match = guid_pattern.match(text_str)
        if match:
            return match.group(0)
        
        labeled_guid_pattern = re.compile(r'^guid[:=\s]+([0-9a-f-]{36})$', re.IGNORECASE)
        match = labeled_guid_pattern.match(text_str)
        if match:
            return match.group(1)
                
        return None

    def get_agent_metadata(self):
        agents_metadata = []
        for agent in self.known_agents.values():
            if hasattr(agent, 'metadata'):
                agents_metadata.append(agent.metadata)
        return agents_metadata

    def reload_agents(self, agent_objects):
        known_agents = {}
        if isinstance(agent_objects, dict):
            for agent_name, agent in agent_objects.items():
                if hasattr(agent, 'name'):
                    known_agents[agent.name] = agent
                else:
                    known_agents[str(agent_name)] = agent
        elif isinstance(agent_objects, list):
            for agent in agent_objects:
                if hasattr(agent, 'name'):
                    known_agents[agent.name] = agent
        else:
            logging.warning(f"Unexpected agent_objects type: {type(agent_objects)}")
        return known_agents

    def prepare_messages(self, conversation_history):
        if not isinstance(conversation_history, list):
            conversation_history = []
            
        messages = []
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        
        # Generate reflective context if available
        reflective_context = ""
        if self.reflection_orchestrator:
            reflective_context = self._generate_reflective_context(self.user_guid)
        
        # System message with optional reflection context
        system_content = f"""
<identity>
You are a Microsoft Copilot assistant named {str(self.config.get('assistant_name', 'Assistant'))}, operating within Microsoft Teams.
</identity>

<shared_memory_output>
These are memories accessible by all users of the system:
{str(self.shared_memory)}
</shared_memory_output>

<specific_memory_output>
These are memories specific to the current conversation:
{str(self.user_memory)}
</specific_memory_output>"""

        if reflective_context:
            system_content += f"""

<reflective_context>
Deep reflection analysis:
{reflective_context}
</reflective_context>"""

        system_content += """

<context_instructions>
- <shared_memory_output> represents common knowledge shared across all conversations
- <specific_memory_output> represents specific context for the current conversation
- Apply specific context with higher precedence than shared context
- Synthesize information from both contexts for comprehensive responses
- If available, use <reflective_context> for deeper insights
</context_instructions>

<agent_usage>
IMPORTANT: You must be honest and accurate about agent usage:
- NEVER pretend or imply you've executed an agent when you haven't actually called it
- NEVER say "using my agent" unless you are actually making a function call to that agent
- NEVER fabricate success messages about data operations that haven't occurred
- If you need to perform an action and don't have the necessary agent, say so directly
- When a user requests an action, either:
  1. Call the appropriate agent and report actual results, or
  2. Say "I don't have the capability to do that" and suggest an alternative
  3. If no details are provided besides the request to run an agent, infer the necessary input parameters by "reading between the lines" of the conversation context so far
</agent_usage>

<response_format>
CRITICAL: You must structure your response in TWO distinct parts separated by the delimiter |||VOICE|||

1. FIRST PART (before |||VOICE|||): Your full formatted response
   - Use **bold** for emphasis
   - Use `code blocks` for technical content
   - Apply --- for horizontal rules to separate sections
   - Utilize > for important quotes or callouts
   - Format code with ```language syntax highlighting
   - Create numbered lists with proper indentation
   - Add personality when appropriate
   - Apply # ## ### headings for clear structure

2. SECOND PART (after |||VOICE|||): A concise voice response
   - Maximum 1-2 sentences
   - Pure conversational English with NO formatting
   - Extract only the most critical information
   - Sound like a colleague speaking casually over a cubicle wall
   - Be natural and conversational, not robotic
   - Focus on the key takeaway or action item

EXAMPLE FORMAT:
Here's the detailed analysis you requested:

**Key Findings:**
- Revenue increased by 12%
- Customer satisfaction scores improved

|||VOICE|||
Revenue's up 12 percent and customers are happier - looking good for Q3.
</response_format>
"""
        
        system_message = {"role": "system", "content": system_content}
        messages.append(ensure_string_content(system_message))
        
        # Process conversation history - skip first message if it's just a GUID
        guid_only_first_message = self._check_first_message_for_guid(conversation_history)
        start_idx = 1 if guid_only_first_message else 0
        
        for i in range(start_idx, len(conversation_history)):
            messages.append(ensure_string_content(conversation_history[i]))
            
        return messages
    
    def get_openai_api_call(self, messages):
        try:
            deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-deployment')
            
            response = self.client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                functions=self.get_agent_metadata(),
                function_call="auto"
            )
            return response
        except Exception as e:
            logging.error(f"Error in OpenAI API call: {str(e)}")
            raise
    
    def parse_response_with_voice(self, content):
        """Parse the response to extract formatted and voice parts"""
        if not content:
            return "", ""
        
        parts = content.split("|||VOICE|||")
        
        if len(parts) >= 2:
            formatted_response = parts[0].strip()
            voice_response = parts[1].strip()
        else:
            formatted_response = content.strip()
            sentences = formatted_response.split('.')
            if sentences:
                voice_response = sentences[0].strip() + "."
                voice_response = re.sub(r'\*\*|`|#|>|---', '', voice_response)
                voice_response = re.sub(r'\s+', ' ', voice_response).strip()
            else:
                voice_response = "I've completed your request."
        
        return formatted_response, voice_response

    def get_response(self, prompt, conversation_history, max_retries=3, retry_delay=2):
        try:
            if isinstance(conversation_history, list):
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                    logging.info(f"Trimmed conversation history to last 20 messages")
            
            guid_from_history = self._check_first_message_for_guid(conversation_history)
            guid_from_prompt = self.extract_user_guid(prompt)
            
            target_guid = guid_from_history or guid_from_prompt
            
            if target_guid and target_guid != self.user_guid:
                self.user_guid = target_guid
                self._initialize_context_memory(self.user_guid)
                logging.info(f"User GUID updated to: {self.user_guid}")
            elif not self.user_guid:
                self.user_guid = DEFAULT_USER_GUID
                self._initialize_context_memory(self.user_guid)
                logging.info(f"Using default User GUID: {self.user_guid}")
            
            prompt = str(prompt) if prompt is not None else ""
            
            if guid_from_prompt and prompt.strip() == guid_from_prompt and self.user_guid == guid_from_prompt:
                formatted = "I've successfully loaded your conversation memory. How can I assist you today?"
                voice = "I've loaded your memory - what can I help you with?"
                return formatted, voice, ""
            
            messages = self.prepare_messages(conversation_history)
            messages.append(ensure_string_content({"role": "user", "content": prompt}))

            agent_logs = []
            retry_count = 0
            needs_follow_up = False

            while retry_count < max_retries:
                try:
                    response = self.get_openai_api_call(messages)
                    assistant_msg = response.choices[0].message
                    msg_contents = assistant_msg.content or ""

                    if not assistant_msg.function_call:
                        formatted_response, voice_response = self.parse_response_with_voice(msg_contents)
                        return formatted_response, voice_response, "\n".join(map(str, agent_logs))

                    agent_name = str(assistant_msg.function_call.name)
                    agent = self.known_agents.get(agent_name)

                    if not agent:
                        return f"Agent '{agent_name}' does not exist", "I couldn't find that agent.", ""

                    json_data = ensure_string_function_args(assistant_msg.function_call)
                    logging.info(f"JSON data before parsing: {json_data}")

                    try:
                        agent_parameters = safe_json_loads(json_data)
                        
                        sanitized_parameters = {}
                        for key, value in agent_parameters.items():
                            if value is None:
                                sanitized_parameters[key] = ""
                            else:
                                sanitized_parameters[key] = value
                        
                        if agent_name in ['ManageMemory', 'ContextMemory']:
                            sanitized_parameters['user_guid'] = self.user_guid
                        
                        result = agent.perform(**sanitized_parameters)
                        
                        if result is None:
                            result = "Agent completed successfully"
                        else:
                            result = str(result)
                            
                        agent_logs.append(f"Performed {agent_name} and got result: {result}")
                            
                    except Exception as e:
                        logging.error(f"Error in agent execution: {str(e)}")
                        return f"Error parsing parameters: {str(e)}", "I hit an error processing that.", ""

                    messages.append({
                        "role": "function",
                        "name": agent_name,
                        "content": result
                    })
                    
                    try:
                        result_json = json.loads(result)
                        needs_follow_up = False
                        if isinstance(result_json, dict):
                            if result_json.get('error') or result_json.get('status') == 'incomplete':
                                needs_follow_up = True
                            if result_json.get('requires_additional_action') == True:
                                needs_follow_up = True
                    except:
                        needs_follow_up = False
                    
                    if not needs_follow_up:
                        final_response = self.get_openai_api_call(messages)
                        final_msg = final_response.choices[0].message
                        final_content = final_msg.content or ""
                        formatted_response, voice_response = self.parse_response_with_voice(final_content)
                        return formatted_response, voice_response, "\n".join(map(str, agent_logs))

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"Error occurred: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"Max retries reached. Error: {str(e)}")
                        return "An error occurred. Please try again.", "Something went wrong - try again.", ""

            return "Service temporarily unavailable. Please try again later.", "Service is down - try again later.", ""
            
        except Exception as e:
            logging.error(f"Critical error in get_response: {str(e)}")
            return "A critical error occurred. Please try again.", "Something went wrong - try again.", ""

app = func.FunctionApp()

@app.route(route="businessinsightbot_function", auth_level=func.AuthLevel.FUNCTION)
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    origin = req.headers.get('origin')
    cors_headers = build_cors_response(origin)

    if req.method == 'OPTIONS':
        return func.HttpResponse(
            status_code=200,
            headers=cors_headers
        )

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON in request body",
            status_code=400,
            headers=cors_headers
        )

    if not req_body:
        return func.HttpResponse(
            "Missing JSON payload in request body",
            status_code=400,
            headers=cors_headers
        )

    user_input = req_body.get('user_input')
    if user_input is None:
        user_input = ""
    else:
        user_input = str(user_input)
    
    conversation_history = req_body.get('conversation_history', [])
    if not isinstance(conversation_history, list):
        conversation_history = []
    
    user_guid = req_body.get('user_guid')
    
    is_guid_only = re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', user_input.strip(), re.IGNORECASE)
    
    if not is_guid_only and not user_input.strip():
        return func.HttpResponse(
            json.dumps({
                "error": "Missing or empty user_input in JSON payload"
            }),
            status_code=400,
            mimetype="application/json",
            headers=cors_headers
        )

    try:
        agents = load_agents_from_folder()
        assistant = Assistant(agents)
        
        if user_guid:
            assistant.user_guid = user_guid
            assistant._initialize_context_memory(user_guid)
        elif is_guid_only:
            assistant.user_guid = user_input.strip()
            assistant._initialize_context_memory(user_input.strip())
            
        assistant_response, voice_response, agent_logs = assistant.get_response(
            user_input, conversation_history)

        response = {
            "assistant_response": str(assistant_response),
            "voice_response": str(voice_response),
            "agent_logs": str(agent_logs),
            "user_guid": assistant.user_guid
        }

        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
            headers=cors_headers
        )
    except Exception as e:
        error_response = {
            "error": "Internal server error",
            "details": str(e)
        }
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json",
            headers=cors_headers
        )