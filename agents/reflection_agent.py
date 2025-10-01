from agents.basic_agent import BasicAgent
from utils.reflection_engine import ReflectionOrchestrator
from utils.azure_file_storage import AzureFileStorageManager
import json

class ReflectionAgent(BasicAgent):
    def __init__(self):
        self.name = "DeepReflection"
        self.metadata = {
            "name": self.name,
            "description": "Performs multi-dimensional reflection analysis on memories to reveal patterns, insights, and temporal dynamics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_guid": {
                        "type": "string",
                        "description": "Optional user identifier to reflect on their specific memories."
                    },
                    "reflection_type": {
                        "type": "string",
                        "description": "Type of reflection to perform.",
                        "enum": ["temporal", "emotional", "relational", "comprehensive"]
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": "Whether to include shared memories in reflection. Default is true."
                    }
                },
                "required": []
            }
        }
        self.storage_manager = AzureFileStorageManager()
        self.orchestrator = ReflectionOrchestrator()
        super().__init__(name=self.name, metadata=self.metadata)
    
    def perform(self, **kwargs):
        user_guid = kwargs.get('user_guid')
        reflection_type = kwargs.get('reflection_type', 'comprehensive')
        include_shared = kwargs.get('include_shared', True)
        
        self.storage_manager.set_memory_context(user_guid)
        user_memories = self.storage_manager.read_json()
        
        shared_memories = None
        if include_shared:
            self.storage_manager.set_memory_context(None)
            shared_memories = self.storage_manager.read_json()
        
        if reflection_type == "comprehensive":
            reflections = self.orchestrator.generate_comprehensive_reflection(
                user_memories, 
                shared_memories
            )
            summary = self.orchestrator.synthesize_reflection_summary(reflections)
            
            return json.dumps({
                "reflection_type": "comprehensive",
                "summary": summary,
                "detailed_reflections": reflections
            }, indent=2)
        
        elif reflection_type == "temporal":
            temporal_reflections = self.orchestrator.temporal_engine.reflect_at_all_scales(user_memories)
            return json.dumps({
                "reflection_type": "temporal",
                "reflections": temporal_reflections
            }, indent=2)
        
        elif reflection_type == "emotional":
            emotional_analysis = self.orchestrator.emotional_engine.analyze_emotional_trajectory(user_memories)
            return json.dumps({
                "reflection_type": "emotional",
                "analysis": emotional_analysis
            }, indent=2)
        
        elif reflection_type == "relational":
            graph_analysis = self.orchestrator.relationship_engine.build_concept_graph(user_memories)
            return json.dumps({
                "reflection_type": "relational",
                "graph": graph_analysis
            }, indent=2)
        
        return "Unknown reflection type requested."
