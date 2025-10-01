import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

class TemporalReflectionEngine:
    def __init__(self):
        self.reflection_scales = {
            'immediate': timedelta(hours=1),
            'session': timedelta(days=1),
            'weekly': timedelta(days=7),
            'monthly': timedelta(days=30),
            'lifetime': None
        }
    
    def reflect_at_all_scales(self, memories):
        now = datetime.now()
        reflections = {}
        
        for scale_name, time_delta in self.reflection_scales.items():
            reflections[scale_name] = self._reflect_at_scale(
                scale_name, time_delta, memories, now
            )
        
        return reflections
    
    def _reflect_at_scale(self, scale_name, time_delta, memories, now):
        if not memories:
            return f"No {scale_name} memories available."
        
        filtered_memories = []
        for memory_id, memory_data in memories.items():
            if not isinstance(memory_data, dict):
                continue
            
            try:
                memory_date = memory_data.get('date', '')
                memory_time = memory_data.get('time', '')
                if memory_date and memory_time:
                    memory_dt = datetime.strptime(
                        f"{memory_date} {memory_time}", 
                        "%Y-%m-%d %H:%M:%S"
                    )
                    
                    if time_delta is None:
                        filtered_memories.append(memory_data)
                    elif (now - memory_dt) <= time_delta:
                        filtered_memories.append(memory_data)
            except:
                continue
        
        if not filtered_memories:
            return f"No memories in {scale_name} timeframe."
        
        return self._generate_scale_insights(scale_name, filtered_memories)
    
    def _generate_scale_insights(self, scale_name, memories):
        if scale_name == 'immediate':
            recent = memories[-3:] if len(memories) > 3 else memories
            items = [m['message'] for m in recent if 'message' in m]
            return f"Recent context: {'; '.join(items)}"
        
        elif scale_name == 'session':
            themes = [m.get('theme', 'general') for m in memories]
            theme_counts = defaultdict(int)
            for theme in themes:
                theme_counts[theme] += 1
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            return f"Today's focus: {', '.join([t[0] for t in top_themes])}"
        
        elif scale_name == 'weekly':
            return f"This week: {len(memories)} interactions across various topics"
        
        elif scale_name == 'monthly':
            themes = [m.get('theme', 'general') for m in memories]
            return f"Monthly trend: Focus on {', '.join(list(set(themes))[:5])}"
        
        elif scale_name == 'lifetime':
            total = len(memories)
            themes = set([m.get('theme', 'general') for m in memories])
            return f"Overall: {total} memories across {len(themes)} themes"
        
        return f"{len(memories)} memories at {scale_name} scale"


class EmotionalReflectionEngine:
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['happy', 'great', 'excellent', 'love', 'excited', 'wonderful', 'amazing'],
            'negative': ['frustrated', 'angry', 'difficult', 'problem', 'issue', 'concern', 'worried'],
            'neutral': ['information', 'data', 'report', 'update', 'meeting', 'discuss']
        }
    
    def analyze_emotional_trajectory(self, memories):
        if not memories:
            return {
                'current_mood': 'neutral',
                'mood_trend': 'stable',
                'emotional_summary': 'No emotional data available'
            }
        
        sentiment_timeline = []
        for memory_id, memory_data in memories.items():
            if isinstance(memory_data, dict) and 'message' in memory_data:
                sentiment = self._detect_sentiment(memory_data['message'])
                sentiment_timeline.append({
                    'sentiment': sentiment,
                    'timestamp': f"{memory_data.get('date', '')} {memory_data.get('time', '')}"
                })
        
        recent_sentiments = [s['sentiment'] for s in sentiment_timeline[-10:]]
        sentiment_counts = defaultdict(int)
        for s in recent_sentiments:
            sentiment_counts[s] += 1
        
        current_mood = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else 'neutral'
        
        if len(recent_sentiments) >= 5:
            early = recent_sentiments[:len(recent_sentiments)//2]
            late = recent_sentiments[len(recent_sentiments)//2:]
            early_pos = sum(1 for s in early if s == 'positive')
            late_pos = sum(1 for s in late if s == 'positive')
            
            if late_pos > early_pos:
                trend = 'improving'
            elif late_pos < early_pos:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'current_mood': current_mood,
            'mood_trend': trend,
            'emotional_summary': f"Mood is {current_mood} and {trend}",
            'sentiment_distribution': dict(sentiment_counts)
        }
    
    def _detect_sentiment(self, text):
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'


class MetaCognitiveEngine:
    def __init__(self):
        self.agent_call_history = defaultdict(int)
        self.interaction_patterns = []
    
    def record_agent_call(self, agent_name):
        self.agent_call_history[agent_name] += 1
    
    def analyze_self_performance(self):
        if not self.agent_call_history:
            return {
                'most_used_agents': [],
                'effectiveness_score': 1.0,
                'self_reflection': 'Insufficient data for self-analysis'
            }
        
        top_agents = sorted(
            self.agent_call_history.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        total_calls = sum(self.agent_call_history.values())
        
        return {
            'most_used_agents': [{'agent': a[0], 'calls': a[1]} for a in top_agents],
            'total_agent_calls': total_calls,
            'self_reflection': f"Most utilized capability: {top_agents[0][0] if top_agents else 'None'}"
        }


class RelationshipGraphEngine:
    def __init__(self):
        self.entity_relationships = defaultdict(lambda: defaultdict(int))
    
    def build_concept_graph(self, memories):
        entities_found = set()
        
        for memory_id, memory_data in memories.items():
            if isinstance(memory_data, dict) and 'message' in memory_data:
                entities = self._extract_entities(memory_data['message'])
                entities_found.update(entities)
                
                # Convert to list for indexing operations
                entities_list = list(entities)
                for i, e1 in enumerate(entities_list):
                    for e2 in entities_list[i+1:]:
                        self.entity_relationships[e1][e2] += 1
                        self.entity_relationships[e2][e1] += 1
        
        entity_scores = {}
        for entity in entities_found:
            entity_scores[entity] = sum(self.entity_relationships[entity].values())
        
        central_concepts = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_concepts': len(entities_found),
            'central_themes': [c[0] for c in central_concepts],
            'graph_summary': f"Knowledge graph contains {len(entities_found)} concepts"
        }
    
    def _extract_entities(self, text):
        words = text.split()
        entities = []
        
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word and len(clean_word) > 2:
                if clean_word[0].isupper():
                    entities.append(clean_word)
        
        return list(set(entities))


class ReflectionOrchestrator:
    def __init__(self):
        self.temporal_engine = TemporalReflectionEngine()
        self.emotional_engine = EmotionalReflectionEngine()
        self.metacognitive_engine = MetaCognitiveEngine()
        self.relationship_engine = RelationshipGraphEngine()
    
    def generate_comprehensive_reflection(self, memories, shared_memories=None):
        reflections = {
            'temporal': self.temporal_engine.reflect_at_all_scales(memories),
            'emotional': self.emotional_engine.analyze_emotional_trajectory(memories),
            'metacognitive': self.metacognitive_engine.analyze_self_performance(),
            'relational': self.relationship_engine.build_concept_graph(memories)
        }
        
        if shared_memories:
            reflections['shared_context'] = {
                'temporal': self.temporal_engine.reflect_at_all_scales(shared_memories),
                'relational': self.relationship_engine.build_concept_graph(shared_memories)
            }
        
        return reflections
    
    def synthesize_reflection_summary(self, reflections):
        summary_parts = []
        
        if 'temporal' in reflections:
            immediate = reflections['temporal'].get('immediate', '')
            summary_parts.append(f"Right now: {immediate}")
            
            weekly = reflections['temporal'].get('weekly', '')
            summary_parts.append(f"This week: {weekly}")
        
        if 'emotional' in reflections:
            emotional = reflections['emotional']
            summary_parts.append(
                f"Emotional state: {emotional.get('emotional_summary', 'Neutral')}"
            )
        
        if 'relational' in reflections:
            relational = reflections['relational']
            themes = relational.get('central_themes', [])
            if themes:
                summary_parts.append(f"Key themes: {', '.join(themes[:3])}")
        
        if 'metacognitive' in reflections:
            meta = reflections['metacognitive']
            summary_parts.append(f"{meta.get('self_reflection', '')}")
        
        return "\n".join(summary_parts)