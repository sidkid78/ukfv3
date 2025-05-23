"""
Enhanced Layer 3: Advanced AI Research Agents (Azure OpenAI Integration)
- Sophisticated multi-agent orchestration using OpenAI Agents framework
- Dynamic consensus mechanisms and conflict resolution
- Advanced fork detection and memory patching
- Recursive research capabilities with confidence thresholding
- Integration with audit logging and compliance checking
- Configured for Azure OpenAI deployments
"""

import asyncio
import uuid
import random
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

from .base import BaseLayer

# OpenAI Agents imports
from agents import Agent, function_tool, handoff, Runner
from agents.run_context import RunContextWrapper
from agents.model_settings import ModelSettings

# Azure configuration
from azure_config import azure_config, setup_azure_openai

# OpenAI client


class ResearchMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


class ConflictResolution(Enum):
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_OVERRIDE = "expert_override"
    FORK_BRANCH = "fork_branch"


@dataclass
class ResearchTask:
    query: str
    context: Dict[str, Any]
    priority: int = 1
    mode: ResearchMode = ResearchMode.PARALLEL
    required_personas: List[str] = None
    confidence_threshold: float = 0.995
    max_iterations: int = 3


@dataclass
class AgentResponse:
    agent_id: str
    persona: str
    answer: Any
    confidence: float
    reasoning: str
    evidence: List[str]
    timestamp: float
    memory_patches: List[Dict] = None


# Define function tools for memory operations
@function_tool
def search_memory_graph(context: RunContextWrapper, coordinate: List[float], persona: str = None) -> str:
    """Search the UKG memory graph for information at specific coordinates."""
    # This would integrate with your actual memory system
    # For now, return a simulated response based on coordinate analysis
    coord_summary = f"[{coordinate[0]:.2f}, {coordinate[1]:.2f}, {coordinate[2]:.2f}...]"
    if persona:
        return f"Memory search at {coord_summary} for persona '{persona}': Found knowledge cell with relevant domain information."
    return f"Memory search at {coord_summary}: Located general knowledge cell with baseline information."

@function_tool
def patch_memory_cell(context: RunContextWrapper, coordinate: List[float], value: str, persona: str) -> str:
    """Patch a memory cell with new information."""
    coord_summary = f"[{coordinate[0]:.2f}, {coordinate[1]:.2f}, {coordinate[2]:.2f}...]"
    return f"Successfully patched memory cell at {coord_summary} with new research findings for persona '{persona}'. Cell updated with enhanced knowledge."

@function_tool
def fork_memory_cell(context: RunContextWrapper, coordinate: List[float], new_value: str, reason: str) -> str:
    """Create a fork of existing memory cell for alternative reasoning paths."""
    coord_summary = f"[{coordinate[0]:.2f}, {coordinate[1]:.2f}, {coordinate[2]:.2f}...]"
    return f"Successfully forked memory cell at {coord_summary}. Created alternative branch for reason: {reason}. Fork allows parallel reasoning paths."

@function_tool
def analyze_consensus(context: RunContextWrapper, responses: List[str]) -> str:
    """Analyze multiple agent responses for consensus patterns."""
    if len(set(responses)) == 1:
        return "Strong consensus detected - all agents converged on the same conclusion with high agreement."
    elif len(set(responses)) == len(responses):
        return "No consensus found - all agents provided different perspectives. Recommend forking for parallel analysis."
    else:
        return f"Partial consensus detected - {len(set(responses))} distinct viewpoints among {len(responses)} agents. Some alignment exists."

@function_tool
def escalate_to_higher_layer(context: RunContextWrapper, reason: str, confidence: float) -> str:
    """Escalate analysis to higher simulation layers when confidence is insufficient."""
    return f"Escalation triggered: {reason}. Current confidence {confidence:.3f} below threshold. Recommending Layer 4+ engagement."


class EnhancedLayer3(BaseLayer):
    """Enhanced Layer 3 using OpenAI Agents framework with Azure OpenAI"""
    
    def __init__(self):
        super().__init__(layer_id=3)
        self.requires_escalation = True
        
        # Ensure Azure OpenAI is properly configured
        if not setup_azure_openai():
            raise RuntimeError("Failed to configure Azure OpenAI client")
        
        # Create specialized research agents using OpenAI Agents with Azure deployments
        self.domain_expert = Agent(
            name="DomainExpert",
            instructions="""You are a domain expert research agent with deep specialized knowledge and access to the UKG memory graph system.
            
            Your role:
            1. Analyze queries with technical precision and domain expertise
            2. Search memory graphs for relevant information using search_memory_graph
            3. Provide high-confidence answers backed by evidence from the knowledge base
            4. Patch memory cells with new insights using patch_memory_cell when you discover new knowledge
            5. Identify when information is insufficient and escalation is needed
            
            IMPORTANT: Always provide confidence scores (0.0-1.0) in your responses. 
            Format: "Confidence: 0.95" or "My confidence in this analysis is 0.87"
            
            Use the available memory tools to enhance your analysis and ensure knowledge persistence.""",
            tools=[search_memory_graph, patch_memory_cell, fork_memory_cell, escalate_to_higher_layer],
            model=azure_config.programmable_deployment,
            model_settings=ModelSettings(temperature=0.2, max_tokens=1000)
        )
        
        self.critical_analyst = Agent(
            name="CriticalAnalyst", 
            instructions="""You are a critical analyst who questions assumptions and validates research findings.
            
            Your role:
            1. Review and challenge findings from other agents with constructive skepticism
            2. Search memory for contradictory or supporting evidence
            3. Look for logical flaws, gaps, or biases in reasoning
            4. Suggest alternative interpretations and fork memory when needed
            5. Lower confidence when uncertainties are found but provide clear reasoning
            
            IMPORTANT: Always provide confidence scores (0.0-1.0) in your responses.
            Be constructively skeptical - challenge ideas while helping improve analysis quality.
            
            When you find conflicting evidence, use fork_memory_cell to preserve alternative viewpoints.""",
            tools=[search_memory_graph, fork_memory_cell, analyze_consensus, escalate_to_higher_layer],
            model=azure_config.sub_agent_deployment,
            model_settings=ModelSettings(temperature=0.4, max_tokens=800)
        )
        
        self.consensus_builder = Agent(
            name="ConsensusBuilder",
            instructions="""You are a consensus-building agent that resolves conflicts between research agents and synthesizes findings.
            
            Your role:
            1. Analyze conflicting viewpoints from multiple agents using analyze_consensus
            2. Find common ground and synthesize different perspectives
            3. Make final recommendations when agents disagree
            4. Decide when to fork vs. when to choose one unified path
            5. Coordinate multi-agent research efforts and build coherent conclusions
            
            IMPORTANT: Always provide confidence scores (0.0-1.0) in your responses.
            Focus on finding the most robust and well-supported conclusions while preserving valuable alternative viewpoints.
            
            Use memory tools to ensure consensus decisions are properly recorded.""",
            tools=[analyze_consensus, patch_memory_cell, fork_memory_cell, escalate_to_higher_layer],
            model=azure_config.programmable_deployment,
            model_settings=ModelSettings(temperature=0.3, max_tokens=1200)
        )
        
        # Set up handoffs between agents
        self.analyst_handoff = handoff(
            self.critical_analyst,
            tool_description_override="Hand off to critical analyst for validation, skeptical review, and identification of potential flaws or alternative perspectives"
        )
        
        self.consensus_handoff = handoff(
            self.consensus_builder,
            tool_description_override="Hand off to consensus builder to resolve conflicts, synthesize findings, and build coherent conclusions from multiple agent perspectives"
        )
        
        # Add handoffs to primary agent
        self.domain_expert = self.domain_expert.clone(
            handoffs=[self.analyst_handoff, self.consensus_handoff]
        )
        
        self.specialized_personas = {
            "domain_expert": {
                "agent": self.domain_expert,
                "description": "Deep domain knowledge specialist with Azure OpenAI integration",
                "confidence_boost": 0.05,
                "specialties": ["technical_analysis", "domain_facts", "memory_integration"]
            },
            "critical_analyst": {
                "agent": self.critical_analyst,
                "description": "Skeptical reviewer and validator with bias detection", 
                "confidence_penalty": -0.02,
                "specialties": ["error_detection", "logical_validation", "alternative_perspectives"]
            },
            "consensus_builder": {
                "agent": self.consensus_builder,
                "description": "Conflict resolver and team coordinator using advanced synthesis",
                "confidence_boost": 0.02,
                "specialties": ["conflict_resolution", "team_coordination", "knowledge_synthesis"]
            }
        }

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced multi-agent research using OpenAI Agents with Azure OpenAI"""
        
        if not context.get('graph_initialized', False):
            raise ValueError("Knowledge graph not initialized (Layer2 required)")
        
        query = context.get('normalized', {}).get('query', '')
        axes = context.get('axes', [0.0] * 13)
        
        # Log the research initiation
        research_start_time = time.time()
        
        # Determine research strategy based on context and previous confidence
        research_strategy = self._determine_research_strategy(context)
        
        try:
            # Execute research using the selected strategy
            if research_strategy == "hierarchical":
                result = await self._execute_hierarchical_research(query, axes, context)
            elif research_strategy == "consensus":
                result = await self._execute_consensus_research(query, axes, context)
            else:  # parallel (default)
                result = await self._execute_parallel_research(query, axes, context)
            
            # Calculate research duration
            research_duration = time.time() - research_start_time
            
            # Process results and update context
            enhanced_context = {
                **context,
                'layer3_research_completed': True,
                'research_strategy': research_strategy,
                'research_duration': research_duration,
                'agent_responses': result['responses'],
                'final_confidence': result['confidence'],
                'escalation_needed': result['escalate'],
                'research_trace': result['trace'],
                'azure_deployment_used': azure_config.programmable_deployment,
                'timestamp': time.time()
            }
            
            # Add escalation flag if confidence is too low
            if result['confidence'] < 0.995:
                enhanced_context['requires_escalation'] = True
                enhanced_context['escalation_reason'] = f"Confidence {result['confidence']:.3f} below threshold 0.995"
            
            return enhanced_context
            
        except Exception as e:
            # Handle errors gracefully and provide context about the failure
            error_context = {
                **context,
                'layer3_error': True,
                'error_message': str(e),
                'research_strategy': research_strategy,
                'research_duration': time.time() - research_start_time,
                'requires_escalation': True,
                'escalation_reason': f"Layer 3 execution failed: {str(e)}"
            }
            return error_context
    
    def _determine_research_strategy(self, context: Dict[str, Any]) -> str:
        """Determine the best research strategy based on context and Azure capabilities"""
        query = context.get('normalized', {}).get('query', '').lower()
        prev_confidence = context.get('confidence', 1.0)
        prev_layer_complexity = context.get('layer2_complexity', 'low')
        
        # Use hierarchical for complex scenarios or when previous confidence is low
        if (prev_confidence < 0.7 or 
            prev_layer_complexity == 'high' or
            any(word in query for word in ['complex', 'uncertain', 'ambiguous', 'multilayered', 'deep'])):
            return "hierarchical"
        
        # Use consensus for conflict-prone or collaborative scenarios
        if any(word in query for word in ['conflict', 'debate', 'controversial', 'disagree', 'multiple', 'perspectives']):
            return "consensus"
        
        # Use parallel for efficiency with Azure's fast response times
        return "parallel"
    
    async def _execute_parallel_research(self, query: str, axes: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel research with domain expert and critical analyst using Azure OpenAI"""
        
        research_context = {
            "query": query,
            "axes": axes, 
            "layer": self.layer_id,
            "simulation_context": context,
            "azure_deployment": azure_config.programmable_deployment
        }
        
        try:
            # Run domain expert research with enhanced prompt including coordinate information
            expert_prompt = f"""Research Query: {query}

Memory Coordinates: {axes[:5]}... (13-dimensional UKG coordinate system)
Context: {context.get('summary', 'Analyzing query within UKG simulation framework')}

Analyze this query thoroughly using your domain expertise. Consider:
1. Search the memory graph at the provided coordinates for relevant information
2. Provide your expert assessment with detailed reasoning
3. Include confidence level in your response (format: "Confidence: 0.XX")
4. Identify any areas needing further investigation
5. Recommend memory patches if new insights are discovered

Be thorough but concise in your analysis."""

            expert_result = await Runner.run(
                starting_agent=self.domain_expert,
                input=expert_prompt,
                context=research_context
            )
            
            # Extract confidence and check if validation needed
            expert_confidence = self._extract_confidence(expert_result.final_output)
            
            responses = [{
                "agent": "domain_expert",
                "output": expert_result.final_output,
                "confidence": expert_confidence,
                "trace": [item.to_input_item() for item in expert_result.new_items],
                "tool_calls": len([item for item in expert_result.new_items if hasattr(item, 'type') and 'tool' in str(item.type)])
            }]
            
            # If confidence is low or query seems complex, run critical analyst
            if expert_confidence < 0.9 or len(query.split()) > 10:
                analyst_prompt = f"""Critical Review Task:

Expert Analysis: {expert_result.final_output}

Original Query: {query}
Memory Coordinates: {axes[:5]}...

Your critical analysis should:
1. Evaluate the expert's reasoning for logical flaws or gaps
2. Search memory for contradictory or supporting evidence
3. Suggest alternative interpretations or missing considerations
4. Provide your confidence assessment (format: "Confidence: 0.XX")
5. Recommend forking if you find significant alternative viewpoints

Be constructively skeptical while helping improve the analysis quality."""

                analyst_result = await Runner.run(
                    starting_agent=self.critical_analyst,
                    input=analyst_prompt,
                    context=research_context
                )
                
                analyst_confidence = self._extract_confidence(analyst_result.final_output)
                responses.append({
                    "agent": "critical_analyst", 
                    "output": analyst_result.final_output,
                    "confidence": analyst_confidence,
                    "trace": [item.to_input_item() for item in analyst_result.new_items],
                    "tool_calls": len([item for item in analyst_result.new_items if hasattr(item, 'type') and 'tool' in str(item.type)])
                })
            
            # Calculate final metrics
            final_confidence = self._calculate_final_confidence(responses)
            escalate = final_confidence < 0.995
            
            return {
                "responses": responses,
                "confidence": final_confidence,
                "escalate": escalate,
                "trace": {
                    "strategy": "parallel",
                    "agent_count": len(responses),
                    "expert_confidence": expert_confidence,
                    "final_confidence": final_confidence,
                    "azure_deployment": azure_config.programmable_deployment,
                    "total_tool_calls": sum(r.get("tool_calls", 0) for r in responses)
                }
            }
            
        except Exception as e:
            # Return error state with escalation recommendation
            return {
                "responses": [],
                "confidence": 0.0,
                "escalate": True,
                "trace": {
                    "strategy": "parallel",
                    "error": str(e),
                    "azure_deployment": azure_config.programmable_deployment
                }
            }
    
    async def _execute_hierarchical_research(self, query: str, axes: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hierarchical research: expert -> analyst -> consensus"""
        
        research_context = {
            "query": query,
            "axes": axes,
            "layer": self.layer_id, 
            "simulation_context": context
        }
        
        responses = []
        
        # Tier 1: Domain Expert
        expert_result = await Runner.run(
            starting_agent=self.domain_expert,
            input=f"Initial Research: {query}\n\nProvide comprehensive domain expert analysis with confidence assessment.",
            context=research_context
        )
        
        expert_confidence = self._extract_confidence(expert_result.final_output)
        responses.append({
            "agent": "domain_expert",
            "tier": 1,
            "output": expert_result.final_output,
            "confidence": expert_confidence,
            "trace": [item.to_input_item() for item in expert_result.new_items]
        })
        
        # Tier 2: Critical Analysis (if needed)
        if expert_confidence < 0.95:
            analyst_result = await Runner.run(
                starting_agent=self.critical_analyst,
                input=f"Critical Review Tier 2: {expert_result.final_output}\n\nOriginal query: {query}\n\nProvide thorough critical analysis and identify any issues.",
                context=research_context
            )
            
            analyst_confidence = self._extract_confidence(analyst_result.final_output)
            responses.append({
                "agent": "critical_analyst",
                "tier": 2,
                "output": analyst_result.final_output,
                "confidence": analyst_confidence,
                "trace": [item.to_input_item() for item in analyst_result.new_items]
            })
            
            # Tier 3: Consensus Building (if still issues)
            if analyst_confidence < 0.9:
                consensus_result = await Runner.run(
                    starting_agent=self.consensus_builder,
                    input=f"Consensus Building Tier 3:\n\nExpert: {expert_result.final_output}\n\nAnalyst: {analyst_result.final_output}\n\nOriginal query: {query}\n\nResolve conflicts and provide final synthesis.",
                    context=research_context
                )
                
                consensus_confidence = self._extract_confidence(consensus_result.final_output)
                responses.append({
                    "agent": "consensus_builder",
                    "tier": 3,
                    "output": consensus_result.final_output,
                    "confidence": consensus_confidence,
                    "trace": [item.to_input_item() for item in consensus_result.new_items]
                })
        
        final_confidence = self._calculate_final_confidence(responses)
        escalate = final_confidence < 0.995 or len(responses) >= 3
        
        return {
            "responses": responses,
            "confidence": final_confidence,
            "escalate": escalate,
            "trace": {
                "strategy": "hierarchical",
                "tiers_used": len(responses),
                "final_confidence": final_confidence,
                "escalation_triggered": escalate
            }
        }
    
    async def _execute_consensus_research(self, query: str, axes: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research with consensus-building approach"""
        
        research_context = {
            "query": query,
            "axes": axes,
            "layer": self.layer_id,
            "simulation_context": context
        }
        
        # Run all agents in parallel first
        expert_task = Runner.run(
            starting_agent=self.domain_expert,
            input=f"Consensus Research - Expert View: {query}\n\nProvide your expert perspective, noting any areas of potential disagreement.",
            context=research_context
        )
        
        analyst_task = Runner.run(
            starting_agent=self.critical_analyst,  
            input=f"Consensus Research - Critical View: {query}\n\nProvide critical analysis and alternative perspectives.",
            context=research_context
        )
        
        # Wait for both to complete
        expert_result, analyst_result = await asyncio.gather(expert_task, analyst_task)
        
        expert_confidence = self._extract_confidence(expert_result.final_output)
        analyst_confidence = self._extract_confidence(analyst_result.final_output)
        
        responses = [
            {
                "agent": "domain_expert",
                "output": expert_result.final_output,
                "confidence": expert_confidence,
                "trace": [item.to_input_item() for item in expert_result.new_items]
            },
            {
                "agent": "critical_analyst",
                "output": analyst_result.final_output, 
                "confidence": analyst_confidence,
                "trace": [item.to_input_item() for item in analyst_result.new_items]
            }
        ]
        
        # Check for consensus
        consensus_needed = self._detect_consensus_needed(responses)
        
        if consensus_needed:
            consensus_result = await Runner.run(
                starting_agent=self.consensus_builder,
                input=f"Build Consensus:\n\nExpert View: {expert_result.final_output}\n\nCritical View: {analyst_result.final_output}\n\nOriginal Query: {query}\n\nSynthesize these perspectives and build consensus.",
                context=research_context
            )
            
            consensus_confidence = self._extract_confidence(consensus_result.final_output)
            responses.append({
                "agent": "consensus_builder",
                "output": consensus_result.final_output,
                "confidence": consensus_confidence,
                "trace": [item.to_input_item() for item in consensus_result.new_items]
            })
        
        final_confidence = self._calculate_final_confidence(responses)
        escalate = final_confidence < 0.995
        
        return {
            "responses": responses,
            "confidence": final_confidence,
            "escalate": escalate,
            "trace": {
                "strategy": "consensus",
                "consensus_needed": consensus_needed,
                "final_confidence": final_confidence,
                "agent_count": len(responses)
            }
        }
    
    def _extract_confidence(self, agent_output: str) -> float:
        """Extract confidence score from agent output"""
        if not agent_output:
            return 0.5
        
        # Look for explicit confidence statements
        confidence_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'confidence.*?([0-9.]+)', 
            r'([0-9.]+).*confidence',
            r'certainty.*?([0-9.]+)',
            r'sure.*?([0-9.]+)'
        ]
        
        text = str(agent_output).lower()
        for pattern in confidence_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    confidence = float(match.group(1))
                    # Handle percentage vs decimal
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    return min(1.0, max(0.0, confidence))
                except:
                    continue
        
        # Heuristic confidence based on output characteristics
        if len(text) > 200 and any(word in text for word in ['evidence', 'analysis', 'reasoning']):
            return 0.92
        elif len(text) > 100:
            return 0.85
        else:
            return 0.75
    
    def _calculate_final_confidence(self, responses: List[Dict]) -> float:
        """Calculate final confidence considering all responses"""
        if not responses:
            return 0.0
        
        confidences = [r['confidence'] for r in responses if r['confidence'] is not None]
        if not confidences:
            # If no explicit confidence scores, or only one, can't calculate variance meaningfully
            # Consider returning an average or a default low confidence if critical
            return sum(confidences) / len(confidences) if confidences else 0.0 

        # Example: Variance-based confidence adjustment (simple version)
        # More sophisticated metrics could be used here (e.g., agreement between top N agents)
        variance = max(confidences) - min(confidences) if len(confidences) > 1 else 0.0
        
        # Average confidence penalized by variance (example logic)
        avg_confidence = sum(confidences) / len(confidences)
        final_confidence = avg_confidence * (1 - variance) # Penalize high variance
        
        return max(0.0, min(1.0, final_confidence)) # Clamp between 0 and 1
    
    def _detect_consensus_needed(self, responses: List[Dict]) -> bool:
        """Detect if consensus is needed based on response variance and conflicts"""
        if not responses or len(responses) < 2:
            return False

        # Check for significant confidence variance
        confidences = [r['confidence'] for r in responses if 'confidence' in r and r['confidence'] is not None]
        if not confidences:
            return True # No confidence scores, assume consensus needed

        conf_variance = max(confidences) - min(confidences) if confidences else 0

        # Check for explicit conflict indicators (example, can be more sophisticated)
        conflict_indicators = [r.get('conflict_detected', False) for r in responses]
        
        return conf_variance > 0.1 or any(conflict_indicators)

    def _detect_forks(self, responses: List[AgentResponse]) -> bool:
        """Detect if responses represent conflicting viewpoints (forks)"""
        if len(responses) < 2:
            return False
        
        # Simple fork detection: more than one unique answer string
        # More advanced: semantic similarity, key entity extraction, etc.
        unique_answers = set()
        for r in responses:
            # Normalize or get a comparable representation of the answer
            # For simplicity, converting to string. For complex objects, use a hash or key fields.
            answer_repr = str(r.answer) 
            unique_answers.add(answer_repr)
            
        # Consider it a fork if there's more than one substantially different answer group
        # This could be refined by clustering answers and checking inter-cluster distance
        
        # Example: if more than 60% of answers are unique, consider it a fork situation
        # or if there are distinct groups of answers.
        # For now, simple check: if more than 1 unique answer, it's a potential fork point.
        # This needs refinement based on how "different" answers should be to count as a fork.
        
        # A more robust check might involve looking at confidence levels as well.
        # If multiple high-confidence answers differ, that's a strong fork indicator.
        
        # Placeholder: If there are multiple distinct answer strings, flag a potential fork.
        # This assumes answers are simple strings or well-represented by stringification.
        
        # Refined simple check: multiple unique answers considered a fork.
        # This could be too sensitive for minor variations.
        # A better approach would group similar answers.
        answer_strings = [str(r.answer) for r in responses]
        significant_groups = {s: answer_strings.count(s) for s in set(answer_strings)}

        return len(significant_groups) > 1

    def _generate_fork_patches(self, responses: List[AgentResponse], task: ResearchTask, memory) -> List[Dict]:
        """Generate memory patches for fork scenarios"""
        patches = []
        if self._detect_forks(responses):
            # Example: Create separate memory cells for each distinct answer group
            answer_groups = {}
            for r in responses:
                answer_key = str(r.answer) # Or a more robust hashing/grouping
                if answer_key not in answer_groups:
                    answer_groups[answer_key] = []
                answer_groups[answer_key].append(r)
            
            base_coordinate = task.context.get('axes', [0.0]*13) # Base coordinate from task
            
            for i, (answer_key, group_responses) in enumerate(answer_groups.items()):
                if len(answer_groups) > 1 : # Only create forks if there are multiple distinct groups
                    # Create a slightly offset coordinate for the fork
                    # This is a simplistic approach; real fork management would be more complex
                    fork_coordinate = base_coordinate[:] # Copy
                    fork_coordinate[0] += (i + 1) * 0.01 # Offset first axis for fork
                    
                    # Synthesize a value for the forked cell (e.g., the answer itself or a summary)
                    fork_value = {"original_query": task.query, "forked_answer": answer_key, "supporting_agents": [r.agent_id for r in group_responses]}
                    
                    patches.append({
                        "coordinate": fork_coordinate,
                        "value": fork_value,
                        "meta": {
                            "fork_reason": "Conflicting agent responses",
                            "original_coordinate": base_coordinate,
                            "timestamp": time.time()
                        }
                    })
        return patches

    def _generate_evidence(self, agent, result: Dict, memory) -> List[str]:
        """Generate evidence list for agent response"""
        evidence = []
        # Example: Extract evidence from agent's reasoning string or memory interactions
        if "found in memory" in result.get("reasoning", "").lower():
            evidence.append("Information retrieved from UKG memory graph.")
        
        if result.get("memory_patches"):
            evidence.append(f"{len(result['memory_patches'])} memory cells updated based on findings.")
            
        # Add any specific evidence markers from the agent's output
        # This would require the agent to tag evidence explicitly
        
        # Simulated evidence based on confidence
        if result.get("confidence", 0.0) > 0.95:
            evidence.append("High confidence in supporting data.")
        elif result.get("confidence", 0.0) > 0.8:
            evidence.append("Moderate confidence, some assumptions made.")
        else:
            evidence.append("Lower confidence, further verification may be needed.")
            
        # Add references to specific memory cells if agent tools provide them
        # E.g., if search_memory_graph tool returns cell IDs
        
        # Placeholder for structured evidence from agent's tools
        # For example, if agent tools return a list of source URIs or memory cell IDs
        # tool_evidence = result.get('tool_outputs', {}).get('search_memory_graph', {}).get('references', [])
        # evidence.extend(tool_evidence)
        
        if not evidence:
            evidence.append("General reasoning based on internal knowledge model.")
            
        return evidence

    def _generate_reasoning(self, agent, result: Dict, persona_config: Dict) -> str:
        """Generate reasoning explanation for agent response"""
        base_reasoning = f"Agent {agent.name} ({persona_config.get('description', 'Specialized Agent')}) analyzed the query." # Use agent.name
        
        # Add details from agent's specific output
        if result.get("reasoning"):
             base_reasoning += f" Key findings: {result['reasoning'][:200]}..." # Summarize
             
        # Add confidence statement
        confidence_reasoning = f" Confidence level: {result.get('confidence', 0.0):.2f}."
        if result.get("confidence", 0.0) < 0.8:
            confidence_reasoning += " Further analysis may refine this."
            
        return base_reasoning + confidence_reasoning

    def _response_to_dict(self, response: AgentResponse) -> Dict:
        """Convert AgentResponse to dictionary for serialization"""
        if not response:
            return None
        return {
            "agent_id": response.agent_id,
            "persona": response.persona,
            "answer": response.answer, # Ensure answer is serializable
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "evidence": response.evidence,
            "timestamp": response.timestamp,
            "memory_patches": response.memory_patches
        }


# Import time for timestamp
import time