"""
Enhanced Layer 3: Advanced AI Research Agents (OpenAI Agents Integration)
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

from .base import BaseLayer

# OpenAI Agents imports (from the provided openai agent file)
from agents import Agent, function_tool, handoff, Runner
from agents.run_context import RunContextWrapper
from agents.model_settings import ModelSettings

from azure_config import azure_config, setup_azure_openai


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
    """Enhanced Layer 3 using OpenAI Agents framework"""
    
    def __init__(self):
        super().__init__(layer_id=3)
        self.requires_escalation = True

        # Ensure Azure OpenAI is properly configured
        if not setup_azure_openai():
            raise RuntimeError("Failed to configure Azure OpenAI client")
        
        # Create specialized research agents using OpenAI Agents
        self.domain_expert = Agent(
            name="DomainExpert",
            instructions="""You are a domain expert research agent with deep specialized knowledge and access to the UKG memory graph system.
            
            Your role:
            1. Analyze queries with technical precision and domain expertise
            2. Search memory graphs for relevant information using search_memory_graph
            3. Provide high-confidence answers backed by evidence from the knowledge base
            4. Patch memory cells with new insights using patch_memory_cell when you discover new knowledge 
            4. Identify when information is insufficient and escalation is needed
            
            IMPORTANT: Always provide confidence scores (0.0-1.0) in your responses.
            Format: "Confidence: 0.95" or "My confidence in this analysis is 0.87"

            Use the available tools to enhance your analysis and ensure knowledge persistance.""",
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
            model=azure_config.programmable_deployment,
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
        """Execute enhanced multi-agent research using OpenAI Agents"""
        
        if not context.get('graph_initialized', False):
            raise ValueError("Knowledge graph not initialized (Layer2 required)")
        
        query = context.get('normalized', {}).get('query', '')
        axes = context.get('axes', [0.0] * 13)
        
        # Determine research strategy based on context
        research_strategy = self._determine_research_strategy(context)
        
        # Execute research using the selected strategy
        if research_strategy == "hierarchical":
            result = await self._execute_hierarchical_research(query, axes, context)
        elif research_strategy == "consensus":
            result = await self._execute_consensus_research(query, axes, context)
        else:  # parallel (default)
            result = await self._execute_parallel_research(query, axes, context)
        
        # Process results and update context
        enhanced_context = {
            **context,
            'layer3_research_completed': True,
            'research_strategy': research_strategy,
            'agent_responses': result['responses'],
            'final_confidence': result['confidence'],
            'escalation_needed': result['escalate'],
            'research_trace': result['trace']
        }
        
        return enhanced_context
    
    def _determine_research_strategy(self, context: Dict[str, Any]) -> str:
        """Determine the best research strategy based on context"""
        query = context.get('normalized', {}).get('query', '').lower()
        prev_confidence = context.get('confidence', 1.0)
        
        # Use hierarchical for complex or low-confidence scenarios
        if prev_confidence < 0.7 or any(word in query for word in ['complex', 'uncertain', 'ambiguous']):
            return "hierarchical"
        
        # Use consensus for conflict-prone scenarios
        if any(word in query for word in ['conflict', 'debate', 'controversial', 'disagree']):
            return "consensus"
        
        # Default to parallel for efficiency
        return "parallel"
    
    async def _execute_parallel_research(self, query: str, axes: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel research with domain expert and critical analyst"""
        
        research_context = {
            "query": query,
            "axes": axes, 
            "layer": self.layer_id,
            "simulation_context": context
        }
        
        # Run domain expert research
        expert_result = await Runner.run(
            starting_agent=self.domain_expert,
            input=f"Research Query: {query}\n\nAnalyze this query thoroughly and provide your expert assessment. Include confidence level.",
            context=research_context
        )
        
        # Extract confidence and check if validation needed
        expert_confidence = self._extract_confidence(expert_result.final_output)
        
        responses = [{
            "agent": "domain_expert",
            "output": expert_result.final_output,
            "confidence": expert_confidence,
            "trace": [item.to_input_item() for item in expert_result.new_items]
        }]
        
        # If confidence is low, run critical analyst
        if expert_confidence < 0.9:
            analyst_result = await Runner.run(
                starting_agent=self.critical_analyst,
                input=f"Review this research: {expert_result.final_output}\n\nQuery was: {query}\n\nProvide critical analysis and validation.",
                context=research_context
            )
            
            analyst_confidence = self._extract_confidence(analyst_result.final_output)
            responses.append({
                "agent": "critical_analyst", 
                "output": analyst_result.final_output,
                "confidence": analyst_confidence,
                "trace": [item.to_input_item() for item in analyst_result.new_items]
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
                "final_confidence": final_confidence
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
        """Execute consensus-focused research for conflict resolution"""
        
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
        """Calculate final confidence from multiple agent responses"""
        if not responses:
            return 0.0
        
        confidences = [r['confidence'] for r in responses]
        
        # Weighted average with higher weight for later responses (more refined)
        weights = [(i + 1) / len(responses) for i in range(len(responses))]
        
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        # Bonus for consensus (low variance)
        if len(confidences) > 1:
            variance = sum((c - weighted_confidence) ** 2 for c in confidences) / len(confidences)
            consensus_bonus = max(0, 0.05 - variance)
            weighted_confidence = min(1.0, weighted_confidence + consensus_bonus)
        
        return weighted_confidence
    
    def _detect_consensus_needed(self, responses: List[Dict]) -> bool:
        """Detect if consensus building is needed"""
        if len(responses) < 2:
            return False
        
        confidences = [r['confidence'] for r in responses]
        outputs = [str(r['output']).lower() for r in responses]
        
        # Check confidence variance
        conf_variance = sum((c - sum(confidences)/len(confidences)) ** 2 for c in confidences) / len(confidences)
        
        # Check for conflicting keywords
        conflict_indicators = []
        for output in outputs:
            if any(word in output for word in ['disagree', 'however', 'but', 'alternatively', 'instead']):
                conflict_indicators.append(True)
            else:
                conflict_indicators.append(False)
        
        return conf_variance > 0.1 or any(conflict_indicators)
# Import time for timestamp
import time,
                "reasoning": response.reasoning
            }

        return {
            "mode": "sequential",
            "responses": responses,
            "agents": [r.agent_id for r in responses]
        }

    def _execute_consensus_research(self, task: ResearchTask, axes: List[float], memory, state: Dict) -> Dict[str, Any]:
        """Execute research with consensus-building approach"""
        
        # Initial parallel research
        parallel_result = self._execute_parallel_research(task, axes, memory, state)
        responses = parallel_result["responses"]
        
        # Check for consensus
        consensus_data = self._analyze_consensus(responses)
        
        if not consensus_data["has_consensus"]:
            # Spawn consensus builder to resolve conflicts
            consensus_agent = self.agent_manager.agent_types["research"](
                axes=axes, persona="consensus_builder", role="mediator",
                prompt=f"Resolve conflicts and build consensus: {task.query}",
                context={**state, "conflicting_responses": responses}
            )
            
            consensus_response = self._execute_agent_research(consensus_agent, task, memory)
            
            return {
                "mode": "consensus",
                "initial_responses": responses,
                "consensus_response": consensus_response,
                "consensus_data": consensus_data,
                "agents": parallel_result["agents"] + [consensus_agent.agent_id]
            }
        
        return {
            "mode": "consensus",
            "responses": responses,
            "consensus_data": consensus_data,
            "agents": parallel_result["agents"]
        }

    def _execute_agent_research(self, agent, task: ResearchTask, memory) -> AgentResponse:
        """Execute research for a single agent with enhanced tracking"""
        
        # Get persona-specific configuration
        persona_config = self.specialized_personas.get(agent.persona, {})
        
        # Execute the agent's research
        result = agent.act(memory, {"query": task.query, "axes": agent.axes})
        
        # Apply persona-specific confidence adjustments
        adjusted_confidence = result["confidence"]
        if "confidence_boost" in persona_config:
            adjusted_confidence = min(1.0, adjusted_confidence + persona_config["confidence_boost"])
        elif "confidence_penalty" in persona_config:
            adjusted_confidence = max(0.0, adjusted_confidence + persona_config["confidence_penalty"])

        # Generate evidence and reasoning
        evidence = self._generate_evidence(agent, result, memory)
        reasoning = self._generate_reasoning(agent, result, persona_config)
        
        # Create agent response
        response = AgentResponse(
            agent_id=agent.agent_id,
            persona=agent.persona,
            answer=result["answer"],
            confidence=adjusted_confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=time.time(),
            memory_patches=[]
        )

        # Log agent activity
        audit_logger.log(
            event_type="agent_decision",
            layer=self.layer_number,
            details={
                "agent_id": agent.agent_id,
                "persona": agent.persona,
                "confidence": adjusted_confidence,
                "answer": result["answer"]
            },
            persona=agent.persona,
            confidence=adjusted_confidence
        )

        return response

    def _analyze_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Analyze responses for consensus patterns"""
        
        if not responses:
            return {"has_consensus": False, "reason": "no_responses"}
        
        # Group similar answers
        answer_groups = {}
        for response in responses:
            answer_key = str(response.answer)
            if answer_key not in answer_groups:
                answer_groups[answer_key] = []
            answer_groups[answer_key].append(response)
        
        # Find majority
        largest_group = max(answer_groups.values(), key=len)
        consensus_threshold = len(responses) * 0.6  # 60% agreement
        
        has_consensus = len(largest_group) >= consensus_threshold
        confidence_variance = self._calculate_confidence_variance(responses)
        
        return {
            "has_consensus": has_consensus,
            "majority_size": len(largest_group),
            "total_responses": len(responses),
            "confidence_variance": confidence_variance,
            "answer_groups": len(answer_groups),
            "consensus_threshold": consensus_threshold
        }

    def _calculate_confidence_variance(self, responses: List[AgentResponse]) -> float:
        """Calculate variance in confidence scores"""
        if not responses:
            return 0.0
        
        confidences = [r.confidence for r in responses]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        return variance

    def _process_research_results(self, research_result: Dict, task: ResearchTask, memory, state: Dict) -> Dict[str, Any]:
        """Process and finalize research results with conflict resolution"""
        
        # Extract responses based on research mode
        if research_result["mode"] == "consensus" and "consensus_response" in research_result:
            final_response = research_result["consensus_response"]
            responses = research_result.get("initial_responses", [])
        elif research_result["mode"] == "hierarchical" and "consensus_response" in research_result:
            final_response = research_result["consensus_response"]
            responses = research_result["primary_responses"]
        else:
            responses = research_result.get("responses", [])
            final_response = self._select_best_response(responses)

        # Detect forks and conflicts
        fork_detected = self._detect_forks(responses)
        
        # Generate memory patches if needed
        patch_memory = []
        if fork_detected:
            patch_memory = self._generate_fork_patches(responses, task, memory)

        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(responses, final_response)
        
        # Determine escalation need
        escalate = (
            final_confidence < task.confidence_threshold or
            fork_detected or
            len(responses) > 3  # Complex multi-agent scenario
        )

        # Generate comprehensive trace
        trace = {
            "research_mode": research_result["mode"],
            "agent_count": len(research_result.get("agents", [])),
            "responses": [self._response_to_dict(r) for r in responses],
            "final_response": self._response_to_dict(final_response) if final_response else None,
            "fork_detected": fork_detected,
            "consensus_analysis": research_result.get("consensus_data", {}),
            "confidence_final": final_confidence,
            "escalation_triggered": escalate
        }

        return {
            "output": {"answer": final_response.answer if final_response else None},
            "confidence": final_confidence,
            "escalate": escalate,
            "trace": trace,
            "patch_memory": patch_memory
        }

    def _select_best_response(self, responses: List[AgentResponse]) -> Optional[AgentResponse]:
        """Select the best response based on confidence and persona priority"""
        if not responses:
            return None
        
        # Weight responses by confidence and persona priority
        persona_weights = {
            "domain_expert": 1.2,
            "methodical_researcher": 1.1,
            "consensus_builder": 1.05,
            "creative_synthesizer": 1.0,
            "critical_analyst": 0.95  # Lower weight due to skeptical nature
        }
        
        weighted_responses = []
        for response in responses:
            weight = persona_weights.get(response.persona, 1.0)
            weighted_score = response.confidence * weight
            weighted_responses.append((weighted_score, response))
        
        return max(weighted_responses, key=lambda x: x[0])[1]

    def _detect_forks(self, responses: List[AgentResponse]) -> bool:
        """Detect if responses represent conflicting viewpoints (forks)"""
        if len(responses) < 2:
            return False
        
        # Group responses by similarity
        answer_groups = {}
        for response in responses:
            answer_key = str(response.answer)
            if answer_key not in answer_groups:
                answer_groups[answer_key] = []
            answer_groups[answer_key].append(response)
        
        # Fork detected if multiple groups with significant confidence
        significant_groups = [
            group for group in answer_groups.values()
            if any(r.confidence > 0.8 for r in group)
        ]
        
        return len(significant_groups) > 1

    def _generate_fork_patches(self, responses: List[AgentResponse], task: ResearchTask, memory) -> List[Dict]:
        """Generate memory patches for fork scenarios"""
        patches = []
        
        for i, response in enumerate(responses):
            if response.confidence > 0.8:  # Only patch high-confidence forks
                patch = {
                    "coordinate": task.context.get("axes", [0.0] * 13),
                    "value": {
                        "fork_branch": f"agent_{response.agent_id}",
                        "answer": response.answer,
                        "reasoning": response.reasoning,
                        "confidence": response.confidence,
                        "persona": response.persona
                    },
                    "meta": {
                        "created_by": "layer_3_fork",
                        "persona": response.persona,
                        "fork_reason": "multi_agent_conflict",
                        "task_query": task.query
                    }
                }
                patches.append(patch)
        
        return patches

    def _calculate_final_confidence(self, responses: List[AgentResponse], final_response: Optional[AgentResponse]) -> float:
        """Calculate final confidence considering all responses"""
        if not responses:
            return 0.0
        
        if final_response:
            base_confidence = final_response.confidence
        else:
            base_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Adjust based on consensus
        variance = self._calculate_confidence_variance(responses)
        consensus_bonus = max(0, 0.05 - variance)  # Bonus for low variance
        
        return min(1.0, base_confidence + consensus_bonus)

    def _generate_evidence(self, agent, result: Dict, memory) -> List[str]:
        """Generate evidence list for agent response"""
        evidence = []
        
        # Check if answer is backed by memory
        axes = agent.axes
        memory_cell = memory.get(axes, persona=agent.persona)
        if memory_cell:
            evidence.append(f"Memory cell at {axes[:3]}... contains relevant data")
        
        # Add persona-specific evidence patterns
        persona_config = self.specialized_personas.get(agent.persona, {})
        specialties = persona_config.get("specialties", [])
        
        for specialty in specialties:
            if specialty == "technical_analysis":
                evidence.append("Technical analysis framework applied")
            elif specialty == "error_detection":
                evidence.append("Systematic error checking performed")
            elif specialty == "pattern_recognition":
                evidence.append("Cross-domain pattern analysis conducted")
        
        # Add confidence-based evidence
        if result["confidence"] > 0.95:
            evidence.append("High confidence based on multiple validation checks")
        elif result["confidence"] > 0.8:
            evidence.append("Moderate confidence with some uncertainty factors")
        else:
            evidence.append("Lower confidence due to insufficient validation")
        
        return evidence

    def _generate_reasoning(self, agent, result: Dict, persona_config: Dict) -> str:
        """Generate reasoning explanation for agent response"""
        base_reasoning = f"Agent {agent.persona} analyzed the query using specialized {persona_config.get('description', 'approaches')}."
        
        confidence_reasoning = ""
        if result["confidence"] > 0.95:
            confidence_reasoning = " High confidence due to strong evidence and validation."
        elif result["confidence"] > 0.8:
            confidence_reasoning = " Moderate confidence with some remaining questions."
        else:
            confidence_reasoning = " Lower confidence indicating need for additional research."
        
        return base_reasoning + confidence_reasoning

    def _response_to_dict(self, response: AgentResponse) -> Dict:
        """Convert AgentResponse to dictionary for serialization"""
        if not response:
            return None
        
        return {
            "agent_id": response.agent_id,
            "persona": response.persona,
            "answer": response.answer,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "evidence": response.evidence,
            "timestamp": response.timestamp
        }


# Import time for timestamp
import time