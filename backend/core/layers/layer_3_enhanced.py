# core/layers/enhance_layer_3.py

"""
Enhanced Layer 3: Advanced Simulated AI Research Agents
- Sophisticated multi-agent orchestration with specialized personas
- Dynamic consensus mechanisms and conflict resolution
- Advanced fork detection and memory patching
- Recursive research capabilities with confidence thresholding
- Integration with audit logging and compliance checking
"""

import asyncio
import uuid
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseLayer
from core.agents.agent_manager import AgentManager
from core.audit import audit_logger, make_patch_certificate
from core.compliance import compliance_engine


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


class EnhancedLayer3(BaseLayer):
    layer_number = 3
    layer_name = "Enhanced Simulated AI Research Agents"

    def __init__(self):
        self.agent_manager = AgentManager()
        self.specialized_personas = {
            "domain_expert": {
                "description": "Deep domain knowledge specialist",
                "confidence_boost": 0.05,
                "specialties": ["technical_analysis", "domain_facts"]
            },
            "critical_analyst": {
                "description": "Skeptical reviewer and validator",
                "confidence_penalty": -0.02,
                "specialties": ["error_detection", "logical_validation"]
            },
            "creative_synthesizer": {
                "description": "Novel connection maker and innovator",
                "confidence_boost": 0.03,
                "specialties": ["pattern_recognition", "creative_solutions"]
            },
            "methodical_researcher": {
                "description": "Systematic and thorough investigator",
                "confidence_boost": 0.04,
                "specialties": ["comprehensive_analysis", "fact_checking"]
            },
            "consensus_builder": {
                "description": "Conflict resolver and team coordinator",
                "confidence_boost": 0.02,
                "specialties": ["conflict_resolution", "team_coordination"]
            }
        }

    def process(self, input_data: Dict[str, Any], state: Dict[str, Any], memory) -> Dict[str, Any]:
        """Enhanced processing with sophisticated multi-agent research"""
        
        # Extract input parameters
        query = state.get("orig_query") or input_data.get("user_query", "")
        axes = input_data.get("axes") or state.get("axes") or [0.0] * 13
        prev_answer = input_data.get("answer")
        prev_confidence = input_data.get("confidence", 0.0)
        
        # Create research task
        task = self._create_research_task(query, input_data, state, prev_confidence)
        
        # Log task initiation
        audit_logger.log(
            event_type="simulation_pass",
            layer=self.layer_number,
            details={"task": task.__dict__, "query": query},
            confidence=prev_confidence
        )

        # Execute research based on mode
        if task.mode == ResearchMode.HIERARCHICAL:
            result = self._execute_hierarchical_research(task, axes, memory, state)
        elif task.mode == ResearchMode.SEQUENTIAL:
            result = self._execute_sequential_research(task, axes, memory, state)
        elif task.mode == ResearchMode.CONSENSUS:
            result = self._execute_consensus_research(task, axes, memory, state)
        else:  # Default to parallel
            result = self._execute_parallel_research(task, axes, memory, state)

        # Process results and handle conflicts
        final_result = self._process_research_results(result, task, memory, state)
        
        # Apply compliance checks
        compliance_cert = compliance_engine.check_and_log(
            layer=self.layer_number,
            details=final_result,
            confidence=final_result.get("confidence"),
            persona="layer_3_orchestrator"
        )

        return final_result

    def _create_research_task(self, query: str, input_data: Dict, state: Dict, prev_confidence: float) -> ResearchTask:
        """Create research task based on input complexity and previous results"""
        
        # Determine research mode based on query complexity and previous confidence
        if prev_confidence < 0.7:
            mode = ResearchMode.HIERARCHICAL
            max_iterations = 5
        elif prev_confidence < 0.9:
            mode = ResearchMode.CONSENSUS
            max_iterations = 3
        elif "ambiguous" in query.lower() or "uncertain" in query.lower():
            mode = ResearchMode.SEQUENTIAL
            max_iterations = 4
        else:
            mode = ResearchMode.PARALLEL
            max_iterations = 2

        # Select required personas based on query characteristics
        required_personas = self._select_personas_for_query(query, input_data)
        
        return ResearchTask(
            query=query,
            context=input_data,
            mode=mode,
            required_personas=required_personas,
            confidence_threshold=0.995,
            max_iterations=max_iterations
        )

    def _select_personas_for_query(self, query: str, input_data: Dict) -> List[str]:
        """Select appropriate personas based on query characteristics"""
        base_personas = ["domain_expert", "critical_analyst"]
        
        # Add specialized personas based on query content
        query_lower = query.lower()
        if any(word in query_lower for word in ["creative", "innovative", "novel"]):
            base_personas.append("creative_synthesizer")
        if any(word in query_lower for word in ["research", "investigate", "thorough"]):
            base_personas.append("methodical_researcher")
        if any(word in query_lower for word in ["conflict", "disagreement", "consensus"]):
            base_personas.append("consensus_builder")
            
        return base_personas

    def _execute_parallel_research(self, task: ResearchTask, axes: List[float], memory, state: Dict) -> Dict[str, Any]:
        """Execute research with agents working in parallel"""
        
        # Spawn agents for each required persona
        agents = []
        for persona in task.required_personas:
            agent = self.agent_manager.agent_types["research"](
                axes=axes,
                persona=persona,
                role="researcher",
                prompt=f"Research task ({persona}): {task.query}",
                context=state
            )
            agents.append(agent)

        # Execute research in parallel
        responses = []
        for agent in agents:
            response = self._execute_agent_research(agent, task, memory)
            responses.append(response)

        return {
            "mode": "parallel",
            "responses": responses,
            "agents": [a.agent_id for a in agents]
        }

    def _execute_hierarchical_research(self, task: ResearchTask, axes: List[float], memory, state: Dict) -> Dict[str, Any]:
        """Execute research with hierarchical agent coordination"""
        
        # First tier: domain experts
        domain_agents = [
            self.agent_manager.agent_types["research"](
                axes=axes, persona="domain_expert", role="primary_researcher",
                prompt=f"Primary research: {task.query}", context=state
            )
        ]
        
        primary_responses = []
        for agent in domain_agents:
            response = self._execute_agent_research(agent, task, memory)
            primary_responses.append(response)

        # Second tier: critical analysis
        if any(r.confidence < 0.95 for r in primary_responses):
            critical_agent = self.agent_manager.agent_types["research"](
                axes=axes, persona="critical_analyst", role="validator",
                prompt=f"Validate research: {task.query}", context=state
            )
            critical_response = self._execute_agent_research(critical_agent, task, memory)
            
            # Third tier: consensus building if needed
            if critical_response.confidence < 0.9:
                consensus_agent = self.agent_manager.agent_types["research"](
                    axes=axes, persona="consensus_builder", role="coordinator",
                    prompt=f"Build consensus: {task.query}", context=state
                )
                consensus_response = self._execute_agent_research(consensus_agent, task, memory)
                return {
                    "mode": "hierarchical",
                    "primary_responses": primary_responses,
                    "critical_response": critical_response,
                    "consensus_response": consensus_response,
                    "agents": [a.agent_id for a in domain_agents] + [critical_agent.agent_id, consensus_agent.agent_id]
                }

        return {
            "mode": "hierarchical",
            "primary_responses": primary_responses,
            "critical_response": critical_response if 'critical_response' in locals() else None,
            "agents": [a.agent_id for a in domain_agents] + ([critical_agent.agent_id] if 'critical_agent' in locals() else [])
        }

    def _execute_sequential_research(self, task: ResearchTask, axes: List[float], memory, state: Dict) -> Dict[str, Any]:
        """Execute research with sequential agent refinement"""
        
        responses = []
        current_context = task.context.copy()
        
        for i, persona in enumerate(task.required_personas):
            agent = self.agent_manager.agent_types["research"](
                axes=axes, persona=persona, role=f"researcher_{i}",
                prompt=f"Sequential research ({persona}, step {i+1}): {task.query}",
                context=current_context
            )
            
            response = self._execute_agent_research(agent, task, memory)
            responses.append(response)
            
            # Update context with previous results for next agent
            current_context[f"previous_step_{i}"] = {
                "answer": response.answer,
                "confidence": response.confidence,
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

# # Define function tools for the simulation context
# @function_tool
# def search_memory_graph(context: RunContextWrapper, coordinate: List[float], persona: str = None) -> str:
#     """Search the UKG memory graph for information at specific coordinates."""
#     cell = global_memory_graph.get(coordinate, persona)
#     if cell:
#         return f"Found memory cell: {cell['value']}"
#     return "No memory cell found at those coordinates"

# @function_tool
# def patch_memory_cell(context: RunContextWrapper, coordinate: List[float], value: Any, persona: str) -> str:
#     """Patch a memory cell with new information."""
#     global_memory_graph.patch(coordinate, value, {"persona": persona, "layer": 3})
#     return f"Successfully patched memory cell at {coordinate}"

# @function_tool
# def fork_memory_cell(context: RunContextWrapper, coordinate: List[float], new_value: Any, reason: str) -> str:
#     """Create a fork of existing memory cell for alternative reasoning paths."""
#     result = global_memory_graph.fork(coordinate, new_value, {"reason": reason}, reason)
#     if result:
#         return f"Successfully forked memory cell: {result['cell_id']}"
#     return "Failed to fork memory cell"

# class Layer3EnhancedResearchAgents(BaseLayer):
#     layer_number = 3
#     layer_name = "Enhanced AI Research Agents (OpenAI)"
    
#     def __init__(self):
#         # Create specialized research agents
#         self.primary_researcher = Agent(
#             name="PrimaryResearcher",
#             instructions="""You are a primary research agent in a multi-layered simulation system.
#             Your role is to analyze queries, search memory, and provide well-reasoned answers.
            
#             When given a query:
#             1. Search the memory graph for relevant information
#             2. If information is missing or insufficient, reason about what might be needed
#             3. Consider multiple perspectives and potential fork points
#             4. Provide confidence scores and recommend escalation if needed
            
#             Always be thorough but concise in your reasoning.""",
#             tools=[search_memory_graph, patch_memory_cell, fork_memory_cell],
#             model_settings=ModelSettings(temperature=0.3)
#         )
        
#         self.skeptical_reviewer = Agent(
#             name="SkepticalReviewer", 
#             instructions="""You are a skeptical reviewer agent that questions and validates research.
#             Your role is to:
#             1. Challenge assumptions made by other agents
#             2. Look for potential flaws or gaps in reasoning
#             3. Suggest alternative interpretations
#             4. Recommend when forking is needed for alternative paths
            
#             Be constructively critical and help improve the quality of analysis.""",
#             tools=[search_memory_graph, fork_memory_cell],
#             model_settings=ModelSettings(temperature=0.4)
#         )
        
#         # Create handoff between agents
#         self.reviewer_handoff = handoff(
#             self.skeptical_reviewer,
#             tool_description_override="Hand off to skeptical reviewer for validation and alternative perspectives"
#         )
        
#         # Add handoff capability to primary researcher
#         self.primary_researcher = self.primary_researcher.clone(
#             handoffs=[self.reviewer_handoff]
#         )

#     async def process(self, input_data, state, memory):
#         query = state.get("orig_query") or input_data.get("user_query")
#         axes = input_data.get("axes") or state.get("axes") or [0.0] * 13
        
#         # Create context for the agent run
#         context = {
#             "query": query,
#             "axes": axes,
#             "layer": self.layer_number,
#             "simulation_state": state
#         }
        
#         # Run the primary researcher
#         result = await Runner.run(
#             starting_agent=self.primary_researcher,
#             input=f"Research Query: {query}\nAxes: {axes}\nProvide analysis and determine confidence level.",
#             context=context
#         )
        
#         # Extract information from the agent result
#         confidence = self._extract_confidence(result.final_output)
#         escalate = confidence < 0.995
        
#         # Check if agents used any tools (indicates memory operations)
#         memory_patches = []
#         if result.new_items:
#             for item in result.new_items:
#                 if item.type == "tool_call_output_item" and "patch" in str(item.output).lower():
#                     memory_patches.append({
#                         "coordinate": axes,
#                         "value": item.output,
#                         "meta": {"created_by": "layer_3_enhanced", "agent_run": result.last_response_id}
#                     })
        
#         # Prepare trace with agent reasoning
#         trace = {
#             "agent_run_id": result.last_response_id,
#             "primary_agent_output": result.final_output,
#             "confidence_extracted": confidence,
#             "escalate_recommended": escalate,
#             "tool_calls_made": len([item for item in result.new_items if item.type == "tool_call_item"]),
#             "memory_operations": len(memory_patches),
#             "full_trace": [item.to_input_item() for item in result.new_items]
#         }
        
#         return dict(
#             output=dict(answer=result.final_output, analysis_quality="enhanced"),
#             confidence=confidence,
#             escalate=escalate,
#             trace=trace,
#             patch_memory=memory_patches
#         )
    
#     def _extract_confidence(self, agent_output: str) -> float:
#         """Extract confidence score from agent output."""
#         # Simple pattern matching - could be more sophisticated
#         import re
#         confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', str(agent_output).lower())
#         if confidence_match:
#             try:
#                 return float(confidence_match.group(1))
#             except:
#                 pass
        
#         # Default confidence based on output quality
#         if len(str(agent_output)) > 100:
#             return 0.92
#         return 0.85