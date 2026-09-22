from google.adk.agents.llm_agent import LlmAgent
from src.schemas import( 
    SingleAgentOutput,
    OracleErrorDescriptionOutput,
    OracleProfileOutput,
    FinalDecisionOutput)
from src.prompts import ( 
    instruction_single_agent_prompt,
    instruction_error_description_agent_prompt,
    instruction_profile_builder_agent_prompt,
    instruction_final_decision_agent_prompt)


eval_single_agent = LlmAgent(
    name="LandmarkQualityEvaluator",
    model="gemini-2.5-flash",
    output_schema=SingleAgentOutput,
    instruction=instruction_single_agent_prompt,
    output_key="quality_verdict",
)

oracle_error_descriptor_agent = LlmAgent(
    name="OracleErrorDescriptor",
    model="gemini-2.5-flash",
    output_schema=OracleErrorDescriptionOutput,
    instruction=instruction_error_description_agent_prompt
)

oracle_profile_builder_agent = LlmAgent(
    name="OracleProfileBuilder",
    model="gemini-2.5-flash",
    output_schema=OracleProfileOutput,
    instruction=instruction_profile_builder_agent_prompt
)
final_decision_agent = LlmAgent(
    name="FinalDecisionAgent",
    model="gemini-2.5-flash",
    output_schema=FinalDecisionOutput,
    instruction=instruction_final_decision_agent_prompt)