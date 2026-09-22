from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
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

MODEL_NAME = "gemini-2.5-flash"
GENERATION_CONFIG = types.GenerateContentConfig(
temperature=0.0,
response_mime_type="application/json",
)
eval_single_agent = LlmAgent(
    name="LandmarkQualityEvaluator",
    model=MODEL_NAME,
    output_schema=SingleAgentOutput,
    instruction=instruction_single_agent_prompt,
    output_key="quality_verdict",
    generate_content_config=GENERATION_CONFIG
)

oracle_error_descriptor_agent = LlmAgent(
    name="OracleErrorDescriptor",
    model=MODEL_NAME,
    output_schema=OracleErrorDescriptionOutput,
    instruction=instruction_error_description_agent_prompt,
    generate_content_config=GENERATION_CONFIG
)

oracle_profile_builder_agent = LlmAgent(
    name="OracleProfileBuilder",
    model=MODEL_NAME,
    output_schema=OracleProfileOutput,
    instruction=instruction_profile_builder_agent_prompt,
    generate_content_config=GENERATION_CONFIG
)
final_decision_agent = LlmAgent(
    name="FinalDecisionAgent",
    model=MODEL_NAME,
    output_schema=FinalDecisionOutput,
    instruction=instruction_final_decision_agent_prompt,
    generate_content_config=GENERATION_CONFIG)