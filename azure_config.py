
# middleware/agents/azure_config.py

import os
import logging
from dataclasses import dataclass
from openai import AsyncAzureOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled, set_tracing_export_api_key

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI deployments."""
    api_version: str = "2025-03-01-preview"
    programmable_deployment: str = "gpt-4.1"
    sub_agent_deployment: str = "gpt-4.1"
    
    def __post_init__(self):
        # Load from environment if provided
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.api_version)
        self.programmable_deployment = os.getenv("AZURE_PROGRAMMABLE_DEPLOYMENT", self.programmable_deployment)
        self.sub_agent_deployment = os.getenv("AZURE_SUB_AGENT_DEPLOYMENT", self.sub_agent_deployment)

# Global configuration instance
azure_config = AzureConfig()

# Configure Azure OpenAI client once
def setup_azure_openai():
    """Set up Azure OpenAI client for the Agents SDK."""
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            # Create the Azure client with the updated API version
            # IMPORTANT: Include azure_deployment parameter
            azure_client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=azure_config.api_version,
                # Each deployment in Azure is set up to use a specific model
                # So we provide the deployment name here, not in the 'model' parameter later
                azure_deployment=azure_config.programmable_deployment
            )
            # Set as default client
            set_default_openai_client(azure_client)
            
            # IMPORTANT: Use chat.completions API since responses API isn't supported
            set_default_openai_api("chat_completions")

            if not os.getenv("OPENAI_API_KEY"):
                set_tracing_disabled(True)
                logger.info("Tracing disabled - no OpenAI API ley found")
            else:
                # Set API key for tracing if available 
                set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))
                logger.info("Tracing enabled with OpenAI API key")
            
            logger.info(f"Azure OpenAI client configured successfully with deployment '{azure_config.programmable_deployment}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Azure OpenAI client: {e}")
            return False
    else:
        logger.warning("Azure OpenAI credentials not found in environment variables")
        return False

# Initialize on module import
setup_azure_openai()