#!/usr/bin/env python3
"""
Improved Chat Architecture for NDFC Assistant
---------------------------------------------

This module implements a more flexible and LLM-centric chat architecture that:
1. Uses Ollama for all response generation (no hardcoded responses)
2. Maintains a clear separation between:
   - Enhanced intent recognition (pattern, keyword, similarity, optional LLM-based)
   - Robust API handling with retries, caching, and endpoint prioritization
   - Advanced context tracking with reference resolution
   - Multi-turn dialog management
   - Troubleshooting recommendations and explanations
   - Response generation with hallucination detection
   - Entity tracking and context resolution
   - Follow-up question handling

This ensures the chatbot provides appropriate, real-time information rather than
defaulting to static knowledge base entries, while maintaining context awareness
across multi-turn conversations.
"""

import os
import sys
import re
import json
import yaml
import time
import logging
import dateparser
from datetime import datetime, timedelta
import requests
import traceback
from conversation_context import ConversationContext
import copy
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
from conversation_context import ConversationContext
from context_objects import HandlerContext
from cisco_api.nexus.dashboard_client import NexusDashboardClient

# Import formatter utilities directly to avoid circular imports
import importlib.util
import sys

# Import LLM processor for response generation
from llm_integration import LLMProcessor, PROVIDER_OLLAMA, DEFAULT_SYSTEM_PROMPT

# Use direct import path to avoid circular imports through __init__.py
llm_formatter_path = os.path.join(os.path.dirname(__file__), 'agents', 'llm_formatter_connector.py')
spec = importlib.util.spec_from_file_location("llm_formatter_module", llm_formatter_path)
llm_formatter_module = importlib.util.module_from_spec(spec)
sys.modules["llm_formatter_module"] = llm_formatter_module
spec.loader.exec_module(llm_formatter_module)
llm_formatter = llm_formatter_module.llm_formatter  # Get the formatter instance

# Helper function to add conversational elements to responses
def add_conversational_elements(message, include_opener=True):
    """
    Add conversational elements to a response message to make it more human-like
    
    Args:
        message: The core message to enhance
        include_opener: Whether to include an opening phrase
        
    Returns:
        Enhanced message with conversational elements
    """
    # Delegate to the llm_formatter to enhance the response
    return llm_formatter.enhance_response(message)

from cisco_api.nexus.dashboard_client import NexusDashboardClient
from llm_integration import LLMProcessor, PROVIDER_OLLAMA, PROVIDER_NONE

from agents.verified_endpoints import (
    ND_ENDPOINTS,
    NDI_ENDPOINTS,
    ND_VALIDATED_ENDPOINTS, 
    NDFC_VALIDATED_ENDPOINTS, 
    NDI_VALIDATED_ENDPOINTS, 
    VALIDATED_ENDPOINTS,
    get_nd_endpoint,
    get_ndi_endpoint,
    get_all_ndi_endpoints
)

# Import structured models for typed responses
from structured_models import (
    FabricHealthStatus,
    NetworkSummary,
    TroubleshootingPlan,
    NetworkIssue,
    IntentClassification
)

# Import formatters for different data types
from formatters import (
    format_ndi_endpoints, format_ndi_interfaces, 
    format_ndi_routes, format_ndi_events
)

# Import the function to get NDFC endpoints

# Import the centralized logging configuration
try:
    from logging_config import get_logger, get_api_logger, get_endpoint_logger, get_param_logger
    logger = get_logger(__name__)
    api_logger = get_api_logger()
    endpoint_logger = get_endpoint_logger()
    param_logger = get_param_logger()
    logger.info("Using centralized logging configuration")
except ImportError:
    # Fallback to basic logging if the module isn't available yet
    import os
    import logging
    
    # Get log level from environment or default to INFO
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Configure detailed logging format
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create separate loggers for different components
    api_logger = logging.getLogger("api_calls")
    endpoint_logger = logging.getLogger("endpoints")
    param_logger = logging.getLogger("param_substitution")
    
    # Set log levels based on environment variables
    api_debug = os.environ.get("DEBUG_API_CALLS", "false").lower() in ("true", "1", "yes")
    api_logger.setLevel(logging.DEBUG if api_debug else logging.INFO)
    endpoint_logger.setLevel(logging.DEBUG)
    param_logger.setLevel(logging.DEBUG)
    
    logger.info("Using fallback logging configuration")

# Helper function to log API requests and responses
def log_api_call(method, url, headers=None, data=None, response=None, error=None, params=None, original_url=None):
    """Log detailed API call information for debugging"""
    try:
        # Use the centralized api_logger
        api_logger.debug("="*80)
        api_logger.debug(f"API {method} REQUEST: {url}")
        
        # Log parameter substitution if applicable
        if original_url and original_url != url:
            api_logger.debug(f"Original URL before parameter substitution: {original_url}")
            param_logger.debug(f"Parameter substitution: {original_url} -> {url}")
        
        # Log parameters passed to the function if any
        if params:
            param_logger.debug(f"Parameters provided for request: {params}")
            
            # Check if any parameters might be missing from the URL
            for param_name, param_value in params.items():
                if '{' + param_name + '}' in url:
                    param_logger.warning(f"Parameter '{param_name}' was provided but not substituted in URL: {url}")
        
        # Check for any remaining template parameters in the URL
        import re
        template_params = re.findall(r'\{([^\}]+)\}', url)
        if template_params:
            param_logger.warning(f"URL contains unsubstituted template parameters: {template_params}")
            param_logger.warning(f"This may cause a 404 error for URL: {url}")
        
        # Log request details
        if headers:
            # Sanitize headers to remove sensitive information
            safe_headers = {k: '***REDACTED***' if k.lower() in ('authorization', 'x-auth-token', 'cookie') else v 
                           for k, v in headers.items()}
            api_logger.debug(f"Headers: {safe_headers}")
        
        if data:
            api_logger.debug(f"Request Data: {data}")
        
        # Log response or error
        if response:
            api_logger.debug(f"Response Status: {response.status_code}")
            
            # Log detailed information for error responses
            if response.status_code >= 400:
                api_logger.error(f"API Error: {method} {url} returned status {response.status_code}")
                endpoint_logger.error(f"Endpoint error: {url} returned {response.status_code}")
                
                # For common error codes, provide more specific logging
                if response.status_code == 404:
                    api_logger.error(f"404 Not Found - Endpoint does not exist or parameters are incorrect: {url}")
                elif response.status_code == 405:
                    api_logger.error(f"405 Method Not Allowed - Method {method} not supported for endpoint: {url}")
                elif response.status_code == 401 or response.status_code == 403:
                    api_logger.error(f"Authentication/Authorization error ({response.status_code}): {url}")
            else:
                api_logger.info(f"Successful API call: {method} {url} returned {response.status_code}")
            
            try:
                # Try to log response as JSON
                resp_content = response.json()
                # Truncate large responses
                if isinstance(resp_content, dict) and len(str(resp_content)) > 1000:
                    api_logger.debug(f"Response JSON (truncated): {str(resp_content)[:1000]}...")
                else:
                    api_logger.debug(f"Response JSON: {resp_content}")
            except Exception as e:
                # If not JSON, log as text (truncated)
                content = response.text[:500] + '...' if len(response.text) > 500 else response.text
                api_logger.debug(f"Response Text: {content}")
                api_logger.debug(f"Non-JSON response. Error: {str(e)}")
        
        if error:
            api_logger.error(f"API Call Error: {error}")
            if hasattr(error, 'traceback'):
                api_logger.debug(f"Error Traceback: {error.traceback}")
        
        api_logger.debug("="*80)
    except Exception as e:
        # Fallback if the enhanced logging itself fails
        logging.error(f"Error in log_api_call: {str(e)}")
        # Ensure we still log the basic API call information
        logging.debug(f"API {method} REQUEST: {url}, Status: {response.status_code if response else 'N/A'}")
        if error:
            logging.error(f"Original API Error: {error}")
        logging.debug("="*80)

class ImprovedChatArchitecture:
    """
    Improved architecture for the NDFC Assistant chatbot that uses a more
    flexible, LLM-centric approach to handling user queries.
    """
    
    def __init__(self, nd_client=None, llm_processor=None):
        """Initialize the improved chat architecture"""
        # Initialize LLM processor for generating responses
        if llm_processor:
            self.llm_processor = llm_processor
        else:
            self.llm_processor = LLMProcessor(
                provider=PROVIDER_OLLAMA,
                ollama_model="llama3.2",
                ollama_base_url="http://localhost:11434"
            )
        
        # Initialize the client (use passed client or set to None)
        self.nd_client = nd_client
        
        # Track conversation context
        self.conversation_history = []
        
        # Enhanced context tracking using ConversationContext class
        self.context = ConversationContext()
        
        # Define intents and their data retrieval functions
        # Use centralized INTENT_REGISTRY for all intent routing
        from agents.intent_registry import INTENT_REGISTRY

        # Intent recognition patterns (simplified for now)
        # Legacy: self.intent_patterns = {...} (keep for pattern matching if needed)
        # All handler lookups now use INTENT_REGISTRY
        self.intent_patterns = {
            # ND/NDFC intent patterns
            # IMPORTANT: fabric_name must be checked before fabric_devices since it's more specific
            "fabric_name": [
                # Very specific patterns that should get highest priority
                "what is my fabric called",  # Exact match for problematic query
                "what's the name of my fabric", 
                "fabric name", 
                "name of fabric",
                "name of my fabric",
                "what is the fabric name",
                "what's my fabric name", 
                "tell me the fabric name", 
                "show fabric name",
                "fabric identifier", 
                "my fabric is called", 
                "fabric's name",
                "name that my fabric has",
                "what fabric am I on", 
                "which fabric", 
                "fabric I'm connected to",
                "current fabric", 
                "active fabric name", 
                "fabric label"
            ],
            "fabric_devices": [
                # More specific phrases about devices
                "devices in fabric", 
                "fabric devices", 
                "switches in fabric",
                "switches on fabric",
                "list fabric devices", 
                "show fabric devices",
                "nodes in fabric", 
                "fabric nodes", 
                "inventory of fabric",
                "what are my devices", 
                "show devices", 
                "list devices",
                "all devices", 
                "connected devices", 
                "show my devices", 
                "device list",
                "show me devices", 
                "devices on my fabric", 
                "devices on my network",
                "what devices are on my network", 
                "show me devices on my fabric",
                "devices on fabric", 
                "network devices", 
                "fabric inventory"
            ],
            "fabric_health": [
                "fabric health", "health of fabric", "fabric status",
                "how is my fabric", "fabric condition", "fabric issues",
                "show my fabric", "show fabric", "fabric status", "my fabric",
                "fabric overview", "fabric summary", "is my fabric", "fabric details"
            ],
            "network_topology": [
                "network topology", "fabric topology", "show topology",
                "network map", "fabric structure", "topology view",
                "how is my network connected", "network structure", "show network map"
            ],
            
            # NDI-specific intent patterns for analytics and insights with enhanced specificity
            "anomaly_summary": [
                "anomaly summary", "summarize anomalies", "summary of anomalies", 
                "anomalies overview", "general anomalies", "show anomaly summary",
                "anomaly stats", "network anomaly statistics", "anomaly counts",
                "high level anomalies", "anomaly report summary", "anomaly status"
            ],
            "anomaly_details": [
                "anomaly details", "detailed anomalies", "show anomaly details",
                "specific anomalies", "anomaly specifics", "all details about anomalies",
                "in-depth anomaly information", "complete anomaly data", "anomaly full report",
                "show me anomaly records", "detailed anomaly logs", "anomaly descriptions"
            ],
            "critical_anomalies": [
                "critical anomalies", "severe anomalies", "high severity anomalies",
                "serious network issues", "critical network problems", "show critical anomalies",
                "critical alerts", "major alerts", "severe warnings", "high priority issues",
                "critical network alerts", "most severe anomalies", "critical errors"
            ],
            "fabric_anomalies": [
                "fabric anomalies", "anomalies on fabric", "fabric issues", 
                "fabric network problems", "show fabric anomalies", "anomalies in my fabric",
                "fabric alerts", "fabric warnings", "problems with my fabric",
                "fabric health issues", "fabric errors", "network fabric anomalies",
                "detect fabric anomalies", "issues on fabric", "what's wrong with my fabric"
            ],
            "endpoint_anomalies": [
                "endpoint anomalies", "device anomalies", "anomalies on endpoints", 
                "endpoint issues", "device issues", "show endpoint anomalies",
                "problems with endpoints", "endpoint alerts", "device warnings", 
                "endpoint health issues", "endpoint errors", "anomalies for devices",
                "anomalies on devices", "device problems", "endpoint troubleshooting"
            ],
            "recent_anomalies": [
                "recent anomalies", "latest anomalies", "new anomalies", 
                "fresh anomalies", "just detected anomalies", "show recent anomalies",
                "anomalies in the last hour", "today's anomalies", "current anomalies", 
                "anomalies right now", "just happened anomalies", "latest issues",
                "recent network issues", "latest problems", "newest alerts"
            ],
            "anomaly_trends": [
                "anomaly trends", "anomaly patterns", "trending issues", 
                "recurring anomalies", "anomaly frequency", "common anomalies",
                "anomaly history", "historical anomalies", "anomaly statistics",
                "most common anomalies", "anomaly hot spots", "problem trends"
            ],
            "advisories": [
                "advisories", "show advisories", "security advisories", "what advisories",
                "list advisories", "software advisories", "bug advisories", "defects",
                "known bugs", "show defects", "cisco advisories", "security alerts"
            ],
            "compliance": [
                "compliance", "compliance status", "compliance check", "compliance summary",
                "show compliance", "check compliance", "compliance requirements", "compliance results",
                "am I compliant", "network compliance", "configuration compliance",
                "compliance issues", "compliance errors", "compliance reports"
            ],
            "flow_rules": [
                "flow rules", "show flow rules", "list flow rules", "get flow rules",
                "flow configuration", "flow analytics rules", "traffic flow rules",
                "configured flow rules", "flow telemetry rules", "flow collection rules",
                "how are flows configured", "flow monitoring configuration"
            ],
            
            # New intent patterns for network visualization data
            "endpoints": [
                "endpoints", "show endpoints", "list endpoints", "network endpoints",
                "connected endpoints", "endpoint status", "endpoint health", 
                "endpoint anomalies", "anomaly level", "endpoint inventory",
                "mac addresses", "connected devices", "show connected devices",
                "what devices are connected", "what endpoints do I have"
            ],
            "interfaces": [
                "interfaces", "show interfaces", "list interfaces", "network interfaces",
                "interface status", "interface health", "down interfaces",
                "interface anomalies", "admin status", "operational status",
                "interface inventory", "switch interfaces", "port status",
                "which interfaces are down", "interface overview"
            ],
            "routes": [
                "routes", "show routes", "list routes", "network routes",
                "routing table", "route summary", "routes per vrf", "vrf routes",
                "routing protocols", "static routes", "dynamic routes",
                "ip routes", "show ip routes", "routing information", "route details"
            ],
            "traffic_analysis": [
                "traffic analysis", "network traffic", "traffic patterns", "analyze traffic",
                "show traffic", "traffic flow", "traffic overview", "traffic report",
                "bandwidth usage", "traffic statistics", "traffic metrics", "traffic trends"
            ],
            "top_talkers": [
                "top talkers", "show top talkers", "busiest devices", "heaviest traffic",
                "bandwidth hogs", "traffic sources", "traffic destinations", "high bandwidth users",
                "traffic hot spots", "bandwidth consumers", "most active endpoints"
            ],
            "interface_errors": [
                "interface errors", "port errors", "errors on interfaces",
                "interface issues", "bad ports", "interface problems"
            ],
            "system_info": [
                "system information", "system info", "ndfc info",
                "version information", "system details", "system status"
            ],
            "alarm_summary": [
                "alarm summary", "alarms", "show alarms",
                "alert summary", "alerts", "active alarms"
            ],
            "event_history": [
                "event history", "recent events", "show events",
                "what happened", "event log", "system events"
            ],
            "events": [
                "network events", "show network events", "list events", "audit logs",
                "event timeline", "system events", "fault events", "event buckets",
                "event summary", "event details", "events report", "audit trail",
                "activity logs", "what events occurred", "recent network events"
            ]
        }
        
        # Initialize with safe defaults
        self.context = ConversationContext()  # Always use ConversationContext, not a dict
        # Defensive: if any code later sets self.context to a dict, convert it back in property setter or before use.
        self._api_cache = {}
        
        # Initialize intent patterns from YAML files
        self._initialize_intent_patterns()
        
        logger.info("Improved chat architecture initialized")
    
    def _initialize_intent_patterns(self):
        """
        Initialize intent patterns from YAML files
        """
        try:
            from intent_loader import load_intents
            from agents.intent_registry import get_intent_metadata
            
            # Load intents from YAML files
            logger.info("Loading intents from YAML files")
            loaded_intents = load_intents()
            
            # Create intent patterns dictionary
            self.intent_patterns = {}
            self.intent_priorities = {}            # Generate pattern lists for each intent
            for intent_name, intent_spec in loaded_intents.items():
                patterns = []
                
                # Store priority if defined in YAML
                if hasattr(intent_spec, 'priority') and intent_spec.priority > 0:
                    self.intent_priorities[intent_name] = intent_spec.priority
                    logger.info(f"Intent '{intent_name}' has custom priority: {intent_spec.priority}")
                
                # Use custom patterns if defined in YAML
                if hasattr(intent_spec, 'patterns') and intent_spec.patterns:
                    patterns.extend([p.lower() for p in intent_spec.patterns])
                    logger.info(f"Intent '{intent_name}' has {len(intent_spec.patterns)} custom patterns from YAML")
                else:
                    # Generate patterns from description if no custom patterns
                    # Get metadata from intent registry
                    metadata = get_intent_metadata(intent_name) or {}
                
                # Add patterns from description and metadata
                description = intent_spec.description.lower()
                if description:
                    # Add full description as a pattern
                    patterns.append(description)
                    
                    # Add key phrases from description
                    words = description.split()
                    if len(words) > 2:
                        patterns.append(" ".join(words[:3]))
                
                # Add intent name variations
                name_parts = intent_name.split('_')
                patterns.append(intent_name.replace('_', ' '))
                patterns.append(' '.join(name_parts))
                
                # Add common question patterns
                patterns.append(f"show {intent_name.replace('_', ' ')}")
                patterns.append(f"get {intent_name.replace('_', ' ')}")
                patterns.append(f"what is {intent_name.replace('_', ' ')}")
                
                # Store unique patterns
                self.intent_patterns[intent_name] = list(set(patterns))
                
            logger.info(f"Loaded {len(self.intent_patterns)} intent patterns")
            
        except Exception as e:
            logger.error(f"Failed to initialize intent patterns: {str(e)}")
            logger.error(traceback.format_exc())
            # Set fallback patterns
            self.intent_patterns = {
                "fabric_devices": ["fabric devices", "devices in fabric", "show devices", "list devices"],
                "device_status": ["device status", "status of device", "switch status"],
                "network_topology": ["topology", "network map", "infrastructure", "network layout"]
            }
    
    def set_credentials(self, credentials: Dict[str, Any]) -> bool:
        """
        Set the authentication credentials for the Nexus Dashboard client
        
        Args:
            credentials: Dictionary with authentication credentials
                         Should contain: api_host, username, password, verify_ssl
        
        Returns:
            bool: True if authentication successful
        """
        try:
            # Try to get cached token first if available
            cached_token = None
            try:
                from web_chat import ndi_cache
                cached_token = ndi_cache.get_token()
                if cached_token:
                    logger.info("Found cached authentication token, will attempt to use it")
            except ImportError:
                logger.warning("NDI cache not available, will perform full authentication")
                
            # Ensure we have properly formatted base_url
            base_url = credentials.get('api_host') or credentials.get('base_url') or credentials.get('ndUrl') or credentials.get('url')
            if not base_url:
                logger.error("No API host URL provided in credentials")
                return False
                
            # Log credential attempt but obscure sensitive info
            logger.info(f"Setting credentials for API host: {base_url}")
            logger.info(f"Credentials contain username: {bool(credentials.get('username'))}, password: {bool(credentials.get('password'))}, api_key: {bool(credentials.get('api_key'))}")
            
            # Create a clean ND client instance with API settings
            self.nd_client = NexusDashboardClient(
                base_url=base_url,
                verify_ssl=credentials.get('verify_ssl', False),
                auto_detect=True
            )
            
            # Save credentials for re-authentication if needed - clean and normalized
            self._username = credentials.get('username', '')
            self._password = credentials.get('password', '')
            self._api_key = credentials.get('api_key', '')
            self._base_url = base_url
            
            # Reset authentication failure counter
            self._auth_failures = 0
                
            # Initialize API cache if not already done
            if not hasattr(self, "_api_cache"):
                self._api_cache = {}
                
            auth_success = False
            
            # Try using cached token first if available
            if cached_token:
                logger.info("Setting cached bearer token in NDI client")
                self.nd_client.token = cached_token
                # Set token in session headers
                self.nd_client._session.headers.update({"Authorization": f"Bearer {cached_token}"})
                
                # Validate token with a simple GET request to verify it's still valid
                try:
                    logger.info("Testing cached token validity with a simple request...")
                    # Try a simple endpoint that requires authentication - use Nexus Dashboard endpoint
                    # Different endpoints for different ND versions
                    endpoints_to_try = [
                        "/nexus/infra/api/platform/v1/nodes",  # Standard ND endpoint for node list
                        "/api/v1/platform/status",            # General platform status
                        "/login"                             # Login endpoint (should at least be accessible)
                    ]
                    
                    token_valid = False
                    for endpoint in endpoints_to_try:
                        try:
                            logger.info(f"Trying endpoint {endpoint} to validate token")
                            test_resp = self.nd_client.get(endpoint, timeout=3)
                            if test_resp and test_resp.status_code in (200, 201, 202):
                                logger.info(f"✅ Cached token is valid, verified with {endpoint}")
                                token_valid = True
                                break
                            else:
                                logger.warning(f"Token validation failed for {endpoint}: {test_resp.status_code}")
                        except Exception as ep_err:
                            logger.warning(f"Error testing endpoint {endpoint}: {str(ep_err)}")
                    
                    if token_valid:
                        auth_success = True
                        # Make sure auth_success is correctly reflected in the client
                        self.nd_client.auth_success = True
                    else:
                        logger.warning("Cached token is invalid or expired, will perform full authentication")
                except Exception as e:
                    logger.warning(f"Error testing cached token: {str(e)}. Will perform full authentication.")
            
            # If token not available or not valid, do full authentication
            if not auth_success:
                # Build credentials object with ALL possible formats to ensure compatibility
                auth_creds = {
                    'username': self._username,
                    'password': self._password,
                    'userName': self._username,  # Include ND-specific format
                    'userPasswd': self._password,  # Include ND-specific format
                    'api_key': self._api_key
                }
                
                # Attempt authentication with credentials
                logger.info("Attempting full authentication with credentials")
                if self._api_key:
                    auth_success = self.nd_client.authenticate_with_credentials(auth_creds)
                elif self._username and self._password:
                    # Try direct authentication with built-in method first
                    auth_success = self.nd_client.authenticate(self._username, self._password)
                    
                    if not auth_success:
                        # Fallback to credentials dict method
                        logger.info("Direct authentication failed, trying with credentials dictionary")
                        auth_success = self.nd_client.authenticate_with_credentials(auth_creds)
                else:
                    logger.error("No valid authentication credentials provided")
                    return False
                    
                # If authentication was successful, update the cache
                if auth_success and hasattr(self.nd_client, 'token') and self.nd_client.token:
                    try:
                        from web_chat import ndi_cache
                        ndi_cache.set_token(self.nd_client.token)
                        logger.info(f"✅ Updated token cache with new authentication token: {self.nd_client.token[:10]}...")
                        
                        # Store this token directly in the instance for easier access
                        self._token = self.nd_client.token
                        
                        # Make this token available to other components
                        import os
                        token_path = os.path.join(os.path.dirname(__file__), 'nd_token.txt')
                        try:
                            with open(token_path, 'w') as f:
                                f.write(self.nd_client.token)
                            logger.info(f"✅ Saved token to {token_path} for other components")
                        except Exception as file_err:
                            logger.error(f"Error saving token to file: {str(file_err)}")
                            
                    except ImportError:
                        logger.warning("Could not update token cache (NDI cache not available)")
                    except Exception as e:
                        logger.error(f"Error updating token cache: {str(e)}")
                
            if auth_success:
                logger.info(f"[ImprovedChatArchitecture] Authentication successful for {self._username} at {base_url}")
                # Log when the token will expire (if known)
                if hasattr(self.nd_client, 'token_expiry') and self.nd_client.token_expiry:
                    logger.info(f"Token will expire at: {self.nd_client.token_expiry}")
                return True
            else:
                logger.error(f"[ImprovedChatArchitecture] Authentication failed for {self._username} at {base_url}")
                return False
        except Exception as e:
            logger.error(f"[ImprovedChatArchitecture] Error setting credentials: {str(e)}")
            return False
    
    def process_message(self, message: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user message and generate a response
        
        Args:
            message: User message
            credentials: Optional credentials to use (will update existing if provided)
            
        Returns:
            Response dictionary with generated response
        """
        # Check if we need to authenticate or re-authenticate
        if not hasattr(self, 'nd_client') or self.nd_client is None:
            logger.warning("No ND client available, attempting to create one with cached token")
            
            # Try to get token from file first (most reliable)
            import os
            token_path = os.path.join(os.path.dirname(__file__), 'nd_token.txt')
            token = None
            
            if os.path.exists(token_path):
                try:
                    with open(token_path, 'r') as f:
                        token = f.read().strip()
                    logger.info(f"✅ Retrieved token from {token_path}")
                except Exception as e:
                    logger.error(f"Error reading token from file: {str(e)}")
            
            # If no token from file, try the cache
            if not token:
                try:
                    from web_chat import ndi_cache
                    token = ndi_cache.get_token()
                    if token:
                        logger.info("✅ Retrieved token from NDI cache")
                except Exception as e:
                    logger.error(f"Error getting token from cache: {str(e)}")
                    
            # If we have a token, create a client and use it
            if token:
                try:
                    from cisco_api.nexus.dashboard_client import NexusDashboardClient
                    self.nd_client = NexusDashboardClient(
                        base_url="https://10.6.11.10",  # Default URL
                        verify_ssl=False
                    )
                    self.nd_client.token = token
                    self.nd_client._session.headers.update({"Authorization": f"Bearer {token}"})
                    self.nd_client.auth_success = True
                    logger.info("✅ Created new ND client with cached token")
                except Exception as e:
                    logger.error(f"Error creating client with token: {str(e)}")
        
        # Continue with regular processing
        try:
            # Log incoming message
            logger.info(f"Processing user message: '{message}'")
            
            # Check if credentials are provided - update if needed
            if credentials:
                logger.info("Credentials provided with message - updating client")
                if not hasattr(self, 'nd_client') or self.nd_client is None:
                    logger.info("Initializing client with provided credentials")
                    self.set_credentials(credentials)
                else:
                    logger.info("Client already initialized - using existing session")
            
            # Ensure we have a client connection
            if not hasattr(self, 'nd_client') or self.nd_client is None:
                error_msg = "No connection to Nexus Dashboard. Please set credentials first."
                logger.error(error_msg)
                return {
                    "response": error_msg,
                    "success": False,
                    "intent": "unknown"
                }
            
            # Test the connection with a simple API call
            try:
                test_endpoint = "/nexus/infra/api/platform/v1/nodes"  # Reliable endpoint
                logger.debug(f"Testing API connection with endpoint: {test_endpoint}")
                
                # Detailed inspection of client state before making the call
                logger.info("=== DEBUG AUTH INFO ===")
                logger.info(f"Has client: {hasattr(self, 'nd_client') and self.nd_client is not None}")
                
                if hasattr(self, 'nd_client') and self.nd_client is not None:
                    # Force token refresh if needed - this helps with token expiry issues
                    if hasattr(self.nd_client, '_ensure_valid_token'):
                        self.nd_client._ensure_valid_token()
                    
                    test_response = self.nd_client.get(test_endpoint)
                    
                    # Reset auth failure counter if successful
                    if test_response and getattr(test_response, 'status_code', 500) == 200:
                        logger.info("API connection test successful")
                        # Reset auth failure counter on success
                        self._auth_failures = 0
                    else:
                        status_code = getattr(test_response, 'status_code', 'unknown')
                        logger.warning(f"API connection test failed with status code: {status_code}")
                        
                        # Increment auth failure counter
                        self._auth_failures += 1
                        
                        # Attempt re-authentication if we have multiple failures
                        if self._auth_failures >= 3 and hasattr(self, '_username') and hasattr(self, '_password'):
                            logger.warning(f"Multiple auth failures detected ({self._auth_failures}), trying to re-authenticate")
                            if self.nd_client.authenticate(self._username, self._password):
                                logger.info("Re-authentication successful")
                                self._auth_failures = 0
                
                logger.error(f"Detailed exception info: {traceback.format_exc()}")
                logger.info("=== END DEBUG AUTH INFO ===")
            except Exception as e:
                logger.error(f"API connection test failed: {str(e)}")
                logger.error(f"Detailed exception info: {traceback.format_exc()}")
                logger.info("=== END DEBUG AUTH INFO ===")

            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Resolve any references in the message (like "it", "that device", etc.)
            original_message = message
            resolved_message = self._resolve_references(message)
            if resolved_message != original_message:
                logger.info(f"Resolved references in message: '{original_message}' -> '{resolved_message}'")
                message = resolved_message
            
            # Check if this is a general question or greeting
            if self._is_general_query(message):
                response = self._handle_general_query(message)
                self.conversation_history.append({"role": "assistant", "content": response})
                return {
                    "response": response,
                    "success": True,
                    "intent": "general_query",
                    "confidence": 1.0
                }
            
            # First, recognize the primary intent using the enhanced intent recognition
            intent_results = self._enhanced_intent_recognition(message)
            logger.info(f"[INTENT RECOGNITION] User message: '{message}' | Intents: {intent_results}")
            logger.info(f"[DEBUG] intent_results type after _enhanced_intent_recognition: {type(intent_results)}")
            
            # Manage conversation flow and determine if clarification is needed
            dialog_state = self._manage_dialog_flow(message, intent_results, self.context)
            logger.info(f"[DEBUG] intent_results type after _manage_dialog_flow: {type(intent_results)}")
            if dialog_state["needs_clarification"]:
                logger.info(f"Needs clarification: {dialog_state['clarification_question']}")
                response = dialog_state["clarification_question"]
                self.conversation_history.append({"role": "assistant", "content": response})
                return {
                    "response": response,
                    "success": True,
                    "intent": "clarification",
                    "confidence": 1.0
                }
            
            # Check if this might be a compound question that needs multiple intent processing
            # Look for question conjunctions and multiple question patterns
            compound_indicators = [
                r'(.*\?)[\s]+(and|also|additionally|moreover|furthermore|besides)[\s]+(.*\??)',
                r'(.*\?)[\s]+(what about|how about)[\s]+(.*\??)',
                r'(.*\?)[\s]+(can you also|could you also)[\s]+(.*\??)',
                r'both.*and',
                r'multiple questions',
            ]
            
            is_compound = any(re.search(pattern, message, re.IGNORECASE) for pattern in compound_indicators)
            
            # Only use multi-intent detection for messages that appear to be compound questions
            if is_compound:
                logger.info("Detected potential compound question, checking for multiple intents")
                try:
                    # Get intents for compound processing
                    intents = self._detect_multiple_intents(message)
                    
                    # Only process as compound if we actually found multiple viable intents
                    if len(intents) > 1:
                        logger.info(f"Processing compound question with {len(intents)} intents: {intents}")
                        return self._process_compound_question(message, intents)
                    else:
                        logger.info("Only one intent found despite compound indicators, processing as single intent")
                except Exception as e:
                    logger.error(f"Error in multi-intent detection: {e}. Falling back to single intent.")
            
            # Single intent processing (default path)
            
            # If confidence is very low, try to understand better with LLM only
            if isinstance(intent_results, dict) and max(intent_results.values()) < 0.25:  # Lowered from 0.5 to 0.25
                # Extract entities from the user message
                self._extract_entities(message)
                
                # Generate natural language response using LLM
                gpt_response = self._generate_response(intent_results, message)
                
                # Add the response to conversation history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": gpt_response
                })
                
                # Return full response with metadata
                return {
                    "response": gpt_response,
                    "success": True,
                    "intent": max(intent_results, key=intent_results.get),
                    "confidence": max(intent_results.values())
                }
            
            # Extract entities from the user message
            self._extract_entities(message)
            
            # Try to retrieve data for the recognized intent
            data_response = self._retrieve_data_for_intent(max(intent_results, key=intent_results.get), message)
            if data_response:
                # Generate response using LLM
                response_text = self._generate_response(max(intent_results, key=intent_results.get), data_response, message)
                
                # Add the response to conversation history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_text
                })
                
                # Update context with successful response
                self.context.update(last_intent=max(intent_results, key=intent_results.get))
                if data_response:
                    # Extract key information for context
                    if isinstance(data_response, dict):
                        if 'fabrics' in data_response or 'fabric' in data_response:
                            # Safely access fabric name to avoid IndexError
                            fabric_name = data_response.get('fabric', {}).get('name')
                            if not fabric_name:
                                fabrics_list = data_response.get('fabrics', [])
                                if fabrics_list:  # Only access [0] if list is not empty
                                    fabric_name = fabrics_list[0]  # fabrics_list[0] is the fabric name string
                            self.context.fabric = fabric_name
                        if 'devices' in data_response:
                            devices = data_response.get('devices', [])
                            if devices and isinstance(devices, list) and len(devices) > 0:
                                self.context.last_device = devices[0].get('name') or devices[0].get('deviceName')
                
                # Extract entities from the response data
                self._extract_entities(message, data_response)
                
                # Return full response
                return {
                    "response": response_text,
                    "data": data_response,
                    "success": True,
                    "intent": max(intent_results, key=intent_results.get),
                    "confidence": max(intent_results.values())
                }
            
            
            # Try dynamic question handling before final LLM fallback
            logger.info("Attempting dynamic question handling...")
            dynamic_result = self._handle_dynamic_question(message)
            if dynamic_result:
                logger.info("Dynamic question handling successful")
                self.conversation_history.append({"role": "assistant", "content": dynamic_result["response"]})
                return dynamic_result
            
            # Final fallback to LLM-based response with context
            self._extract_entities(message)  # Extract any entities from the message for context
            response = self._generate_response(intent_results, message)  # Generate context-aware response
            self.conversation_history.append({"role": "assistant", "content": response})
            if isinstance(intent_results, dict):
                return {
                    "response": response,
                    "success": True,
                    "intent": max(intent_results, key=intent_results.get),
                    "confidence": max(intent_results.values())
                }
            else:
                return {
                    "response": response,
                    "success": True,
                    "intent": "unknown",
                    "confidence": 0.0
                }
            
        except Exception as e:
            logger.error(f"[ImprovedArchitecture] Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "response": f"I'm sorry, but I encountered an error processing your message. Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _detect_multiple_intents(self, message: str) -> List[Tuple[str, float]]:
        """
        Detect multiple intents in a single message to support compound questions.
        Returns a list of (intent_name, confidence_score) tuples sorted by confidence.
        """
        try:
            # First, get the primary intent using the standard recognition
            # Use internal pattern matching and INTENT_REGISTRY for intent recognition
            primary_intent, primary_confidence = self._pattern_match_intent(message)
            
            # Create a list to hold all detected intents
            detected_intents = [(primary_intent, primary_confidence)]
            
            # Minimum confidence threshold for secondary intents
            SECONDARY_INTENT_THRESHOLD = 0.4
            
            # Get all other possible intents that meet the threshold
            # Some intent mappers support getting multiple intents directly
            # Use internal pattern matching for all possible intents above the threshold
            
            # Log the detected intents
            logger.info(f"Detected {len(detected_intents)} intents: {detected_intents}")
            
            # Add the primary intent to the context for conversation tracking
            self._update_context_intents(primary_intent, primary_confidence)
            
            return detected_intents
            
        except Exception as e:
            logger.error(f"Failed to detect multiple intents: {str(e)}")
            # Return only the unknown intent as fallback
            return [("unknown", 0.0)]
    
    def _pattern_match_intent(self, message: str):
        """
        Pattern match the user message to the best intent using self.intent_patterns.
        Returns (intent_name, confidence) where confidence is between 0 and 1.
        """
        from utils.debug_utils import dump_intent_recognition_debug
        
        message_lower = message.lower()
        best_intent = None
        best_score = 0.0
        intent_scores = {}
        
        # Log for debugging
        logger.info(f"Pattern matching intent for: '{message_lower}'")
        
        # Special case handling for problematic fabric name queries
        fabric_name_special_patterns = [
            "what is my fabric called",
            "what's my fabric called", 
            "what is the name of my fabric",
            "what's the name of my fabric",
        ]
        
        # Direct check for problematic queries first
        if message_lower in fabric_name_special_patterns:
            logger.info(f"Found direct match for special fabric_name pattern: '{message_lower}'")
            # Create a dictionary with the fabric_name intent having score 1.0
            scores_dict = {"fabric_name": 1.0}
            dump_intent_recognition_debug(message, "fabric_name", scores_dict)
            return ("fabric_name", 1.0)
        
        # Check if intent_patterns exists and has patterns
        if not hasattr(self, 'intent_patterns') or not self.intent_patterns:
            logger.warning("No intent patterns found in self.intent_patterns!")
            # Add some default fallbacks for testing
            self.intent_patterns = {
                "fabric_devices": ["fabric devices", "devices in fabric", "show devices", "list devices"],
                "device_status": ["device status", "status of device", "switch status"],
                "network_topology": ["topology", "network map", "infrastructure", "network layout"]
            }
            logger.info(f"Added fallback patterns: {self.intent_patterns}")
            
        # Debug the available patterns
        pattern_counts = {intent: len(patterns) for intent, patterns in self.intent_patterns.items()}
        logger.info(f"Available intents with pattern counts: {pattern_counts}")
            
        # First look for exact substring matches
        for intent, patterns in self.intent_patterns.items():
            intent_highest_score = 0.0
            
            # Special higher priority handling for fabric name intent
            if intent == 'fabric_name' and 'fabric name' in message_lower:
                score = 0.95  # Give very high score
                intent_highest_score = max(intent_highest_score, score)
                if score > best_score:
                    best_score = score
                    best_intent = intent
                
                for pattern in patterns:
                    if pattern in message_lower:
                        # Score: longer matches and earlier matches are better
                        base_score = len(pattern) / max(len(message_lower), 1)
                        
                        # Apply priority multiplier from YAML
                        priority_multiplier = 1.0
                        if hasattr(self, 'intent_priorities') and intent in self.intent_priorities:
                            priority = self.intent_priorities[intent]
                            priority_multiplier = 1.0 + (priority / 100.0)
                        
                        score = base_score * priority_multiplier
                    
                    # Boost fabric_name matching for specific patterns
                    if intent == 'fabric_name' and any(keyword in pattern for keyword in ['name', 'called']):
                        score = min(1.0, score * 1.5)  # Boost but cap at 1.0
                    
                    logger.debug(f"Exact match - Intent: {intent}, Pattern: '{pattern}', Score: {score:.4f}")
                    
                    # Track highest score per intent
                    intent_highest_score = max(intent_highest_score, score)
                    
                    if score > best_score:
                        best_score = score
                        best_intent = intent
            
            # Save the highest score for this intent
            if intent_highest_score > 0:
                intent_scores[intent] = intent_highest_score
                
        # If no exact matches found, try word overlap approach
        if not best_intent:
            logger.info("No exact matches found, trying word overlap matching")
            message_words = set(message_lower.split())
            
            for intent, patterns in self.intent_patterns.items():
                intent_highest_score = intent_scores.get(intent, 0.0)
                
                for pattern in patterns:
                    pattern_words = set(pattern.split())
                    overlap = message_words & pattern_words
                    if overlap:  # Only calculate if there's any overlap
                        # Score based on percentage of pattern words matched
                        score = len(overlap) / max(len(pattern_words), 1)
                        logger.debug(f"Word overlap - Intent: {intent}, Pattern: '{pattern}', "
                                    f"Overlap: {overlap}, Score: {score:.4f}")
                        
                        # Track highest score per intent
                        intent_highest_score = max(intent_highest_score, score)
                        
                        if score > best_score:
                            best_score = score
                            best_intent = intent
                
                # Save the highest score for this intent
                if intent_highest_score > 0:
                    intent_scores[intent] = intent_highest_score
        
        # If we still have no good matches, try fuzzy matching for each word
        if not best_intent or best_score < 0.3:
            logger.info("Low confidence matches, trying fuzzy matching")
            try:
                from difflib import SequenceMatcher
                
                for intent, patterns in self.intent_patterns.items():
                    intent_highest_score = intent_scores.get(intent, 0.0)
                    
                    # Try fuzzy matching each word in the message against each pattern
                    for pattern in patterns:
                        # Calculate overall similarity ratio
                        similarity = SequenceMatcher(None, message_lower, pattern).ratio()
                        
                        # Calculate partial token matching
                        msg_tokens = message_lower.split()
                        pattern_tokens = pattern.split()
                        
                        token_similarities = []
                        for msg_token in msg_tokens:
                            # Find best match for this token
                            best_token_sim = max([SequenceMatcher(None, msg_token, p_token).ratio() 
                                                for p_token in pattern_tokens] or [0])
                            token_similarities.append(best_token_sim)
                            
                        # Average token similarity weighted by length
                        if token_similarities:
                            avg_token_sim = sum(token_similarities) / len(token_similarities)
                            # Combined score: balance overall similarity with token matching
                            score = (similarity * 0.6) + (avg_token_sim * 0.4)
                            
                            if score > 0.5:  # Only consider reasonably good fuzzy matches
                                logger.debug(f"Fuzzy match - Intent: {intent}, Pattern: '{pattern}', "
                                            f"Similarity: {similarity:.4f}, Token Sim: {avg_token_sim:.4f}, "
                                            f"Score: {score:.4f}")
                                
                                # Track highest score per intent
                                intent_highest_score = max(intent_highest_score, score)
                                
                                if score > best_score:
                                    best_score = score
                                    best_intent = intent
                    
                    # Save the highest score for this intent
                    if intent_highest_score > 0:
                        intent_scores[intent] = intent_highest_score
            except Exception as e:
                logger.error(f"Error during fuzzy matching: {e}")
        
        # Confidence: clamp to [0, 1]
        confidence = min(1.0, max(0.0, best_score))
        
        # Dump detailed recognition info using our debug util
        dump_intent_recognition_debug(message, best_intent, intent_scores)
        
        # If we have intent tracking enabled, store this intent
        self._update_context_intents(best_intent, confidence)
        
        # Add special debug for 0 scores
        if best_score == 0.0:
            logger.warning(f"⚠️ Zero score recognition for message: '{message}' - check intent patterns!")
            logger.info(f"Words in message: {message_lower.split()}")
            
        return (best_intent, confidence) if best_intent else (None, 0.0)  # Return None with 0 confidence if no match

    def hallucination_detected(self, llm_output: str, highlights: Dict[str, Any], grounded_data: Dict[str, Any]) -> bool:
        """
        Check if LLM output contains hallucinations based on provided ground truth data.
        
        Args:
            llm_output: The text output from the LLM
            highlights: Dictionary of key data points to verify against
            grounded_data: Dictionary of ground truth data to check against
            
        Returns:
            True if hallucination detected, False otherwise
        """
        # Debug logging
        logger.info(f"[HALLUCINATION CHECK] LLM output: {llm_output[:100]}...")
        logger.info(f"[HALLUCINATION CHECK] Highlights: {highlights}")
        logger.info(f"[HALLUCINATION CHECK] Grounded data: {grounded_data}")
        
        # If we have no grounded data, be more lenient
        if not highlights and not grounded_data:
            logger.info("[HALLUCINATION CHECK] No grounded data available, being lenient")
            # Only flag obvious hallucinations like specific IP addresses or device IDs
            suspicious_patterns = [
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
                r'\b[A-Z]{2,4}-\d{3,}\b',  # Device names like SW-1234, LEAF-001
                r'\b\d+\.\d+\.\d+\b',  # Version numbers like 1.2.3
                r'\bPort\s+\d+/\d+/\d+\b',  # Specific port numbers
                r'\bVLAN\s+\d{3,}\b',  # Specific VLAN numbers
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, llm_output):
                    logger.info(f"[HALLUCINATION CHECK] Found suspicious pattern: {pattern}")
                    return True
            return False
        
        safe_values = set()
        for v in list(highlights.values()) + list(grounded_data.values() if isinstance(grounded_data, dict) else []):
            if isinstance(v, (str, int, float)):
                safe_values.add(str(v))
            elif isinstance(v, list):
                safe_values.update([str(x) for x in v if isinstance(x, (str, int, float))])
        
        logger.info(f"[HALLUCINATION CHECK] Safe values: {safe_values}")
        
        # Check for suspicious numbers
        llm_numbers = set(re.findall(r"\b\d+\b", llm_output))
        for n in llm_numbers:
            if n not in safe_values and int(n) > 100:  # Only flag large numbers
                logger.info(f"[HALLUCINATION CHECK] Suspicious number: {n}")
                return True
        
        # Whitelist for common technical terms that should not trigger hallucination detection
        whitelist_terms = [
            'switch fabric',
            'fabric overview',
            'topology',
            'health status',
            'fabric health',
            'spine',
            'leaf',
            'device interface',
            'interface',
            'network device',
            'device',
            'network infrastructure',
            'score',
            'anomaly score',
            'advisory score',
        ]
        
        # Be less aggressive with technical keywords - only flag if they're used with specific data
        suspicious_technical_patterns = [
            r'device\s+[A-Z0-9\-]{4,}',  # device names with 4+ chars
            r'interface\s+[A-Za-z0-9/\-]{6,}',  # interface names with 6+ chars  
            r'switch\s+[A-Z0-9\-]{4,}',  # switch names with 4+ chars
        ]
        for pattern in suspicious_technical_patterns:
            matches = re.findall(pattern, llm_output, re.IGNORECASE)
            for match in matches:
                # Skip if the match is a whitelisted term
                if any(term.lower() in match.lower() for term in whitelist_terms):
                    logger.info(f"[HALLUCINATION CHECK] Whitelisted term found: {match}")
                    continue
                    
                # Otherwise check if it's in the safe values
                if not any(match.lower() in str(v).lower() for v in safe_values):
                    logger.info(f"[HALLUCINATION CHECK] Suspicious technical pattern: {match}")
                    return True
        
        logger.info("[HALLUCINATION CHECK] No hallucination detected")
        return False

    def _handle_dynamic_question(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced dynamic question handler with sophisticated intent understanding
        Handles complex network management questions across all endpoints
        """
        try:
            # Initialize enhanced handler if not already done
            if not hasattr(self, '_enhanced_handler'):
                try:
                    from enhanced_dynamic_handler import EnhancedDynamicHandler
                    self._enhanced_handler = EnhancedDynamicHandler(self)
                    logger.info("Enhanced dynamic handler initialized successfully")
                except ImportError as e:
                    logger.warning(f"Could not import enhanced handler: {e}")
                    return self._basic_dynamic_fallback(message)
            
            # Use enhanced handler
            result = self._enhanced_handler.handle_dynamic_question(message)
            
            if result:
                logger.info(f"Enhanced handler provided response with confidence: {result.get('confidence', 0)}")
                return result
            else:
                logger.info("Enhanced handler returned no result, trying fallback")
                return self._basic_dynamic_fallback(message)
                
        except Exception as e:
            logger.error(f"Enhanced dynamic handler error: {str(e)}")
            # Fallback to basic handler
            return self._basic_dynamic_fallback(message)

    def _basic_dynamic_fallback(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Basic fallback for dynamic questions if enhanced handler fails
        Provides simple keyword-based routing as backup
        """
        try:
            logger.info(f"Using basic dynamic fallback for: '{message}'")
            message_lower = message.lower()
            
            # Enhanced keyword routing with better coverage
            if any(word in message_lower for word in ['anomal', 'issue', 'problem', 'critical', 'error', 'alert']):
                logger.info(f"[ANOMALY ROUTING] Detected anomaly keywords in message: '{message}'")
                logger.info(f"[ANOMALY ROUTING] Routing to anomalies handler")
                result = self._get_network_anomalies(message)
                logger.info(f"[ANOMALY ROUTING] Handler returned: {result}")
                intent_suffix = "anomalies"
            elif any(word in message_lower for word in ['device', 'switch', 'node', 'fabric', 'hardware']):
                logger.info("Routing to fabric devices handler")
                result = self._get_fabric_devices(message)
                intent_suffix = "devices"
            elif any(word in message_lower for word in ['interface', 'port', 'link', 'ethernet', 'bandwidth']):
                logger.info("Routing to interfaces handler")
                result = self._get_network_interfaces(message)
                intent_suffix = "interfaces"
            elif any(word in message_lower for word in ['event', 'log', 'history', 'recent']):
                logger.info("Routing to events handler")
                result = self._get_network_events(message)
                intent_suffix = "events"
            elif any(word in message_lower for word in ['flow', 'traffic', 'path', 'route']):
                logger.info("Routing to flows handler")
                result = self._get_network_flows(message)
                intent_suffix = "flows"
            elif any(word in message_lower for word in ['advisory', 'bulletin', 'security', 'psirt']):
                logger.info("Routing to advisories handler")
                result = self._get_advisories(message)
                intent_suffix = "advisories"
            elif any(word in message_lower for word in ['system', 'version', 'cluster', 'service']):
                logger.info("Routing to system info handler")
                result = self._get_system_information(message)
                intent_suffix = "system"
            else:
                # Default to fabric devices for general queries
                logger.info("Using default routing to fabric devices")
                result = self._get_fabric_devices(message)
                intent_suffix = "general"
            
            if result and result.get('message'):
                response_text = result.get('message', 'No data available.')
                
                # Add context-aware prefix
                if 'show me' in message_lower:
                    response_text = f"Here's what I found: {response_text}"
                elif 'any' in message_lower and '?' in message:
                    response_text = f"Checking your network: {response_text}"
                elif 'how many' in message_lower:
                    response_text = f"Based on current data: {response_text}"
                
                return {
                    "response": response_text,
                    "data": result.get('data'),
                    "success": True,
                    "intent": f"dynamic_{intent_suffix}",
                    "confidence": 0.7,
                    "handler": "basic_fallback"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in basic dynamic fallback: {str(e)}")
            return None
    def _basic_dynamic_fallback(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Basic fallback for dynamic questions if enhanced handler fails
        Provides simple keyword-based routing as backup
        """
        try:
            logger.info(f"Using basic dynamic fallback for: '{message}'")
            message_lower = message.lower()
            
            # Enhanced keyword routing with better coverage
            if any(word in message_lower for word in ['anomal', 'issue', 'problem', 'critical', 'error', 'alert']):
                logger.info(f"[ANOMALY ROUTING] Detected anomaly keywords in message: '{message}'")
                logger.info(f"[ANOMALY ROUTING] Routing to anomalies handler")
                result = self._get_network_anomalies(message)
                logger.info(f"[ANOMALY ROUTING] Handler returned: {result}")
                intent_suffix = "anomalies"
            elif any(word in message_lower for word in ['device', 'switch', 'node', 'fabric', 'hardware']):
                logger.info("Routing to fabric devices handler")
                result = self._get_fabric_devices(message)
                intent_suffix = "devices"
            elif any(word in message_lower for word in ['interface', 'port', 'link', 'ethernet', 'bandwidth']):
                logger.info("Routing to interfaces handler")
                result = self._get_network_interfaces(message)
                intent_suffix = "interfaces"
            elif any(word in message_lower for word in ['event', 'log', 'history', 'recent']):
                logger.info("Routing to events handler")
                result = self._get_network_events(message)
                intent_suffix = "events"
            elif any(word in message_lower for word in ['flow', 'traffic', 'path', 'route']):
                logger.info("Routing to flows handler")
                result = self._get_network_flows(message)
                intent_suffix = "flows"
            elif any(word in message_lower for word in ['advisory', 'bulletin', 'security', 'psirt']):
                logger.info("Routing to advisories handler")
                result = self._get_advisories(message)
                intent_suffix = "advisories"
            elif any(word in message_lower for word in ['system', 'version', 'cluster', 'service']):
                logger.info("Routing to system info handler")
                result = self._get_system_information(message)
                intent_suffix = "system"
            else:
                # Default to fabric devices for general queries
                logger.info("Using default routing to fabric devices")
                result = self._get_fabric_devices(message)
                intent_suffix = "general"
            
            if result and result.get('message'):
                response_text = result.get('message', 'No data available.')
                
                # Add context-aware prefix
                if 'show me' in message_lower:
                    response_text = f"Here's what I found: {response_text}"
                elif 'any' in message_lower and '?' in message:
                    response_text = f"Checking your network: {response_text}"
                elif 'how many' in message_lower:
                    response_text = f"Based on current data: {response_text}"
                
                return {
                    "response": response_text,
                    "data": result.get('data'),
                    "success": True,
                    "intent": f"dynamic_{intent_suffix}",
                    "confidence": 0.7,
                    "handler": "basic_fallback"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in basic dynamic fallback: {str(e)}")
            return None
    def _basic_dynamic_fallback(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Basic fallback for dynamic questions if enhanced handler fails
        Provides simple keyword-based routing as backup
        """
        try:
            logger.info(f"Using basic dynamic fallback for: '{message}'")
            message_lower = message.lower()
            
            # Enhanced keyword routing with better coverage
            if any(word in message_lower for word in ['anomal', 'issue', 'problem', 'critical', 'error', 'alert']):
                logger.info(f"[ANOMALY ROUTING] Detected anomaly keywords in message: '{message}'")
                logger.info(f"[ANOMALY ROUTING] Routing to anomalies handler")
                result = self._get_network_anomalies(message)
                logger.info(f"[ANOMALY ROUTING] Handler returned: {result}")
                intent_suffix = "anomalies"
            elif any(word in message_lower for word in ['device', 'switch', 'node', 'fabric', 'hardware']):
                logger.info("Routing to fabric devices handler")
                result = self._get_fabric_devices(message)
                intent_suffix = "devices"
            elif any(word in message_lower for word in ['interface', 'port', 'link', 'ethernet', 'bandwidth']):
                logger.info("Routing to interfaces handler")
                result = self._get_network_interfaces(message)
                intent_suffix = "interfaces"
            elif any(word in message_lower for word in ['event', 'log', 'history', 'recent']):
                logger.info("Routing to events handler")
                result = self._get_network_events(message)
                intent_suffix = "events"
            elif any(word in message_lower for word in ['flow', 'traffic', 'path', 'route']):
                logger.info("Routing to flows handler")
                result = self._get_network_flows(message)
                intent_suffix = "flows"
            elif any(word in message_lower for word in ['advisory', 'bulletin', 'security', 'psirt']):
                logger.info("Routing to advisories handler")
                result = self._get_advisories(message)
                intent_suffix = "advisories"
            elif any(word in message_lower for word in ['system', 'version', 'cluster', 'service']):
                logger.info("Routing to system info handler")
                result = self._get_system_information(message)
                intent_suffix = "system"
            else:
                # Default to fabric devices for general queries
                logger.info("Using default routing to fabric devices")
                result = self._get_fabric_devices(message)
                intent_suffix = "general"
            
            if result and result.get('message'):
                response_text = result.get('message', 'No data available.')
                
                # Add context-aware prefix
                if 'show me' in message_lower:
                    response_text = f"Here's what I found: {response_text}"
                elif 'any' in message_lower and '?' in message:
                    response_text = f"Checking your network: {response_text}"
                elif 'how many' in message_lower:
                    response_text = f"Based on current data: {response_text}"
                
                return {
                    "response": response_text,
                    "data": result.get('data'),
                    "success": True,
                    "intent": f"dynamic_{intent_suffix}",
                    "confidence": 0.7,
                    "handler": "basic_fallback"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in basic dynamic fallback: {str(e)}")
            return None
    def _generate_response(self, intent: Union[str, Dict[str, Any]], data: Optional[Dict[str, Any]] = None, message: Optional[str] = None):
        """
        Generate a natural language response for the given intent and data
        using the LLM processor. Handles hallucination detection and fallbacks.
        
        Args:
            intent: The detected intent name or dictionary
            data: The retrieved data for the intent
            message: The original user message
            
        Returns:
            A natural language response string
        """
        try:
            # Extract key information from the data
            highlights = {}
            grounded_data = {}
            
            # Populate grounded_data from API response
            if data:
                grounded_data = data.copy()
                # Extract key values for hallucination checking
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (str, int, float)):
                            highlights[key] = value
                        elif isinstance(value, list) and value:
                            highlights[f"{key}_list"] = value[:5]  # First 5 items
            
            # Extract context from recent intents for LLM
            recent_intents = [intent_obj["name"] for intent_obj in self.context.recent_intents[-3:]] if hasattr(self.context, "recent_intents") else []
            context_info = ""
            
            # Convert complex data to string for LLM to use
            data_str = json.dumps(data, indent=2, default=str) if data else "No data available."
            
            if isinstance(intent, dict):
                intent_name = intent.get("name")
                intent_confidence = intent.get("confidence")
                system_prompt = f"""You are Nexus, an AI assistant for Cisco Nexus Dashboard.
            You specialize in data center networking, Cisco switches, routers, and network fabric infrastructure.
            
            IMPORTANT: When the user asks about "fabric devices", they are referring to NETWORK SWITCHES and ROUTERS in a data center fabric, NOT textile or clothing materials.
            
            User's network question: {message}
            Current intent: {intent_name}
            Recent conversation flow: {', '.join(recent_intents) if recent_intents else 'None'}{context_info}
            
            Network device data from Cisco Nexus Dashboard:
            {data_str}
            
            Response guidelines:
            1. Focus ONLY on network infrastructure (switches, routers, network devices)
            2. Be conversational but concise about network information
            3. Never mention textiles, clothing, or fabric materials
            4. Interpret device names, models, and network status information
            5. Prioritize network health issues or alerts if present
            6. Use only the provided network data - don't invent information"""
            else:
                system_prompt = f"""You are Nexus, an AI assistant for Cisco Nexus Dashboard.
            You specialize in data center networking, Cisco switches, routers, and network fabric infrastructure.
            
            IMPORTANT: When the user asks about "fabric devices", they are referring to NETWORK SWITCHES and ROUTERS in a data center fabric, NOT textile or clothing materials.
            
            User's network question: {message}
            Current intent: {intent}
            Recent conversation flow: {', '.join(recent_intents) if recent_intents else 'None'}{context_info}
            
            Network device data from Cisco Nexus Dashboard:
            {data_str}
            
            Response guidelines:
            1. Focus ONLY on network infrastructure (switches, routers, network devices)
            2. Be conversational but concise about network information
            3. Never mention textiles, clothing, or fabric materials
            4. Interpret device names, models, and network status information
            5. Prioritize network health issues or alerts if present
            6. Use only the provided network data - don't invent information"""

            conversation = []
            for msg in self.conversation_history[-10:] if hasattr(self, "conversation_history") else []:
                conversation.append({"role": msg["role"], "content": msg["content"]})

            llm_response = self.llm_processor.complete(
                system_prompt=system_prompt,
                conversation=conversation,
                max_tokens=350
            )
            logger.info(f"[LLM OUTPUT] Response: {llm_response}")
            
            # Special handling for fabric_devices intent - check for textile-related content
            if ((isinstance(intent, dict) and intent.get("name") == "fabric_devices") or 
                (isinstance(intent, str) and intent == "fabric_devices")) and data and "message" in data:
                
                textile_keywords = [
                    'weft', 'warp', 'loom', 'weave', 'yarn', 'thread', 'textile', 'cotton', 'silk', 
                    'fabric production', 'spinning', 'heddle', 'shuttle', 'clothing', 'garment', 
                    'embellishment', 'button', 'sequin', 'applique', 'printed design', 'wearable technology',
                    'smart fabric', 'bluetooth', 'phone', 'tablet', 'mobile device', 'wi-fi signal',
                    'physical', 'environment', 'location', 'nearby device', 'connected device',
                    'check your phone', 'check your tablet', 'bluetooth setting', 'wi-fi network',
                    'text-based AI', 'physically interact', 'capability', 'access specific information'
                ]
                
                # Check for textile content OR generic AI responses
                llm_lower = (llm_response or '').lower()
                has_textile_content = any(keyword.lower() in llm_lower for keyword in textile_keywords)
                has_generic_ai_response = ('text-based ai' in llm_lower or 
                                         'physically interact' in llm_lower or
                                         'capability' in llm_lower)
                
                if has_textile_content or has_generic_ai_response:
                    logger.warning("[TEXTILE/GENERIC CONTENT DETECTED] LLM generated inappropriate content for fabric_devices. Using structured summary.")
                    return data["message"]
            
            # Check for data in API responses and extract meaningful information
            # This is a more robust approach to handling various API response formats
            extracted_data = {}
            data_found = False
            empty_data_message = ""
            
            if grounded_data:
                # Log the entire grounded data structure for debugging
                logger.info(f"[API RESPONSE DEBUG] Full grounded_data structure: {json.dumps(grounded_data, default=str)[:1000]}...")
                
                # Check if this is an API call with results
                if isinstance(grounded_data, dict):
                    endpoint = grounded_data.get('endpoint_used', '')
                    api_data = grounded_data.get('data', {})
                    logger.info(f"[API RESPONSE DEBUG] Processing endpoint: {endpoint}")
                    logger.info(f"[API RESPONSE DEBUG] API data type: {type(api_data).__name__}")
                    
                    # Flexible data extraction from different response formats
                    if isinstance(api_data, dict):
                        # Try multiple common response formats
                        entries = None
                        
                        # Extract entries/items from various possible fields
                        for field in ['entries', 'endpoints', 'devices', 'items', 'data', 'results']:
                            if field in api_data and isinstance(api_data[field], list):
                                entries = api_data[field]
                                logger.info(f"[API RESPONSE DEBUG] Found {len(entries)} items in field '{field}'")
                                break
                                
                        # Check for direct array of objects at the root level
                        if not entries and isinstance(api_data, dict) and len(api_data) > 0:
                            # Look for array fields that might contain data
                            for key, value in api_data.items():
                                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                    entries = value
                                    logger.info(f"[API RESPONSE DEBUG] Found {len(entries)} items in root field '{key}'")
                                    break
                        
                        # Check if we found any data
                        if entries and len(entries) > 0:
                            data_found = True
                            extracted_data['items'] = entries
                            extracted_data['count'] = len(entries)
                        else:
                            # Check for pagination info
                            total_count = api_data.get('totalCount', api_data.get('totalItemsCount', 0))
                            if total_count > 0:
                                data_found = True
                                extracted_data['total_count'] = total_count
                            else:
                                # Create context-appropriate empty data messages
                                if 'endpoint' in endpoint.lower():
                                    empty_data_message = "I don't see any endpoints in the network right now. This could be because there are no endpoints connected, or because the Nexus Dashboard doesn't have visibility into endpoints yet."
                                elif 'anomal' in endpoint.lower():
                                    empty_data_message = "Good news! There are currently no anomalies detected in your network. The system is operating normally without any identified issues."
                                elif 'device' in endpoint.lower() or 'inventory' in endpoint.lower():
                                    empty_data_message = "I don't see any network devices registered in the system. You may need to add devices to the Nexus Dashboard or check your connection settings."
                                else:
                                    empty_data_message = "I searched for the requested information, but there are no results available at this time."
                    
                    # Handle case where API data is directly a list
                    elif isinstance(api_data, list) and len(api_data) > 0:
                        data_found = True
                        extracted_data['items'] = api_data
                        extracted_data['count'] = len(api_data)
                        logger.info(f"[API RESPONSE DEBUG] Found {len(api_data)} items in direct list response")
                            
                    # Log extraction results
                    if data_found:
                        logger.info(f"[API RESPONSE DEBUG] Successfully extracted data with {extracted_data.get('count', 0)} items")
                    else:
                        logger.info(f"[API RESPONSE DEBUG] No data extracted from API response")
            else:
                logger.info("[API RESPONSE DEBUG] No grounded_data available")
                empty_data_message = "I don't have enough information to answer that question. Please try a different query."
            
            # If no data was found, return the context-specific message
            if not data_found and empty_data_message:
                logger.info(f"[EMPTY DATA DETECTED] Returning empty data message for query related to: {intent}")
                return empty_data_message
            
            # Otherwise continue with hallucination detection                
            if self.hallucination_detected(llm_response or '', highlights, grounded_data):
                logger.warning("[LLM HALLUCINATION DETECTED] LLM output may contain invented technical data. Returning fallback message.")
                
                # For fabric_devices intent, use the structured summary if available
                if isinstance(intent, dict) and intent.get("name") == "fabric_devices" and data and "message" in data:
                    return data["message"]
                elif isinstance(intent, str) and intent == "fabric_devices" and data and "message" in data:
                    return data["message"]
                
                return "I was unable to find specific technical details for your request. Please refer to the dashboard for more information or try rephrasing your question."
            
            # Conditionally add troubleshooting and follow-up information
            if data and intent in ["fabric_health", "device_health", "interface_status", "network_anomalies"]:
                troubleshooting_explanation = self._explain_anomaly(data)
                follow_up_suggestions = self._generate_follow_up(intent, data)
                if troubleshooting_explanation["explanation"] or follow_up_suggestions:
                    return llm_response + "\n\nTroubleshooting Explanation: " + troubleshooting_explanation["explanation"] + "\n\nFollow-up Suggestions: " + ", ".join(follow_up_suggestions)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error in _generate_response: {str(e)}")
            # Fallback to a simple response
            return f"I found some information about your request, but I'm having trouble formatting it. Let me know if you'd like more specific details about {intent}."

    def _retrieve_data_for_intent(self, intent: str, message: str):
        """
        Main method that calls intent handlers with improved exception handling
        """
        try:
            logger.info(f"[INTENT HANDLER] Attempting to handle intent: '{intent}' for message: '{message}'")
            from agents.intent_registry import INTENT_REGISTRY
            meta = INTENT_REGISTRY.get(intent)
            handler = meta.handler if meta else None
            if handler:
                logger.info(f"[INTENT HANDLER] Found handler for intent: '{intent}'. Calling handler...")
                try:
                    # Create proper HandlerContext for consistency
                    logger.info(f"[INTENT HANDLER] ND client availability before HandlerContext creation: {self.nd_client is not None}")
                    # --- PATCH: Ensure ND client is always injected into context ---
                    nd_client = self.nd_client
                    if nd_client is None:
                        logger.error("[INTENT HANDLER] ND client is None! Attempting fallback recovery...")
                        try:
                            from improved_chat_architecture import improved_chat
                            if hasattr(improved_chat, 'nd_client') and improved_chat.nd_client is not None:
                                nd_client = improved_chat.nd_client
                                logger.info("[INTENT HANDLER] Recovered ND client from improved_chat")
                            else:
                                from chat_integration_bridge import chat_bridge
                                if hasattr(chat_bridge, 'nd_client') and chat_bridge.nd_client is not None:
                                    nd_client = chat_bridge.nd_client
                                    logger.info("[INTENT HANDLER] Recovered ND client from chat_bridge")
                        except Exception as e:
                            logger.error(f"[INTENT HANDLER] Error during ND client fallback: {e}")
                        # Final fallback: try loading from token cache
                        if nd_client is None:
                            try:
                                from ndi_cache import get_token
                                from cisco_api.nexus.dashboard_client import NexusDashboardClient
                                token = get_token()
                                if token:
                                    nd_client = NexusDashboardClient(token=token)
                                    logger.info("[INTENT HANDLER] Recovered ND client from token cache")
                            except Exception as e:
                                logger.error(f"[INTENT HANDLER] Error creating ND client from token cache: {e}")
                    if nd_client is None:
                        logger.error("[INTENT HANDLER] All ND client recovery attempts failed! Handler will receive empty context.")
                    handler_context = HandlerContext(
                        nd_client=nd_client,
                        llm_processor=self.llm_processor,
                        conversation_context=self.context
                    )
                    
                    # Log the context creation with detailed client status
                    client_in_context = handler_context.get_param('nd_client') is not None
                    logger.info(f"[INTENT HANDLER] Created HandlerContext for intent '{intent}'. ND client in context: {client_in_context}")
                    
                    # Debug the client in context
                    if client_in_context:
                        logger.info(f"[INTENT HANDLER] ND client type: {type(handler_context.get_param('nd_client'))}")
                    else:
                        logger.error("[INTENT HANDLER] ND client is missing from context after creation!")
                    
                    # Safely call handler, avoiding double-passing 'context'
                    import inspect
                    handler_fn = handler
                    bound = hasattr(handler, "__self__") and handler.__self__ is not None
                    if not bound:
                        handler_fn = handler.__get__(self)
                    sig = inspect.signature(handler_fn)
                    params = sig.parameters
                    kwargs = {}
                    if "context" in params:
                        kwargs["context"] = handler_context
                    result = handler_fn(message, **kwargs)
                    # Validate the handler result
                    if result is None:
                        logger.warning(f"[INTENT HANDLER] Handler for '{intent}' returned None")
                        return {"error": "No data available for this query", "message": "I couldn't find any relevant data for your query."}
                    # Check if there's an error in the response
                    if isinstance(result, dict) and result.get("error"):
                        logger.warning(f"[INTENT HANDLER] Error in handler response: {result['error']}")
                        # We'll keep the error dict as is, it will be handled by _generate_response
                    else:
                        logger.info(f"[INTENT HANDLER] Successfully retrieved data for '{intent}'")
                        # Log a preview of the successful data response
                        try:
                            data_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                            logger.info(f"[INTENT HANDLER] Data preview: {data_preview}")
                        except Exception:
                            logger.info(f"[INTENT HANDLER] Data retrieved but could not generate preview")
                    return result
                except Exception as handler_error:
                    # More specific error handling for handler exceptions
                    logger.error(f"[INTENT HANDLER] Error in handler '{intent}': {handler_error}\n{traceback.format_exc()}")
                    return {"error": str(handler_error), "message": f"An error occurred while processing your request: {str(handler_error)}"}
            else:
                logger.warning(f"[INTENT HANDLER] No handler found for intent: '{intent}'")
                return {"error": f"No handler for intent: {intent}", "message": "I don't know how to handle that type of request yet."}
        except Exception as e:
            logger.error(f"Error in _retrieve_data_for_intent: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "message": "An unexpected error occurred while processing your request."}
    
    def _get_network_topology(self, message: str):
        """
        Get network topology information from Nexus Dashboard
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with topology information or error message
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            # Use verified ND endpoint for network topology (using get_fabrics as a basic topology view)
            from agents.verified_endpoints import get_nd_endpoint
            topology_endpoint = get_nd_endpoint("site_management", "get_fabrics")
            if not topology_endpoint:
                logger.error("ND network topology endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve network topology information."
                }
            logger.info(f"Attempting to get network topology from ND: {topology_endpoint}")
            response = self._api_call_with_retry(topology_endpoint, max_retries=3, cache_duration=300)
            if not response:
                error_msg = "Failed to retrieve network topology"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve network topology information."
                }
            topology_data = response
            logger.info(f"Successfully retrieved network topology from ND. Sample: {str(topology_data)[:200]}...")
            # Compose a conversational summary
            fabric_count = 0
            topology_names = []
            if isinstance(topology_data, dict):
                fabrics = topology_data.get("fabrics") or topology_data.get("data") or []
                if isinstance(fabrics, list):
                    fabric_count = len(fabrics)
                    topology_names = [f.get("displayName") or f.get("name") for f in fabrics if isinstance(f, dict)]
            summary = (
                f"I've mapped out your network topology and found {fabric_count} fabric(s). "
                f"Fabrics: {', '.join(topology_names[:5]) + ('...' if len(topology_names) > 5 else '')}" if fabric_count else
                "I couldn't find any fabrics to display network topology."
            )
            return {
                "topology": topology_data,
                "total": fabric_count,
                "names": topology_names,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error getting network topology: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving network topology information."
            }

    def _get_fabric_health(self, message: str) -> Optional[dict]:
        """
        Get fabric health information from Nexus Dashboard using a verified endpoint.
        Returns a conversational summary and raw health data.
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with fabric health information or error message
        """
        # Import required modules at the beginning of the method
        from agents.verified_endpoints import get_nd_endpoint
        
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            # Use verified ND endpoint for fabric health
            health_endpoint = get_nd_endpoint("site_management", "get_fabrics")
            if not health_endpoint:
                logger.error("ND fabric health endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve fabric health information."
                }
            logger.info(f"Attempting to get fabric health from ND: {health_endpoint}")
            response = self._api_call_with_retry(health_endpoint, max_retries=3, cache_duration=300)
            if not response:
                error_msg = "Failed to retrieve fabric health"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve fabric health information."
                }
            health_data = response
            logger.info(f"Successfully retrieved fabric health from ND. Sample: {str(health_data)[:200]}...")
            # Compose a conversational summary
            fabric_count = 0
            health_statuses = []
            if isinstance(health_data, dict):
                fabrics = health_data.get("fabrics") or health_data.get("data") or []
                if isinstance(fabrics, list):
                    fabric_count = len(fabrics)
                    for f in fabrics:
                        if isinstance(f, dict):
                            name = f.get("displayName") or f.get("name")
                            health = f.get("healthStatus") or f.get("health") or "Unknown"
                            health_statuses.append(f"{name}: {health}")
            summary = (
                f"I've checked your fabric health. {fabric_count} fabric(s) found. "
                f"Status: {', '.join(health_statuses[:5]) + ('...' if len(health_statuses) > 5 else '')}" if fabric_count else
                "I couldn't find any fabrics to check health status for in your Nexus Dashboard inventory."
            )
            return {
                "health": health_data,
                "total": fabric_count,
                "statuses": health_statuses,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error getting fabric health: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving fabric health information."
            }

    def _get_network_topology(self, message: str) -> Optional[dict]:
        """
        Get network topology information from Nexus Dashboard using a verified endpoint.
        Returns a conversational summary and raw topology data.
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            # Use verified ND endpoint for network topology (using get_fabrics as a basic topology view)
            from agents.verified_endpoints import get_nd_endpoint
            topology_endpoint = get_nd_endpoint("site_management", "get_fabrics")
            if not topology_endpoint:
                logger.error("ND network topology endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve network topology information."
                }
            logger.info(f"Attempting to get network topology from ND: {topology_endpoint}")
            response = self._api_call_with_retry(topology_endpoint, max_retries=3, cache_duration=300)
            if not response:
                error_msg = "Failed to retrieve network topology"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve network topology information."
                }
            topology_data = response
            logger.info(f"Successfully retrieved network topology from ND. Sample: {str(topology_data)[:200]}...")
            # Compose a conversational summary
            fabric_count = 0
            topology_names = []
            if isinstance(topology_data, dict):
                fabrics = topology_data.get("fabrics") or topology_data.get("data") or []
                if isinstance(fabrics, list):
                    fabric_count = len(fabrics)
                    topology_names = [f.get("displayName") or f.get("name") for f in fabrics if isinstance(f, dict)]
            summary = (
                f"I've mapped out your network topology and found {fabric_count} fabric(s). "
                f"Fabrics: {', '.join(topology_names[:5]) + ('...' if len(topology_names) > 5 else '')}" if fabric_count else
                "I couldn't find any fabrics to display network topology."
            )
            return {
                "topology": topology_data,
                "total": fabric_count,
                "names": topology_names,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error getting network topology: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving network topology information."
            }
    def _get_fabric_health(self, message: str) -> Optional[dict]:
        """
        Get comprehensive fabric health information by combining device status and anomalies.
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with fabric health information or error message
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            
            # Get device information
            devices_result = self._get_fabric_devices(message)
            
            # Get anomaly information  
            anomalies_result = self._get_network_anomalies(message)
            
            # Combine the results for a comprehensive health view
            health_summary = {
                "devices": devices_result if devices_result and not devices_result.get("error") else {"message": "Device data unavailable"},
                "anomalies": anomalies_result if anomalies_result and not anomalies_result.get("error") else {"message": "Anomaly data unavailable"}
            }
            # Create a conversational summary
            device_status = "Device inventory available" if not devices_result.get("error") else "Device inventory unavailable"
            anomaly_status = "Anomaly data available" if not anomalies_result.get("error") else "Anomaly data unavailable"
            
            summary = f"Fabric Health Overview: {device_status}, {anomaly_status}."
            
            if not devices_result.get("error") and not anomalies_result.get("error"):
                # Extract key metrics if both are available
                device_count = devices_result.get("data", {}).get("device_count", 0)
                anomaly_count = len(anomalies_result.get("data", {}).get("anomalies", [])) if isinstance(anomalies_result.get("data", {}).get("anomalies"), list) else 0
                
                summary = f"Your fabric has {device_count} devices with {anomaly_count} reported anomalies."
            
            return {
                "message": summary,
                "data": health_summary,
                "device_count": devices_result.get("data", {}).get("device_count", 0) if devices_result and not devices_result.get("error") else 0,
                "anomaly_count": len(anomalies_result.get("data", {}).get("anomalies", [])) if anomalies_result and not anomalies_result.get("error") and isinstance(anomalies_result.get("data", {}).get("anomalies"), list) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in _get_fabric_health: {e}")
            return {
                "error": str(e),
                "message": f"Unable to retrieve fabric health information: {str(e)}"
            }

    def _get_fabric_devices(self, message: str) -> Optional[dict]:
        """
        Get fabric devices information from Nexus Dashboard using a verified endpoint.
        Returns a conversational summary and raw device data.
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with fabric device information or error message
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            
            # Use the verified working endpoint for switches inventory
            devices_endpoint = "/nexus/infra/api/platform/v1/nodes"
            
            logger.info(f"Attempting to get fabric devices from ND: {devices_endpoint}")
            response = self._api_call_with_retry(devices_endpoint, max_retries=3, cache_duration=300)
            
            if not response:
                error_msg = "Failed to retrieve fabric devices"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve fabric device information."
                }
            
            devices_data = response
            logger.info(f"Successfully retrieved fabric devices from ND. Sample: {str(devices_data)[:200]}...")
            
            # Compose a conversational summary
            device_count = 0
            device_names = []
            device_types = []
            device_statuses = []
            
            if isinstance(devices_data, dict):
                # Handle the specific structure: resourcesList.switches
                switches = devices_data.get("resourcesList", {}).get("switches", [])
                if isinstance(switches, list):
                    device_count = len(switches)
                    for device in switches:
                        if isinstance(device, dict):
                            name = device.get("name") or device.get("deviceName") or device.get("hostname") or device.get("switchName")
                            device_type = device.get("model") or device.get("deviceType") or device.get("type") or "Switch"
                            status = device.get("nodeState") or device.get("status") or device.get("operStatus") or device.get("health") or "Unknown"
                            if name:
                                device_names.append(name)
                            if device_type not in device_types:
                                device_types.append(device_type)
                            if status not in device_statuses:
                                device_statuses.append(status)
                else:
                    # Fallback to original parsing logic
                    devices = devices_data.get("devices") or devices_data.get("data") or devices_data.get("inventory") or []
                    if isinstance(devices, list):
                        device_count = len(devices)
                        for device in devices:
                            if isinstance(device, dict):
                                name = device.get("deviceName") or device.get("name") or device.get("hostname") or device.get("switchName")
                                device_type = device.get("deviceType") or device.get("type") or device.get("model") or "Switch"
                                status = device.get("status") or device.get("operStatus") or device.get("health") or "Unknown"
                                if name:
                                    device_names.append(name)
                                if device_type not in device_types:
                                    device_types.append(device_type)
                                if status not in device_statuses:
                                    device_statuses.append(status)
            elif isinstance(devices_data, list):
                device_count = len(devices_data)
                for device in devices_data:
                    if isinstance(device, dict):
                        name = device.get("deviceName") or device.get("name") or device.get("hostname") or device.get("switchName")
                        device_type = device.get("deviceType") or device.get("type") or device.get("model") or "Switch"
                        status = device.get("status") or device.get("operStatus") or device.get("health") or "Unknown"
                        if name:
                            device_names.append(name)
                        if device_type not in device_types:
                            device_types.append(device_type)
                        if status not in device_statuses:
                            device_statuses.append(status)
            
            if device_count > 0:
                summary = (
                    f"I found {device_count} device(s) in your fabric. "
                    f"Devices: {', '.join(device_names[:5]) }. "
                    f"Types: {', '.join(device_types) }. "
                    f"Status: {', '.join(device_statuses) }"
                )
            else:
                summary = "I couldn't find any devices in your fabric inventory."
            
            return {
                "devices": devices_data,
                "total": device_count,
                "names": device_names,
                "types": device_types,
                "statuses": device_statuses,
                "message": summary
            }
            
        except Exception as e:
            logger.error(f"Error getting fabric devices: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving fabric device information."
            }

    def parse_time_window(self, message: str, now=None) -> tuple:
        """
        Parse natural language time window from user message
        
        Args:
            message: User message containing time information
            now: Optional datetime object representing current time
            
        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        if not now:
            now = datetime.now()
        
        # Default to last 24 hours if no time specified
        default_start = now - timedelta(hours=24)
        default_end = now
        
        # Check for time-related phrases
        time_keywords = [
            'last', 'past', 'previous', 'recent', 'since', 'from', 
            'hour', 'day', 'week', 'month', 'year',
            'yesterday', 'today', 'this week', 'this month'
        ]
        
        has_time_reference = any(keyword in message.lower() for keyword in time_keywords)
        if not has_time_reference:
            return default_start, default_end
            
        # Try dateparser for natural language processing
        try:
            # Handle common patterns
            if 'last' in message.lower() or 'past' in message.lower() or 'previous' in message.lower():
                # "last X days/hours/etc"
                match = re.search(r'\b(?:last|past|previous)\s+(\d+)\s+(hour|day|week|month|year)s?\b', message.lower())
                if match:
                    num = int(match.group(1))
                    unit = match.group(2)
                    if unit == 'hour':
                        start_time = now - timedelta(hours=num)
                    elif unit == 'day':
                        start_time = now - timedelta(days=num)
                    elif unit == 'week':
                        # Approximate months as 30 days
                        start_time = now - timedelta(days=num*7)
                    elif unit == 'month':
                        # Approximate years as 365 days
                        start_time = now - timedelta(days=num*30)
                    elif unit == 'year':
                        # Approximate years as 365 days
                        start_time = now - timedelta(days=num*365)
                    return start_time, now
            
            # Handle "since [date]" or "from [date]" format
            since_match = re.search(r'\b(?:since|from)\s+([^\n]+?)(?:\s+to|\s+until|$)', message.lower())
            if since_match:
                date_str = since_match.group(1)
                start_time = dateparser.parse(date_str)
                if start_time:
                    return start_time, now
            
            # Handle "between [date1] and [date2]" format
            between_match = re.search(r'\bbetween\s+([^\n]+?)\s+(?:and|to)\s+([^\n]+)', message.lower())
            if between_match:
                start_str = between_match.group(1)
                end_str = between_match.group(2)
                start_time = dateparser.parse(start_str)
                end_time = dateparser.parse(end_str)
                if start_time and end_time:
                    return start_time, end_time
        
        except Exception as e:
            logging.warning(f"Error parsing time window: {e}")
        
        # Default fallback
        return default_start, default_end

    def _get_network_anomalies(self, message: str) -> Optional[dict]:
        """
        Redirect to the enhanced anomaly method.
        """
        logger.info(f"[ANOMALY REDIRECT] Calling enhanced anomaly method with message: '{message}'")
        result = self._get_network_anomalies_enhanced(message)
        logger.info(f"[ANOMALY REDIRECT] Enhanced method returned: {result}")
        return result

    def _get_advisories(self, message: str) -> Optional[dict]:
        """
        Get advisories from Nexus Dashboard using a verified endpoint.
        Returns a conversational summary and raw advisories data.
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            # Use verified ND endpoint for advisories (using get_fabrics as a placeholder)
            from agents.verified_endpoints import get_nd_endpoint
            advisories_endpoint = get_nd_endpoint("site_management", "get_fabrics")
            if not advisories_endpoint:
                logger.error("ND advisories endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve advisories."
                }
            logger.info(f"Attempting to get advisories from ND: {advisories_endpoint}")
            response = self.nd_client.get(advisories_endpoint, timeout=10)
            if not response or getattr(response, 'status_code', 500) != 200:
                error_msg = f"Failed to retrieve advisories with status code {getattr(response, 'status_code', 'None')}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve advisories."
                }
            advisories_data = response.json()
            logger.info(f"Successfully retrieved advisories from ND. Sample: {str(advisories_data)[:200]}...")
            # Compose a conversational summary
            advisory_count = 0
            advisory_names = []
            if isinstance(advisories_data, dict):
                fabrics = advisories_data.get("fabrics") or advisories_data.get("data") or []
                if isinstance(fabrics, list):
                    advisory_count = len(fabrics)
                    advisory_names = [f.get("displayName") or f.get("name") for f in fabrics if isinstance(f, dict)]
            summary = (
                f"I've checked for advisories and found {advisory_count} fabric(s) with data. "
                f"Fabrics: {', '.join(advisory_names[:5]) + ('...' if len(advisory_names) > 5 else '')}" if advisory_count else
                "I couldn't find any advisories in your Nexus Dashboard inventory."
            )
            return {
                "advisories": advisories_data,
                "total": advisory_count,
                "names": advisory_names,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error in _get_advisories (intent: 'advisories'): {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving advisories."
            }

    def _get_flow_rules(self, message: str) -> Optional[dict]:
        """
        Get flow rules from Nexus Dashboard using a verified endpoint.
        Returns a conversational summary and raw flow rules data.
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
           # Use verified ND endpoint for flow rules (using get_fabrics as a placeholder)
            from agents.verified_endpoints import get_nd_endpoint
            flow_rules_endpoint = get_nd_endpoint("site_management", "get_fabrics")
            if not flow_rules_endpoint:
                logger.error("ND flow rules endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve flow rules."
                }
            logger.info(f"Attempting to get flow rules from ND: {flow_rules_endpoint}")
            response = self.nd_client.get(flow_rules_endpoint, timeout=10)
            if not response or getattr(response, 'status_code', 500) != 200:
                error_msg = f"Failed to retrieve flow rules with status code {getattr(response, 'status_code', 'None')}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve flow rules."
                }
            flow_rules_data = response.json()
            logger.info(f"Successfully retrieved flow rules from ND. Sample: {str(flow_rules_data)[:200]}...")
            # Compose a conversational summary
            flow_rule_count = 0
            flow_rule_names = []
            if isinstance(flow_rules_data, dict):
                fabrics = flow_rules_data.get("fabrics") or flow_rules_data.get("data") or []
                if isinstance(fabrics, list):
                    flow_rule_count = len(fabrics)
                    flow_rule_names = [f.get("displayName") or f.get("name") for f in fabrics if isinstance(f, dict)]
            summary = (
                f"I've checked for flow rules and found {flow_rule_count} fabric(s) with data. "
                f"Fabrics: {', '.join(flow_rule_names[:5]) + ('...' if len(flow_rule_names) > 5 else '')}" if flow_rule_count else
                "I couldn't find any flow rules in your Nexus Dashboard inventory."
            )
            return {
                "flow_rules": flow_rules_data,
                "total": flow_rule_count,
                "names": flow_rule_names,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error in _get_flow_rules (intent: 'flow_rules'): {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving flow rules."
            }
            
    def _get_network_endpoints(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Get network endpoints from NDI using a verified endpoint.
        Returns a conversational summary and raw endpoints data.
        """
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }
            # Use verified NDI endpoint for endpoints
            from agents.verified_endpoints import get_ndi_endpoint
            endpoints_endpoint = get_ndi_endpoint("endpoints", "get_all_endpoints")
            if not endpoints_endpoint:
                logger.error("NDI endpoints endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve endpoints information."
                }
            
            logger.info(f"Attempting to get endpoints from NDI: {endpoints_endpoint}")
            response = self.nd_client.get(endpoints_endpoint, timeout=10)
            if not response or getattr(response, 'status_code', 500) != 200:
                error_msg = f"Failed to retrieve endpoints with status code {getattr(response, 'status_code', 'None')}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve endpoints information."
                }
            endpoints_data = response.json()
            logger.info(f"Successfully retrieved endpoints from NDI. Sample: {str(endpoints_data)[:200]}...")
            # Compose a conversational summary
            endpoint_count = 0
            endpoint_names = []
            if isinstance(endpoints_data, dict):
                endpoints = endpoints_data.get("endpoints") or endpoints_data.get("data") or []
                if isinstance(endpoints, list):
                    endpoint_count = len(endpoints)
                    endpoint_names = [e.get("displayName") or e.get("name") for e in endpoints if isinstance(e, dict)]
            summary = (
                f"I've checked your network and found {endpoint_count} endpoint(s). "
                f"Endpoints: {', '.join(endpoint_names[:5]) + ('...' if len(endpoint_names) > 5 else '')}" if endpoint_count else
                "I couldn't find any endpoints in your Nexus Dashboard inventory."
            )
            return {
                "endpoints": endpoints_data,
                "total": endpoint_count,
                "names": endpoint_names,
                "message": summary
            }
        except Exception as e:
            logger.error(f"Error in _get_network_endpoints (intent: 'network_endpoints'): {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving endpoints information."
            }
    
    def _get_network_interfaces(self, message: str) -> Optional[Dict[str, Any]]:
        """Get network interfaces information from NDI"""
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard"
                }
            
            # Import the NDI endpoint helper
            from agents.verified_endpoints import get_ndi_endpoint
            
            # Get interfaces data
            interfaces_endpoint = get_ndi_endpoint("interfaces", "get_interfaces")
            if not interfaces_endpoint:
                logger.error("NDI interfaces endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve interfaces information"
                }
                
            logger.info(f"Attempting to get interfaces from NDI: {interfaces_endpoint}")
            
            response = self.nd_client.get(interfaces_endpoint, timeout=10)
            
            if not response or response.status_code != 200:
                error_msg = f"Failed to retrieve interfaces with status code {response.status_code if response else 'None'}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve interfaces information"
                }
                
            # Process the interfaces data
            interfaces_data = response.json()
            logger.info(f"Successfully retrieved interfaces from NDI")
            
            # Debug - log the raw response data structure
            try:
                logger.info(f"Raw interfaces response structure: {json.dumps(interfaces_data, indent=2, default=str)[:500]}...")
            except Exception as e:
                logger.info(f"Could not serialize interfaces response: {str(e)}")
            
            # Try to get interfaces anomaly summary for additional context
            anomaly_summary = None
            try:
                anomaly_endpoint = get_ndi_endpoint("interfaces", "get_interfaces_anomaly_summary")
                if anomaly_endpoint:
                    summary_response = self.nd_client.get(anomaly_endpoint, timeout=10)
                    if summary_response and summary_response.status_code == 200:
                        anomaly_summary = summary_response.json()
                        logger.info("Successfully retrieved interfaces anomaly summary")
            except Exception as e:
                logger.error(f"Error retrieving interfaces anomaly summary: {str(e)}")
            
            # Get alarm count
            alarm_count_endpoint = None  # NDFC logic removed
            if alarm_count_endpoint:
                alarm_count = self.nd_client.get(alarm_count_endpoint)
                if alarm_count and isinstance(alarm_count, dict):
                    anomaly_summary["alarm_count_data"] = alarm_count
                    if "totalAlarmCount" in alarm_count:
                        anomaly_summary["total_alarms"] = alarm_count["totalAlarmCount"]
                    
                    # Extract counts by severity if available
                    for severity in ["critical", "major", "minor", "warning", "info"]:
                        key = f"{severity}AlarmCount"
                        if key in alarm_count:
                            anomaly_summary["alarms_by_severity"][severity] = alarm_count[key]
            
            # Get alarms by device category
            category_endpoint = None  # NDFC logic removed
            if category_endpoint:
                category_data = self.nd_client.get(category_endpoint)
                if category_data and isinstance(category_data, dict):
                    anomaly_summary["category_data"] = category_data
                    for category, count in category_data.items():
                        if isinstance(count, int):
                            anomaly_summary["alarms_by_category"][category] = count
            
            # Get recent alarms
            alarms_endpoint = None  # NDFC logic removed
            if alarms_endpoint:
                alarms_data = self.nd_client.get(alarms_endpoint)
                if alarms_data and isinstance(alarms_data, list):
                    # Take the 5 most recent alarms
                    anomaly_summary["recent_alarms"] = alarms_data[:5]
            
            return anomaly_summary
        except Exception as e:
            logger.error(f"Error getting alarm summary: {str(e)}")
            return None
    
    
    def _get_network_events(self, message: str):
        """Get network events information from NDI"""
        try:
            # Check if credentials are valid
            if not self.nd_client or not self.nd_client.check_auth():
                logger.error("Not authenticated to NDI")
                return {
                    "error": "Authentication failed",
                    "message": "Please ensure your NDI credentials are correct."
                }
                
            # Import the NDI endpoint helper
            from agents.verified_endpoints import get_ndi_endpoint
            
            # Get events buckets data first for a summary
            events_buckets_endpoint = get_ndi_endpoint("events", "get_events_buckets")
            if not events_buckets_endpoint:
                logger.error("NDI events buckets endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve events information"
                }
                
            logger.info(f"Attempting to get events buckets from NDI: {events_buckets_endpoint}")
            
            response = self.nd_client.get(events_buckets_endpoint, timeout=10)
            
            if not response or response.status_code != 200:
                error_msg = f"Failed to retrieve events buckets with status code {response.status_code if response else 'None'}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "message": "Unable to retrieve events information"
                }
                
            # Process the events buckets data
            events_buckets_data = response.json()
            logger.info(f"Successfully retrieved events buckets from NDI")
            
            # Debug - log the raw response data structure
            try:
                logger.info(f"Raw events buckets response structure: {json.dumps(events_buckets_data, indent=2, default=str)[:500]}...")
            except Exception as e:
                logger.info(f"Could not serialize events buckets response: {str(e)}")
            
            # Try to get events details for additional context
            events_details = None
            try:
                details_endpoint = get_ndi_endpoint("events", "get_events_details")
                if details_endpoint:
                    details_response = self.nd_client.get(details_endpoint, timeout=10)
                    if details_response and details_response.status_code == 200:
                        events_details = details_response.json()
                        logger.info("Successfully retrieved events details")
            except Exception as e:
                logger.warning(f"Error retrieving events details: {str(e)}")
            
            # Try to get events summary for additional context
            events_summary = None
            try:
                summary_endpoint = get_ndi_endpoint("events", "get_events_summary")
                if summary_endpoint:
                    summary_response = self.nd_client.get(summary_endpoint, timeout=10)
                    if summary_response and summary_response.status_code == 200:
                        events_summary = summary_response.json()
                        logger.info("Successfully retrieved events summary")
            except Exception as e:
                logger.warning(f"Error retrieving events summary: {str(e)}")
            
            # Format the data for the LLM
            return {
                "events": events_buckets_data,
                "details": events_details,
                "summary": events_summary
            }
            
        except Exception as e:
            logger.error(f"Error in _get_network_events (intent: 'network_events'): {str(e)}")
            return {
                "error": str(e),
                "message": "Unable to retrieve events information due to an error."
            }
    
    def _get_event_history(self, message: str) -> Optional[Dict[str, Any]]:
        """Get event history"""
        # Early return if client is not initialized
        if not self.nd_client:
            logger.error("No ND client initialized")
            return {"message": "Unable to retrieve events information: ND client not initialized."}

        try:
            # Use verified endpoint to get events
            events_endpoint = None  # NDFC logic removed
            
            if not events_endpoint:
                # Fallback to a known endpoint if not found in verified endpoints
                events_endpoint = "/nexus/infra/api/eventmonitoring/v1/eventrecords"
                logger.info(f"Using fallback endpoint for events: {events_endpoint}")
                
            # Make API call
            response = self.nd_client.get(events_endpoint)
            
            # Check if response is valid
            if not response:
                logger.warning(f"No response received from {events_endpoint}")
                return {"events": [], "message": "No response from events endpoint"}
                
            # Check status code
            if not hasattr(response, 'status_code') or response.status_code >= 400:
                status = getattr(response, 'status_code', 'unknown')
                logger.warning(f"Bad response status from {events_endpoint}: {status}")
                return {"events": [], "message": f"Error retrieving events (status: {status})"}
                
            # Extract data from response
            try:
                events_data = response.json()
            except Exception as json_err:
                logger.error(f"Failed to parse JSON from events response: {str(json_err)}")
                return {"events": [], "message": "Failed to parse events data"}
                
            # Check if we got valid data
            if not events_data:
                logger.info("No events found in response")
                return {"events": [], "message": "No events found"}
                
            # Format the data for the LLM
            return {
                "events": events_data,
                "count": len(events_data) if isinstance(events_data, list) else 0,
                "message": "Retrieved events successfully"
            }
        except Exception as e:
            logger.error(f"Error in _get_event_history (intent: 'event_history'): {str(e)}")
            return {"error": str(e), "message": "An error occurred while retrieving event history."}

    def _get_interface_errors(self, message: str) -> Dict[str, Any]:
        """Get interface errors across the fabric"""
        logger.info("Retrieving interface errors")
        
        try:
            if not self.nd_client:
                return {"error": "Not authenticated", "message": "Please authenticate first."}
            
            # Try to get interface errors from various endpoints
            interface_data = {"interfaces": [], "total_errors": 0}
            
            # Try NDI endpoint first
            try:
                from agents.verified_endpoints import get_ndi_endpoint
                ndi_endpoint = get_ndi_endpoint("interfaces", "get_interfaces_errors")
                if ndi_endpoint:
                    response = self.nd_client.get(ndi_endpoint, timeout=10)
                    if response and response.status_code == 200:
                        interface_data = response.json()
                        logger.info("Successfully retrieved interface errors from NDI")
            except Exception as e:
                logger.warning(f"NDI endpoint failed: {str(e)}")
            
            # Fallback to standard endpoint
            if not interface_data.get("interfaces"):
                response = self.nd_client.get("/api/v2/interfaces/errors", timeout=10)
                if response and response.status_code == 200:
                    interface_data = response.json()
                    logger.info("Successfully retrieved interface errors from standard endpoint")
            
            # Process the data
            total_errors = 0
            error_interfaces = []
            
            if isinstance(interface_data, dict) and "interfaces" in interface_data:
                interfaces = interface_data["interfaces"]
                for interface in interfaces:
                    if interface.get("errorCount", 0) > 0:
                        error_interfaces.append({
                            "name": interface.get("name", "Unknown"),
                            "device": interface.get("device", "Unknown"),
                            "errorCount": interface.get("errorCount", 0),
                            "lastError": interface.get("lastError", "Unknown")
                        })
                        total_errors += interface.get("errorCount", 0)
            
            logger.info(f"Found {len(error_interfaces)} interfaces with errors")
            return {
                "summary": f"Found {len(error_interfaces)} interfaces with a total of {total_errors} errors.",
                "interfaces": error_interfaces,
                "total_errors": total_errors,
                "message": f"I found {len(error_interfaces)} interfaces with errors, totaling {total_errors} errors across the fabric."
            }
            
        except Exception as e:
            logger.error(f"Error in _get_interface_errors (intent: 'interface_errors'): {str(e)}")
            return {"error": str(e), "message": "An error occurred while retrieving interface errors."}
    
    def _get_system_info(self, message: str) -> Dict[str, Any]:
        """Get system information for Nexus Dashboard"""
        logger.info("Retrieving system information")
        
        try:
            if not self.nd_client:
                return {"error": "Not authenticated", "message": "Please authenticate first."}
            
            # Try multiple endpoints for system info
            system_data = {}
            
            # Try standard system info endpoint
            try:
                response = self.nd_client.get("/api/v2/system/info", timeout=10)
                if response and response.status_code == 200:
                    system_data = response.json()
                    logger.info("Successfully retrieved system information")
            except Exception as e:
                logger.warning(f"Standard system info endpoint failed: {str(e)}")
            
            # Try alternative endpoint
            if not system_data:
                try:
                    response = self.nd_client.get("/api/v1/system", timeout=10)
                    if response and response.status_code == 200:
                        system_data = response.json()
                        logger.info("Successfully retrieved system information from alternative endpoint")
                except Exception as e:
                    logger.warning(f"Alternative system info endpoint failed: {str(e)}")
            
            if system_data:
                # Extract key information
                version = system_data.get("version", "Unknown")
                uptime = system_data.get("uptime", "Unknown")
                cluster_size = system_data.get("clusterSize", system_data.get("nodes", 1))
                
                return {
                    "summary": f"System version: {version}, Uptime: {uptime}, Cluster size: {cluster_size} nodes",
                    "data": system_data,
                    "message": f"Your Nexus Dashboard is running version {version} with {cluster_size} nodes in the cluster."
                }
            else:
                return {"error": "No data", "message": "Unable to retrieve system information from any endpoint."}
                
        except Exception as e:
            logger.error(f"Error in _get_system_info (intent: 'system_info'): {str(e)}")
            return {"error": str(e), "message": "An error occurred while retrieving system information."}
    
    def _get_alarm_summary(self, message: str) -> Dict[str, Any]:
        """Get alarm summary for the network"""
        logger.info("Retrieving alarm summary")
        
        try:
            if not self.nd_client:
                return {"error": "Not authenticated", "message": "Please authenticate first."}
            
            # Try to get alarm summary
            alarm_data = {}
            
            # Try NDI alarms endpoint
            try:
                from agents.verified_endpoints import get_ndi_endpoint
                ndi_endpoint = get_ndi_endpoint("alarms", "get_alarms_summary")
                if ndi_endpoint:
                    response = self.nd_client.get(ndi_endpoint, timeout=10)
                    if response and response.status_code == 200:
                        alarm_data = response.json()
                        logger.info("Successfully retrieved alarm summary from NDI")
            except Exception as e:
                logger.warning(f"NDI alarms endpoint failed: {str(e)}")
            
            # Fallback to standard endpoint
            if not alarm_data:
                try:
                    response = self.nd_client.get("/api/v2/alarms/summary", timeout=10)
                    if response and response.status_code == 200:
                        alarm_data = response.json()
                        logger.info("Successfully retrieved alarm summary from standard endpoint")
                except Exception as e:
                    logger.warning(f"Standard alarms endpoint failed: {str(e)}")
            
            # Try alternative endpoint
            if not alarm_data:
                try:
                    response = self.nd_client.get("/api/v1/alarms", timeout=10)
                    if response and response.status_code == 200:
                        alarm_data = response.json()
                        logger.info("Successfully retrieved alarms from alternative endpoint")
                except Exception as e:
                    logger.warning(f"Alternative alarms endpoint failed: {str(e)}")
            
            if alarm_data:
                # Process alarm data
                total_alarms = alarm_data.get("total", len(alarm_data.get("alarms", [])))
                critical_alarms = alarm_data.get("critical", 0)
                major_alarms = alarm_data.get("major", 0)
                minor_alarms = alarm_data.get("minor", 0)
                
                # Build summary message
                alarm_breakdown = []
                if critical_alarms > 0:
                    alarm_breakdown.append(f"{critical_alarms} critical")
                if major_alarms > 0:
                    alarm_breakdown.append(f"{major_alarms} major")
                if minor_alarms > 0:
                    alarm_breakdown.append(f"{minor_alarms} minor")
                
                breakdown_str = ", ".join(alarm_breakdown) if alarm_breakdown else "no severity breakdown available"
                
                return {
                    "summary": f"Found {total_alarms} active alarms: {breakdown_str}",
                    "data": alarm_data,
                    "total": total_alarms,
                    "critical": critical_alarms,
                    "major": major_alarms,
                    "minor": minor_alarms,
                    "message": f"I found {total_alarms} active alarms in your network. " +
                              (f"Breaking it down: {breakdown_str}." if alarm_breakdown else "")
                }
            else:
                return {"error": "No data", "message": "Unable to retrieve alarm information from any endpoint."}
                
        except Exception as e:
            logger.error(f"Error in _get_alarm_summary: {str(e)}")
            return {"error": str(e), "message": "An error occurred while retrieving alarm summary."}

    def _get_network_routes(self, message: str) -> Dict[str, Any]:
        """Get network routes information including VRF details"""
        logger.info("Retrieving network routes")
        
        try:
            if not self.nd_client:
                return {"error": "Not authenticated", "message": "Please authenticate first."}
            
            # Try to get routes from various endpoints
            routes_data = {"routes": [], "total": 0}
            
            # Try NDI endpoint first
            try:
                from agents.verified_endpoints import get_ndi_endpoint
                ndi_endpoint = get_ndi_endpoint("routes", "get_routes")
                if ndi_endpoint:
                    response = self.nd_client.get(ndi_endpoint, timeout=10)
                    if response and response.status_code == 200:
                        routes_data = response.json()
                        logger.info("Successfully retrieved routes from NDI")
            except Exception as e:
                logger.warning(f"NDI routes endpoint failed: {str(e)}")
            
            # Fallback to standard endpoints
            if not routes_data.get("routes"):
                try:
                    response = self.nd_client.get("/api/v2/network/routes", timeout=10)
                    if response and response.status_code == 200:
                        routes_data = response.json()
                        logger.info("Successfully retrieved routes from standard endpoint")
                except Exception as e:
                    logger.warning(f"Standard routes endpoint failed: {str(e)}")
            
            # Try alternative endpoint
            if not routes_data.get("routes"):
                try:
                    response = self.nd_client.get("/api/v1/routes", timeout=10)
                    if response and response.status_code == 200:
                        routes_data = response.json()
                        logger.info("Successfully retrieved routes from alternative endpoint")
                except Exception as e:
                    logger.warning(f"Alternative routes endpoint failed: {str(e)}")
            
            # Process routes data
            total_routes = 0
            vrf_summary = {}
            route_list = []
            
            if isinstance(routes_data, dict) and "routes" in routes_data:
                routes = routes_data["routes"]
                total_routes = len(routes)
                
                for route in routes:
                    if isinstance(route, dict):
                        vrf = route.get("vrf", "default")
                        if vrf not in vrf_summary:
                            vrf_summary[vrf] = 0
                        vrf_summary[vrf] += 1
                        
                        route_list.append({
                            "prefix": route.get("prefix", "Unknown"),
                            "vrf": vrf,
                            "nextHop": route.get("nextHop", route.get("next_hop", "Unknown")),
                            "interface": route.get("interface", "Unknown"),
                            "metric": route.get("metric", 0)
                        })
            
            # Build summary message
            vrf_breakdown = ", ".join([f"{vrf}: {count} routes" for vrf, count in vrf_summary.items()])
            
            logger.info(f"Found {total_routes} routes across {len(vrf_summary)} VRFs")
            return {
                "summary": f"Found {total_routes} routes across {len(vrf_summary)} VRFs",
                "routes": route_list[:100],  # Limit to first 100 routes
                "total": total_routes,
                "vrfs": vrf_summary,
                "message": f"I found {total_routes} routes in your network. " +
                          (f"VRF breakdown: {vrf_breakdown}." if vrf_breakdown else "")
            }
            
        except Exception as e:
            logger.error(f"Error in _get_network_routes: {str(e)}")
            return {"error": str(e), "message": "An error occurred while retrieving network routes."}

    def analyze_with_structure(self, data: Dict[str, Any], intent: str) -> Optional[Dict[str, Any]]:
        """
        Analyze data and return structured output using Pydantic models
        
        Args:
            data: Raw data from API
            intent: The intent type
            
        Returns:
            Structured analysis result
        """
        try:
            if intent in ["fabric_health", "network_topology"]:
                # Generate structured fabric health analysis
                prompt = f"""Analyze this network fabric data and provide a structured health assessment:

{json.dumps(data, indent=2)}

Focus on:
1. Overall health status (healthy/minor/major/critical)
2. Health score (0-100)
3. Number of active alarms
4. List of current issues"""
                
                result = self.llm_processor.generate_structured_output(
                    prompt=prompt,
                    output_model=FabricHealthStatus,
                    temperature=0.1
                )
                
                if result:
                    return result.dict()
                    
            elif intent in ["anomalies", "network_anomalies"]:
                # Generate structured network issues list
                prompt = f"""Analyze these network anomalies and create a structured list of issues:

{json.dumps(data, indent=2)}

For each issue provide:
- Unique ID
- Severity (info/warning/error/critical)
- Category
- Description
- Affected devices
- Recommended action"""
                
                issues = []
                for i in range(min(5, len(data.get('anomalies', [])))):  # Limit to 5 issues
                    issue_result = self.llm_processor.generate_structured_output(
                        prompt=prompt,
                        output_model=NetworkIssue,
                        temperature=0.1
                    )
                    if issue_result:
                        issues.append(issue_result.dict())
                
                return {"issues": issues}
                
        except Exception as e:
            logger.error(f"Error generating structured analysis: {e}")
            return None
    
    def generate_troubleshooting_plan(self, issue_description: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a structured troubleshooting plan for a network issue
        
        Args:
            issue_description: Description of the issue
            context: Additional context (affected devices, symptoms, etc.)
            
        Returns:
            Structured troubleshooting plan
        """
        try:
            prompt = f"""Create a detailed troubleshooting plan for this network issue:

Issue: {issue_description}

Context:
{json.dumps(context, indent=2)}

Generate a step-by-step plan with:
- Clear action items
- CLI commands where applicable
- Expected results
- Verification methods

Focus on Cisco Nexus environment."""
            
            result = self.llm_processor.generate_structured_output(
                prompt=prompt,
                output_model=TroubleshootingPlan,
                temperature=0.2
            )
            
            if result:
                return result.dict()
                
        except Exception as e:
            logger.error(f"Error generating troubleshooting plan: {e}")
            return None
    
    def classify_intent_structured(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Use structured output to classify user intent
        
        Args:
            message: User message
            
        Returns:
            Intent classification with confidence and entities
        """
        try:
            # Get list of available intents
            available_intents = list(self.intent_patterns.keys())
            
            # Use LLM to classify intent
            result = self.llm_processor.classify_intent(
                user_query=message,
                available_intents=available_intents,
                context={
                    "environment": "Cisco Nexus Dashboard",
                    "capabilities": "network monitoring, troubleshooting, configuration"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in structured intent classification: {e}")
            return None
    
    def generate_network_summary(self, timeframe: str = "current") -> Optional[Dict[str, Any]]:
        """
        Generate a comprehensive network summary using multiple data sources
        
        Args:
            timeframe: Time period for the summary
            
        Returns:
            Structured network summary
        """
        try:
            # Gather data from multiple sources
            summary_data = {}
            
            # Get fabric health
            # Use the timeframe or a descriptive message for context
            health_message = f"Get fabric health for summary (timeframe: {timeframe})"
            health_data = self._get_fabric_health(health_message)
            if health_data:
                summary_data["health"] = health_data
            
            # Get device inventory
            devices_data = self._get_fabric_devices("")
            if devices_data:
                summary_data["devices"] = devices_data
            
            # Get anomalies
            anomalies_data = self._get_network_anomalies("")
            if anomalies_data:
                summary_data["anomalies"] = anomalies_data
            
            # Generate structured summary
            prompt = f"""Analyze this network data and provide a comprehensive summary:

{json.dumps(summary_data, indent=2)}

Include:
- Total devices and their status
- Interface statistics
- Critical issues count
- Overall health assessment
- Key performance metrics"""
            
            result = self.llm_processor.generate_structured_output(
                prompt=prompt,
                output_model=NetworkSummary,
                temperature=0.1
            )
            
            if result:
                return result.dict()
                
        except Exception as e:
            logger.error(f"Error generating network summary: {e}")
            return None

    def _is_general_query(self, message: str) -> bool:
        """
        Check if the message is a general question or greeting
        
        Args:
            message: User message
            
        Returns:
            True if it's a general query
        """
        message_lower = message.lower().strip()
        
        # Greetings and pleasantries
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                     "howdy", "greetings", "what's up", "how are you", "how's it going"]
        
        # General questions about the assistant
        assistant_questions = ["who are you", "what are you", "what can you do", "help", 
                              "what do you do", "tell me about yourself", "capabilities",
                              "what can i ask", "how do you work", "what features"]
        
        # Thank you messages
        thanks = ["thank you", "thanks", "appreciate it", "that helps", "perfect", "great"]
        
        # Goodbyes
        goodbyes = ["bye", "goodbye", "see you", "later", "farewell", "take care"]
        
        # Check for matches
        for greeting in greetings:
            if greeting in message_lower:
                return True
                
        for question in assistant_questions:
            if question in message_lower:
                return True
                
        for thank in thanks:
            if thank in message_lower:
                return True
                
        for goodbye in goodbyes:
            if goodbye in message_lower:
                return True
        
        # Check for very short messages that might be general
        if len(message.split()) <= 2 and "?" in message:
            return True
            
        return False
    
    def _handle_general_query(self, message: str) -> str:
        """
        Handle general queries with human-like responses
        
        Args:
            message: User message
            
        Returns:
            Appropriate response
        """
        message_lower = message.lower().strip()
        
        # Handle greetings
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "howdy"]):
            greetings = [
                "Hey there! I'm your Nexus Dashboard assistant. How can I help you with your network today?",
                "Hello! Ready to help you monitor and troubleshoot your network. What would you like to know?",
                "Hi! I'm here to help with your Cisco Nexus environment. What can I do for you?",
                "Hey! Let's take a look at your network. What information do you need?"
            ]
            import random
            return random.choice(greetings)
        
        # Handle capability questions
        if any(q in message_lower for q in ["what can you do", "help", "capabilities", "what do you do"]):
            return """I'm your AI-powered Nexus Dashboard assistant! Here's what I can help you with:

📊 **Network Monitoring**
- Check fabric health and device status
- Monitor interface statistics and errors
- View network topology and connections
- Track performance metrics and anomalies

🔍 **Troubleshooting**
- Analyze network issues and anomalies
- Generate troubleshooting plans
- Suggest CLI commands for diagnostics
- Review event history and alarms

📈 **Analytics & Insights**
- Identify traffic patterns and top talkers
- Detect security advisories and compliance issues
- Analyze flow rules and configurations
- Provide network summaries and reports

💬 **Natural Conversation**
- Answer general networking questions
- Explain Cisco technologies and best practices
- Help with configuration guidance
- Provide context-aware recommendations

Just ask me anything about your network, and I'll do my best to help! For example:
- "Show me the health of my fabric"
- "Why is spine-01 showing high CPU?"
- "What interfaces have errors?"
- "Generate a troubleshooting plan for VLAN issues"
"""
        
        # Import required modules at the beginning of the method
        import random
        
        # Handle thank you
        if any(thank in message_lower for thank in ["thank you", "thanks", "appreciate"]):
            responses = [
                "You're welcome! Let me know if you need anything else.",
                "Happy to help! Is there anything else you'd like to check?",
                "No problem at all! Feel free to ask if you have more questions.",
                "Glad I could help! Your network looking good?"
            ]
            return random.choice(responses)
        
        # Handle goodbyes
        if any(bye in message_lower for bye in ["bye", "goodbye", "see you"]):
            responses = [
                "Take care! I'll be here whenever you need to check on your network.",
                "Goodbye! Hope your network stays healthy!",
                "See you later! Don't hesitate to come back if you need help.",
                "Bye! Keep those packets flowing smoothly!"
            ]
            return random.choice(responses)
        
        # Handle "how are you" type questions
        if "how are you" in message_lower or "how's it going" in message_lower:
            return "I'm doing great, thanks for asking! All systems operational here. How can I help you with your network today?"
        
        # Default response for other general queries
        return self.llm_processor.complete(
            system_prompt="You are a friendly and knowledgeable Nexus Dashboard AI assistant. Respond to this general query in a helpful, conversational way. Keep it brief but informative. If relevant, mention your network monitoring capabilities.",
            conversation=[{"role": "user", "content": message}],
            max_tokens=200
        )
    
    def _handle_unclear_intent(self, message: str, guessed_intent: str, confidence: float) -> str:
        """
        Handle cases where intent is unclear
        
        Args:
            message: User message
            guessed_intent: Best guess intent
            confidence: Confidence score
            
        Returns
        
        Example format:
        "Hmm, I want to make sure I understand what you're looking for. Are you asking about [interpretation 1], or perhaps you want to [interpretation 2]? Just let me know a bit more and I'll get you the right information!"
        """
        
        # Create a prompt for clarification
        prompt = f"""The user asked: '{message}'

I'm not confident about what they're asking (confidence: {confidence}).

Please generate a friendly response that:
1. Acknowledges that I'm not sure what they're asking
2. Offers 2-3 possible interpretations of their question
3. Asks them to clarify what they're looking for
4. Keeps it brief and conversational
"""
        
        # Use the summarize_api_response method with empty data as a workaround
        # since there's no direct generate_response method
        try:
            if hasattr(self.llm_processor, 'summarize_api_response'):
                response = self.llm_processor.summarize_api_response(
                    response_data={"user_message": message, "confidence": confidence},
                    intent="clarification",
                    user_query=message,
                    system_prompt="You are a helpful network assistant. Be conversational and friendly."
                )
                if response and not response.startswith('[LLM ERROR]'):
                    return response
        except Exception as e:
            logger.error(f"Error using LLM for clarification: {e}")
        
        # Fallback if LLM fails
        return f"I'm not quite sure what you're asking about. Could you provide a bit more detail? For example, are you looking for fabric health, device status, or something else?"

    def _generate_follow_up(self, intent: str, data: Any) -> Optional[str]:
        """
        Generate contextual follow-up questions based on current interaction.
        
        Args:
            intent: Current intent
            data: Data retrieved for the intent
            
        Returns:
            Follow-up question or None
        """
        if not data:
            return None
            
        follow_ups = {
            "fabric_health": [
                "Would you like me to check on any specific devices?",
                "Should I look into the recent events or alarms?",
                "Want me to analyze what might be causing any issues?"
            ],
            "device_status": [
                "Need me to check the interfaces on any of these devices?",
                "Should I look at the CPU and memory usage trends?",
                "Check for recent configuration changes?"
            ],
            "interfaces": [
                "Should I check for any error patterns?",
                "Want me to look at the traffic statistics?",
                "Need details on any specific interface?"
            ],
            "anomalies": [
                "Would you like a troubleshooting plan for any of these issues?",
                "Should I check which devices are most affected?",
                "Want me to look at the historical trends?"
            ],
            "events": [
                "Need more details about any specific event?",
                "Should I check what triggered these events?",
                "Want to see events from a specific time period?"
            ]
        }
        
        # Import required modules at the beginning of the method
        import random
        
        if intent in follow_ups:
            return random.choice(follow_ups[intent])
            
        return None
    
    def _generate_suggestions(self, intent: str, context_data: Any) -> List[str]:
        """
        Generate suggested actions for the user
        
        Args:
            intent: The identified intent
            context_data: Additional context data for generating suggestions
            
        Returns:
            List of suggested actions
        """
        # Implement suggestion generation logic here
        suggestions = []
        
        # Intent-specific suggestions
        intent_suggestions = {
            "interface_stats": [
                "Check error counters",
                "View traffic statistics",
                "Show interface events"
            ],
            "anomalies": [
                "Get troubleshooting plan",
                "Check affected devices",
                "View event history"
            ]
        }
        
        if intent in intent_suggestions:
            suggestions.extend(intent_suggestions[intent])
        
        # Add general suggestions
        general_suggestions = [
            "Show fabric health",
            "Check for anomalies",
            "View network summary"
        ]
        
        # Return up to 3 suggestions, avoiding duplicates
        all_suggestions = list(set(suggestions + general_suggestions))
        return all_suggestions[:3]  # Limit to 3 suggestions for clarity
    
    def _update_context_intents(self, intent_name, confidence):
        """
        Update the conversation context with detected intents
        """
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        # Initialize intents list if not exists
        if not hasattr(self.context, 'intent_history') or self.context.intent_history is None:
            self.context.intent_history = []
            
        # Add the new intent to history if it's valid
        if intent_name and confidence > 0.0:
            # Add new intent with timestamp and confidence score
            intent_record = {
                'intent': intent_name,
                'confidence': confidence,
                'timestamp': current_time
            }
            
            # Add to history but keep only last 10 intents to avoid bloat
            self.context.intent_history.append(intent_record)
            if len(self.context.intent_history) > 10:
                self.context.intent_history = self.context.intent_history[-10:]
                
            # Also track the last detected intent separately for easy access
            self.context.last_intent = intent_record
            
            logger.debug(f"Updated context with intent: {intent_name} (confidence: {confidence:.2f})")
        
    def _initialize_intent_patterns(self):
        """
        Initialize intent patterns from YAML files
        """
        try:
            from intent_loader import load_intents
            from agents.intent_registry import get_intent_metadata
            
            # Load intents from YAML files
            logger.info("Loading intents from YAML files")
            loaded_intents = load_intents()
            
            # Create intent patterns dictionary
            self.intent_patterns = {}
            self.intent_priorities = {}            # Generate pattern lists for each intent
            for intent_name, intent_spec in loaded_intents.items():
                patterns = []
                
                # Store priority if defined in YAML
                if hasattr(intent_spec, 'priority') and intent_spec.priority > 0:
                    self.intent_priorities[intent_name] = intent_spec.priority
                    logger.info(f"Intent '{intent_name}' has custom priority: {intent_spec.priority}")
                
                # Use custom patterns if defined in YAML
                if hasattr(intent_spec, 'patterns') and intent_spec.patterns:
                    patterns.extend([p.lower() for p in intent_spec.patterns])
                    logger.info(f"Intent '{intent_name}' has {len(intent_spec.patterns)} custom patterns from YAML")
                else:
                    # Generate patterns from description if no custom patterns
                    # Get metadata from intent registry
                    metadata = get_intent_metadata(intent_name) or {}
                
                # Add patterns from description and metadata
                description = intent_spec.description.lower()
                if description:
                    # Add full description as a pattern
                    patterns.append(description)
                    
                    # Add key phrases from description
                    words = description.split()
                    if len(words) > 2:
                        patterns.append(" ".join(words[:3]))
                
                # Add intent name variations
                name_parts = intent_name.split('_')
                patterns.append(intent_name.replace('_', ' '))
                patterns.append(' '.join(name_parts))
                
                # Add common question patterns
                patterns.append(f"show {intent_name.replace('_', ' ')}")
                patterns.append(f"get {intent_name.replace('_', ' ')}")
                patterns.append(f"what is {intent_name.replace('_', ' ')}")
                
                # Store unique patterns
                self.intent_patterns[intent_name] = list(set(patterns))
                
            logger.info(f"Loaded {len(self.intent_patterns)} intent patterns")
            
        except Exception as e:
            logger.error(f"Failed to initialize intent patterns: {str(e)}")
            logger.error(traceback.format_exc())
            # Set fallback patterns
            self.intent_patterns = {
                "fabric_devices": ["fabric devices", "devices in fabric", "show devices", "list devices"],
                "device_status": ["device status", "status of device", "switch status"],
                "network_topology": ["topology", "network map", "infrastructure", "network layout"],
                "interface_stats": [
                    "Check error counters",
                    "View traffic statistics",
                    "Show interface events"
                ],
                "anomalies": [
                    "Get troubleshooting plan",
                    "Check affected devices",
                    "View event history"
                ]
            }


    def _enhanced_intent_recognition(self, message: str):
        """
        Enhanced hybrid intent recognition that combines multiple techniques.
        
        Uses a combination of pattern matching, keyword matching, similarity matching,
        and optionally LLM-based classification for more accurate intent detection.
        
        Args:
            message: User message to analyze
            
        Returns:
            Dictionary with intent name as key and confidence as value,
            possibly containing multiple intents
        """
        results = {}
        message_lower = message.lower()
        
        # 1. Pattern matching (existing approach) - give this the highest weight
        try:
            pattern_results = self._pattern_match_intent(message)
            # Safely check if pattern_results is a tuple containing an intent and score
            if (isinstance(pattern_results, tuple) and 
                len(pattern_results) >= 2 and 
                isinstance(pattern_results[0], str) and 
                isinstance(pattern_results[1], (int, float))):
                
                intent, score = pattern_results
                # Give much higher weight to pattern matches and boost the score
                boosted_score = min(1.0, score * 2.5)  # Boost pattern matches significantly
                results[intent] = boosted_score
                logger.info(f"Pattern match found intent '{intent}' with score {score} (boosted to {boosted_score})")
            else:
                logger.warning(f"Pattern matching returned unexpected result: {pattern_results}")
        except Exception as e:
            logger.error(f"Error in pattern matching: {e}")
            # Continue with other methods
            
        # 2. Keyword matching (more flexible than pattern matching)
        keywords = self._extract_keywords(message_lower)
        for intent, patterns in self.intent_patterns.items():
            intent_keywords = set()
            for pattern in patterns:
                intent_keywords.update(pattern.lower().split())
                
            # Calculate overlap between message keywords and intent keywords
            overlap = len(keywords.intersection(intent_keywords))
            if overlap > 0:
                keyword_score = overlap / max(len(keywords), 1) * 0.6  # Reduced weight
                # Only update if we don't already have a higher pattern match score
                if intent not in results or results[intent] < keyword_score:
                    results[intent] = keyword_score
        # 3. Example similarity matching - reduced weight
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                similarity = self._calculate_text_similarity(message_lower, pattern.lower())
                if similarity > 0.7:  # Only consider very high similarities
                    similarity_score = similarity * 0.5  # Reduced weight
                    # Only update if we don't already have a higher score
                    if intent not in results or results[intent] < similarity_score:
                        results[intent] = similarity_score
        # 4. Optional LLM-based intent classification if confidence is low
        max_confidence = max(results.values()) if results else 0
        if max_confidence < 0.5 and self.llm_processor:  # Raised threshold
            try:
                llm_intent = self._llm_classify_intent(message)
                if llm_intent and llm_intent in self.intent_patterns:
                    results[llm_intent] = 0.8  # Higher confidence for LLM classification
            except Exception as e:
                logger.warning(f"Error in LLM intent classification: {str(e)}")
                
        # Filter out low confidence scores and normalize remaining ones
        results = {k: v for k, v in results.items() if v > 0.15}  # Lower threshold
        if results:
            # Check if we have a high-confidence match - if so, don't normalize
            max_score = max(results.values())
            if max_score >= 0.5:  # High confidence match - keep original scores
                logger.info(f"High confidence match found ({max_score:.3f}) - keeping original scores")
            else:
                # Only normalize if all scores are relatively low
                total = sum(results.values())
                results = {k: v/total for k, v in results.items()}
            
        logger.info(f"Enhanced intent recognition for '{message}': {results}")
        return results
    
    def _extract_keywords(self, text):
        """Extract keywords from text by removing stopwords."""
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'with', 'by', 'for', 'is', 'are', 'am', 'was', 'were', 'be', 'been',
                    'has', 'have', 'had', 'do', 'does', 'did', 'to', 'at', 'in', 'on',
                    'of', 'from', 'my', 'our', 'your', 'it', 'its', 'this', 'that',
                    'these', 'those', 'me', 'him', 'her', 'them', 'i', 'we', 'you', 'he',
                    'she', 'they', 'how', 'when', 'where', 'why', 'which', 'who'}
                    
        words = text.lower().split()
        keywords = {word for word in words if word not in stopwords}
        return keywords
        
    def _calculate_text_similarity(self, text1, text2):
        """Calculate simple text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _llm_classify_intent(self, message):
        """Use LLM to classify intent when other methods have low confidence."""
        if not self.llm_processor:
            return None
            
        # Create a prompt that asks the LLM to classify the intent
        prompt = f"""Determine the intent of this user message for a network management chatbot.

User message: "{message}"

Available intents: {', '.join(self.intent_patterns.keys())}

Return just the single most likely intent name without explanation."""
        
        try:
            # Generate a simple response containing just the intent name
            response = self.llm_processor.complete(
                system_prompt=prompt,
                conversation=[{"role": "user", "content": message}],
                max_tokens=50
            )
            # Extract just the intent name (remove any extra text)
            response = response.strip().lower()
            
            # Find the closest matching intent if the response isn't an exact match
            best_match = None
            best_score = 0
            
            for intent in self.intent_patterns.keys():
                if intent.lower() in response:
                    return intent  # Direct match found
                
                # Calculate similarity for fuzzy matching
                score = self._calculate_text_similarity(intent.lower(), response)
                if score > best_score and score > 0.5:  # Require minimum similarity
                    best_score = score
                    best_match = intent
            
            return best_match  # Will be None if no reasonable match found
            
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {str(e)}")
            return None

    def _api_call_with_retry(self, endpoint, params=None, max_retries=3, cache_duration=300):
        """
        Enhanced API call with automatic retries, caching, and telemetry.
        
        Args:
            endpoint: API endpoint to call
            params: Parameters for the API call
            max_retries: Maximum number of retry attempts
            cache_duration: Cache duration in seconds
            
        Returns:
            API response data or None if failed
        """
        import time
        import hashlib
        
        # Create cache key
        cache_key = hashlib.md5(f"{endpoint}_{str(params)}".encode()).hexdigest()
        
        # Check cache first
        if hasattr(self, '_api_cache'):
            cached_result = self._api_cache.get(cache_key)
            if cached_result:
                cache_time, data = cached_result
                if time.time() - cache_time < cache_duration:
                    logger.debug(f"Cache hit for {endpoint}")
                    return data
        else:
            self._api_cache = {}
            
        # Attempt API call with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if not self.nd_client:
                    logger.error("ND client not initialized")
                    return None
                    
                # Make the API call
                response = self.nd_client.get(endpoint, params=params, timeout=30)
                
                if response and hasattr(response, 'status_code') and response.status_code == 200:
                    data = response.json()
                    
                    # Cache successful response
                    self._api_cache[cache_key] = (time.time(), data)
                    
                    logger.info(f"API call successful: {endpoint} (attempt {attempt + 1})")
                    return data
                else:
                    raise Exception(f"API returned status {getattr(response, 'status_code', 'unknown')}")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries + 1} attempts: {str(e)}")
                    
        return None
        
    def _normalize_api_response(self, response, endpoint_type):
        """
        Normalize API responses to consistent format.
        
        Args:
            response: Raw API response
            endpoint_type: Type of endpoint (fabric_health, network_topology, etc.)
            
        Returns:
            Normalized response dictionary
        """
        if not response:
            return {"items": [], "total": 0, "status": "error", "message": "No data available"}
            
        # Handle different response formats
        normalized = {
            "items": [],
            "total": 0,
            "status": "success",
            "endpoint_type": endpoint_type,
            "timestamp": time.time()
        }
        
        try:
            # Extract items based on common patterns
            if isinstance(response, dict):
                # Try common item keys
                for key in ['items', 'data', 'results', 'nodes', 'devices', 'anomalies']:
                    if key in response and isinstance(response[key], list):
                        normalized["items"] = response[key]
                        normalized["total"] = len(response[key])
                        break
                        
                # If no list found, treat the whole response as a single item
                if not normalized["items"] and response:
                    normalized["items"] = [response]
                    normalized["total"] = 1
                    
                # Copy metadata
                for meta_key in ['total', 'count', 'totalCount']:
                    if meta_key in response:
                        normalized["total"] = response[meta_key]
                        break
                        
            elif isinstance(response, list):
                normalized["items"] = response
                normalized["total"] = len(response)
                
        except Exception as e:
            logger.error(f"Error normalizing API response: {str(e)}")
            normalized["status"] = "error"
            normalized["message"] = f"Error processing response: {str(e)}"
            
        return normalized

    def _explain_anomaly(self, anomaly_data):
        """
        Provide human-readable explanations for network anomalies.
        
        Args:
            anomaly_data: Dictionary containing anomaly information
            
        Returns:
            Dictionary with explanation and recommendations
        """
        if not anomaly_data:
            return {"explanation": "No anomaly data available", "recommendations": []}
            
        # Knowledge base for common anomalies
        anomaly_patterns = {
            "high_cpu": {
                "keywords": ["cpu", "processor", "utilization", "load"],
                "explanation": "High CPU utilization detected on network device",
                "recommendations": [
                    "Check for processes consuming excessive CPU",
                    "Review traffic patterns for unusual spikes",
                    "Consider load balancing if persistent",
                    "Monitor for potential security threats"
                ]
            },
            "interface_down": {
                "keywords": ["interface", "down", "link", "port", "connection"],
                "explanation": "Network interface or link is down",
                "recommendations": [
                    "Check physical cable connections",
                    "Verify interface configuration",
                    "Check for hardware failures",
                    "Review logs for error messages"
                ]
            },
            "high_latency": {
                "keywords": ["latency", "delay", "response", "slow"],
                "explanation": "High network latency detected",
                "recommendations": [
                    "Check network congestion",
                    "Verify routing paths",
                    "Monitor bandwidth utilization",
                    "Consider QoS adjustments"
                ]
            },
            "packet_loss": {
                "keywords": ["packet", "loss", "drop", "discard"],
                "explanation": "Packet loss detected in network traffic",
                "recommendations": [
                    "Check interface statistics",
                    "Verify buffer sizes",
                    "Monitor for congestion",
                    "Review error counters"
                ]
            },
            "memory_high": {
                "keywords": ["memory", "ram", "heap", "buffer"],
                "explanation": "High memory utilization on network device",
                "recommendations": [
                    "Check memory-intensive processes",
                    "Review routing table size",
                    "Monitor for memory leaks",
                    "Consider memory upgrade if persistent"
                ]
            }
            }
        
        # Extract anomaly information
        anomaly_text = str(anomaly_data).lower()
        anomaly_type = anomaly_data.get('type', '').lower()
        severity = anomaly_data.get('severity', 'medium').lower()
        
        # Find matching pattern
        matched_pattern = None
        for pattern_name, pattern_info in anomaly_patterns.items():
            if any(keyword in anomaly_text or keyword in anomaly_type 
                   for keyword in pattern_info["keywords"]):
                matched_pattern = pattern_info
                break
                
        if matched_pattern:
            explanation = matched_pattern["explanation"]
            recommendations = matched_pattern["recommendations"]
        else:
            # Generic explanation
            explanation = f"Network anomaly detected: {anomaly_data.get('description', 'Unknown issue')}"
            recommendations = [
                "Review device logs for additional details",
                "Check device health metrics",
                "Monitor the situation for changes",
                "Contact network administrator if issue persists"
            ]
            
        # Enhance with LLM if available
        if self.llm_processor and severity in ['high', 'critical']:
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(anomaly_data, explanation)
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                logger.warning(f"Failed to enhance explanation with LLM: {str(e)}")
                
        return {
            "explanation": explanation,
            "recommendations": recommendations,
            "severity": severity,
            "pattern_matched": matched_pattern is not None
        }
        
    def _enhance_explanation_with_llm(self, anomaly_data, base_explanation):
        """
        Use LLM to enhance anomaly explanations with more context.
        
        Args:
            anomaly_data: Raw anomaly data
            base_explanation: Base explanation from pattern matching
            
        Returns:
            Enhanced explanation string
        """
        if not self.llm_processor:
            return base_explanation
            
        prompt = f"""As a network expert, provide a clear explanation of this network anomaly:

Anomaly Data: {json.dumps(anomaly_data, indent=2)}

Base Explanation: {base_explanation}

Provide a concise, technical explanation that a network administrator would find helpful. Focus on:
1. What the anomaly means
2. Potential root causes
3. Impact on network operations

Keep the response under 150 words and avoid speculation."""

        try:
            enhanced = self.llm_processor.complete(
                system_prompt=prompt,
                conversation=[{"role": "assistant", "content": base_explanation}],
                max_tokens=200
            )
            if enhanced and len(enhanced.strip()) > 20:  # Ensure meaningful response
                return enhanced.strip()
        except Exception as e:
            logger.error(f"Error enhancing explanation with LLM: {str(e)}")
            
        return base_explanation
        
    def _generate_troubleshooting_steps(self, intent, data):
        """
        Generate specific troubleshooting steps based on intent and data.
        
        Args:
            intent: The user's intent (e.g., 'device_health', 'network_issues')
            data: Relevant data for the intent
            
        Returns:
            List of troubleshooting steps
        """
        steps = []
        
        if intent == "device_health" and data:
            # Check for health issues in the data
            items = data.get('items', [])
            unhealthy_devices = [item for item in items 
                               if item.get('healthScore', 100) < 80]
            if unhealthy_devices:
                steps.extend([
                    f"Investigate device {item.get('name', 'unknown')} with health score {item.get('healthScore', 100)}",
                    "Check device logs for error messages",
                    "Verify all interfaces are operational",
                    "Monitor resource utilization (CPU, memory)"
                ])
                    
        elif intent == "network_anomalies" and data:
            items = data.get('items', [])
            for anomaly in items:
                explanation = self._explain_anomaly(anomaly)
                steps.append(f"Address {anomaly.get('type', 'anomaly')}: {explanation['explanation']}")
                steps.extend(explanation['recommendations'][:2])  # Add top 2 recommendations
                
        elif intent == "interface_status" and data:
            items = data.get('items', [])
            down_interfaces = [item for item in items 
                             if item.get('status', '').lower() == 'down']
            if down_interfaces:
                steps.extend([
                    "Check physical connections for down interfaces",
                    "Verify interface configurations",
                    "Review interface error counters",
                    "Test connectivity with ping/traceroute"
                ])
                
        # Add generic steps if no specific ones found
        if not steps:
            steps = [
                "Review the current status and metrics",
                "Check for any error messages or alerts",
                "Verify configuration settings",
                "Monitor the situation for changes"
            ]
            
        return steps[:5]  # Limit to 5 steps for clarity

    def _manage_dialog_flow(self, message, intent_results, context):
        """
        Manage conversation flow and determine if clarification is needed.
        
        Args:
            message: User message
            intent_results: Results from intent recognition
            context: Current conversation context
            
        Returns:
            Dictionary with dialog management decisions
        """
        dialog_state = {
            "needs_clarification": False,
            "clarification_question": None,
            "primary_intent": None,
            "confidence": 0.0,
            "follow_up_suggested": False,
            "context_resolved": True
        }
        
        # Determine primary intent and confidence
        if isinstance(intent_results, dict) and intent_results:
            primary_intent = max(intent_results.items(), key=lambda x: x[1])
            dialog_state["primary_intent"] = primary_intent[0]
            dialog_state["confidence"] = primary_intent[1]
        elif isinstance(intent_results, dict) and not intent_results:
            # Handle empty intent results
            dialog_state["primary_intent"] = "unknown"
            dialog_state["confidence"] = 0.0
            dialog_state["needs_clarification"] = True
            dialog_state["clarification_question"] = "I'm not sure what you're asking about. Could you please rephrase your question or be more specific?"
        elif isinstance(intent_results, tuple):
            dialog_state["primary_intent"] = intent_results[0]
            dialog_state["confidence"] = intent_results[1]
        else:
            dialog_state["confidence"] = 0.0
            
        # Check if clarification is needed
        if dialog_state["confidence"] < 0.3:  # Lowered from 0.6 to 0.3
            logger.info(f"[DEBUG] Triggering clarification due to low confidence: {dialog_state['confidence']}")
            dialog_state["needs_clarification"] = True
            dialog_state["clarification_question"] = self._generate_clarification_question(
                message, intent_results, context
            )
            
        # Check for unresolved references - only for pronouns and ambiguous terms
        elif context and hasattr(context, 'is_reference'):
            logger.info(f"[DEBUG] Checking for unresolved references")
            # Only check for references if the message contains pronouns or ambiguous terms
            ambiguous_terms = ['it', 'this', 'that', 'them', 'they', 'these', 'those']
            message_words = message.lower().split()
            has_ambiguous_terms = any(term in message_words for term in ambiguous_terms)
            logger.info(f"[DEBUG] Has ambiguous terms: {has_ambiguous_terms}")
            
            if has_ambiguous_terms and context.is_reference(message):
                logger.info(f"[DEBUG] Triggering clarification due to unresolved references")
                # Try to resolve references
                resolved = False
                for entity_type in ['device', 'fabric', 'interface', 'anomaly']:
                    if context.resolve_reference(entity_type, message):
                        resolved = True
                        break
                        
                if not resolved:
                    dialog_state["needs_clarification"] = True
                    dialog_state["clarification_question"] = (
                        "I'm not sure which specific item you're referring to. "
                        "Could you please be more specific?"
                    )
                    dialog_state["context_resolved"] = False
                    
        logger.info(f"[DEBUG] Final dialog_state needs_clarification: {dialog_state['needs_clarification']}")
        return dialog_state
        
    def _generate_clarification_question(self, message: str, intent_results, context):
        """
        Generate appropriate clarification questions based on ambiguous input.
        
        Args:
            message: User message
            intent_results: Intent recognition results
            context: Conversation context
            
        Returns:
            Clarification question string
        """
        # If multiple intents with similar confidence
        if isinstance(intent_results, dict) and len(intent_results) > 1:
            top_intents = sorted(intent_results.items(), key=lambda x: x[1], reverse=True)[:3]
            # Only trigger clarification if top confidence is low AND intents are very close
            if (len(top_intents) >= 2 and 
                top_intents[0][1] < 0.4 and  # Top confidence must be low
                abs(top_intents[0][1] - top_intents[1][1]) < 0.1):  # And very close to second
                intent_descriptions = {
                    "fabric_health": "check fabric health status",
                    "device_health": "check device health and status", 
                    "fabric_devices": "show fabric devices",
                    "network_topology": "view network topology",
                    "interface_status": "check interface status",
                    "network_anomalies": "investigate network anomalies",
                    "device_details": "get detailed device information"
                }
                
                options = []
                for intent, _ in top_intents[:2]:
                    if intent in intent_descriptions:
                        options.append(intent_descriptions[intent])
                        
                if options:
                    return f"I can help you {' or '.join(options)}. Which would you prefer?"
                    
        # If no clear intent
        if not intent_results or max(intent_results.values()) < 0.3:
            # Analyze message for hints
            message_lower = message.lower()
            if any(word in message_lower for word in ['health', 'status', 'check']):
                return "Would you like to check the health of devices, fabrics, or interfaces?"
            elif any(word in message_lower for word in ['show', 'list', 'get']):
                return "What would you like me to show you? For example: device list, fabric topology, or anomalies?"
            elif any(word in message_lower for word in ['problem', 'issue', 'error']):
                return "Are you looking for network anomalies, device issues, or interface problems?"
            else:
                return "I can help you with network monitoring tasks. What would you like to know about your network?"
                
        # Generic fallback
        return "Could you please provide more details about what you'd like to know?"
        
    def _handle_follow_up(self, intent, data, context):
        """
        Generate contextual follow-up suggestions based on current interaction.
        
        Args:
            intent: Current intent
            data: Data retrieved for the intent
            context: Conversation context
            
        Returns:
            List of follow-up suggestions
        """
        follow_ups = []
        
        if intent == "fabric_health" and data:
            items = data.get('items', [])
            unhealthy_fabrics = [item for item in items 
                               if item.get('healthScore', 100) < 80]
            if unhealthy_fabrics:
                follow_ups.append("Would you like me to investigate the unhealthy fabrics in detail?")
                follow_ups.append("Should I check for specific anomalies in these fabrics?")
                
        elif intent == "device_health" and data:
            items = data.get('items', [])
            critical_devices = [item for item in items 
                              if item.get('healthScore', 100) < 60]
            if critical_devices:
                follow_ups.append("Would you like troubleshooting steps for the critical devices?")
                follow_ups.append("Should I check the interface status for these devices?")
                
        elif intent == "network_anomalies" and data:
            items = data.get('items', [])
            if items:
                follow_ups.append("Would you like detailed explanations for these anomalies?")
                follow_ups.append("Should I provide troubleshooting recommendations?")
                
        elif intent == "interface_status" and data:
            items = data.get('items', [])
            down_interfaces = [item for item in items 
                             if item.get('status', '').lower() == 'down']
            if down_interfaces:
                follow_ups.append("Would you like me to check what's causing the interface issues?")
                follow_ups.append("Should I look for related network anomalies?")
                
        # Add context-aware follow-ups
        if context and hasattr(context, 'recent_intents'):
            recent_intents = [intent_obj["name"] for intent_obj in context.recent_intents[-3:]]
            if "device_health" in recent_intents and intent != "interface_status":
                follow_ups.append("Would you also like to check the interface status for these devices?")
            elif "fabric_health" in recent_intents and intent != "network_anomalies":
                follow_ups.append("Should I check for any anomalies in this fabric?")
                
        return follow_ups[:2]  # Limit to 2 follow-ups for clarity

    def _resolve_references(self, message: str) -> str:
        """
        Resolve references in the user message (e.g., "it", "that device", etc.)
        
        Args:
            message: User message with potential references
            
        Returns:
            Resolved message with replaced references
        """
        # Import required modules at the beginning of the method
        import re
        
        # Check for pronouns that might be references
        pronouns = ['it', 'this', 'that', 'them', 'they', 'these', 'those']
        for pronoun in pronouns:
            if pronoun in message.lower():
                # Try to find the referenced entity in the conversation context
                referenced_entity = self.context.get_referenced_entity(pronoun)
                if referenced_entity:
                    # Replace the pronoun with the entity name
                    message = re.sub(r'\b' + pronoun + r'\b', referenced_entity, message, flags=re.IGNORECASE)
        
        return message

    def _extract_entities(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Extract entities from the user message or response data
        
        Args:
            message: User message or response text
            data: Optional response data for entity extraction
            
        Returns:
            None
        """
        # Import required modules at the beginning of the method
        import re
        
        # Extract entities from the message
        entities = []
        for entity_type in ['device', 'fabric', 'interface', 'anomaly']:
            pattern = rf'\b({entity_type}s?)(?:\s*\d+)?\b'
            matches = re.findall(pattern, message, flags=re.IGNORECASE)
            if matches:
                entities.extend(matches)
        
        # Extract entities from the response data if provided
        if data:
            for entity_type in ['device', 'fabric', 'interface', 'anomaly']:
                if entity_type in data:
                    entities.extend(data[entity_type])
        
        # Update the conversation context with the extracted entities
        for entity in entities:
            # Ensure entities attribute exists on context
            if not hasattr(self.context, 'entities') or self.context.entities is None:
                self.context.entities = {}
            
            # Ensure 'general' key exists in entities dictionary
            if 'general' not in self.context.entities:
                self.context.entities['general'] = []
                
            # Add the entity
            self.context.entities['general'].append({
                'value': entity,
                'source': 'message_extraction'
            })

    def _get_network_anomalies_enhanced(self, message: str) -> Optional[dict]:
        """
        Enhanced: Get network anomalies from NDI using verified endpoint.
        Parses actual anomaly data to extract types, affected devices, and severity levels.
        """
        from agents.verified_endpoints import get_ndi_endpoint
        
        try:
            logger.info(f"[ANOMALY ENTRY] _get_network_anomalies_enhanced called with message: '{message}'")
            logger.info(f"[ANOMALY DEBUG] Using client: {self.nd_client}")
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }

            # Get anomalies endpoint - try multiple working endpoints
            anomalies_endpoint = get_ndi_endpoint("anomalies", "get_anomalies_summary")
            if not anomalies_endpoint:
                anomalies_endpoint = get_ndi_endpoint("anomalies", "get_anomalies_details")
            if not anomalies_endpoint:
                anomalies_endpoint = get_ndi_endpoint("anomalies", "get_anomalies_count")
            logger.info(f"[ANOMALY ENDPOINT] Using endpoint: {anomalies_endpoint}")
            logger.info(f"[ANOMALY CLIENT] ND client initialized: {self.nd_client is not None}")
            if not anomalies_endpoint:
                logger.error("NDI anomalies endpoint not found")
                return {
                    "error": "Endpoint not found", 
                    "message": "Unable to retrieve network anomalies information."
                }

            logger.info(f"Attempting to get network anomalies from NDI: {anomalies_endpoint}")
            logger.info(f"[ANOMALY ENDPOINT] Using endpoint: {anomalies_endpoint}")
            
            # Determine if we need anomaly summary or details based on the query
            needs_details = any(term in message.lower() for term in ["detail", "specific", "explain", "what are"])
            
            # Build dynamic parameters to match what the UI would use
            # These are the same parameters extracted from the UI network requests
            query_params = {}
            
            # For grouped view (default in UI)
            if "ungrouped" not in message.lower():
                query_params["grouped"] = "true"
            
            # Default filters used by the UI
            query_params["filter"] = "cleared:false AND acknowledged:false"
            query_params["includeAnomalies"] = "rootAndUncorrelated"
            
            # Use default site group unless specified
            if "all sites" not in message.lower():
                query_params["siteGroupName"] = "default"
                
            # Add time range if the endpoint requires it
            if "details" in anomalies_endpoint:
                from datetime import datetime, timedelta
                end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
                start_time = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S%z")
                query_params["endDate"] = end_time
                query_params["startDate"] = start_time
                
                # If details are needed, add pagination parameters
                if needs_details:
                    query_params["offset"] = "0"
                    query_params["count"] = "50"  # Get more details if specifically asked
            logger.info(f"[ANOMALY PARAMS] Using query parameters: {query_params}")
            
            # Make API call with parameters
            response = self._api_call_with_retry(anomalies_endpoint, params=query_params)
            logger.info(f"[ANOMALY RAW RESPONSE] {response}")
            
            if not response:
                return {
                    "error": "Failed to retrieve anomalies",
                    "message": "Unable to retrieve network anomalies information."
                }

            anomalies_data = response
            logger.info(f"Successfully retrieved network anomalies. Sample: {str(anomalies_data)[:200]}...")

            # Parse actual anomaly data from API response
            anomaly_count = 0
            anomaly_types = []
            affected_devices = []
            severity_levels = []
            
            if isinstance(anomalies_data, dict):
                # Handle both summary and details response structures
                
                # Check for totalAnomalyCount in summary API response
                if 'totalAnomalyCount' in anomalies_data:
                    anomaly_count = anomalies_data.get('totalAnomalyCount', 0)
                    entries = anomalies_data.get('entries', [])
                    for entry in entries:
                        if isinstance(entry, dict):
                            severity = entry.get('severity')
                            if severity and severity not in severity_levels:
                                severity_levels.append(severity)
                
                # Check for entries in details API response
                elif 'entries' in anomalies_data and isinstance(anomalies_data['entries'], list):
                    entries = anomalies_data.get('entries', [])
                    anomaly_count = len(entries)
                    
                    for anomaly in entries:
                        if isinstance(anomaly, dict):
                            # Extract actual anomaly details
                            anom_type = (anomaly.get('category') or 
                                      anomaly.get('subCategory') or 
                                      anomaly.get('mnemonicTitle') or 
                                      anomaly.get('type') or 
                                      'Unknown')
                            
                            device = (anomaly.get('entityName') or 
                                    anomaly.get('fabricName') or
                                    anomaly.get('deviceName') or 
                                    anomaly.get('node') or 
                                    anomaly.get('source'))
                            
                            severity = anomaly.get('severity') or 'Medium'
                            
                            if anom_type not in anomaly_types:
                                anomaly_types.append(anom_type)
                            if device and device not in affected_devices:
                                affected_devices.append(device)
                            if severity not in severity_levels:
                                severity_levels.append(severity)
                
                # Try legacy formats as fallback
                else:
                    anomalies = (anomalies_data.get('anomalies') or 
                               anomalies_data.get('data') or 
                               anomalies_data.get('results') or [])
                    
                    if isinstance(anomalies, list):
                        anomaly_count = len(anomalies)
                        for anomaly in anomalies:
                            if isinstance(anomaly, dict):
                                # Extract actual anomaly details
                                anom_type = anomaly.get('type') or anomaly.get('category') or 'Unknown'
                                device = anomaly.get('deviceName') or anomaly.get('node') or anomaly.get('source')
                                severity = anomaly.get('severity') or anomaly.get('level') or 'Medium'
                                
                                if anom_type not in anomaly_types:
                                    anomaly_types.append(anom_type)
                                if device and device not in affected_devices:
                                    affected_devices.append(device)
                                if severity not in severity_levels:
                                    severity_levels.append(severity)

            # Debug log before summary construction
            logger.info(f"[ANOMALY DEBUG] anomaly_count: {anomaly_count}, anomaly_types: {anomaly_types}, affected_devices: {affected_devices}, severity_levels: {severity_levels}, anomalies_data keys: {list(anomalies_data.keys())}")
            logger.info(f"[ANOMALY COUNT] Parsed anomaly_count: {anomaly_count}")
            # Build summary from actual data
            if anomaly_count > 0:
                summary = f"I found {anomaly_count} network anomal{'y' if anomaly_count == 1 else 'ies'} in your environment. "
                if anomaly_types:
                    summary += f"Types: {', '.join(anomaly_types[:3])}{'...' if len(anomaly_types) > 3 else ''}. "
                if affected_devices:
                    summary += f"Affected devices: {', '.join(affected_devices[:3])}{'...' if len(affected_devices) > 3 else ''}. "
                if severity_levels:
                    summary += f"Severity levels: {', '.join(severity_levels)}."
            else:
                summary = "Great news! I didn't find any active network anomalies in your environment."

            return {
                "anomalies": anomalies_data,
                "total": anomaly_count,
                "types": anomaly_types,
                "affected_devices": affected_devices,
                "severity_levels": severity_levels,
                "message": summary
            }

        except Exception as e:
            logger.error(f"Error getting network anomalies: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving network anomalies information."
            }

    def _get_advisories(self, message: str) -> Optional[dict]:
        """
        Enhanced: Get advisories from NDI using verified endpoint.
        Parses actual advisory data to extract types, affected products, and severity levels.
        """
        from agents.verified_endpoints import get_ndi_endpoint
        
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }

            advisories_endpoint = get_ndi_endpoint("advisories", "summary")
            if not advisories_endpoint:
                logger.error("NDI advisories endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve advisories information."
                }

            logger.info(f"Attempting to get advisories from NDI: {advisories_endpoint}")
            response = self._api_call_with_retry(advisories_endpoint)
            
            if not response:
                return {
                    "error": "Failed to retrieve advisories",
                    "message": "Unable to retrieve advisories information."
                }

            advisories_data = response
            logger.info(f"Successfully retrieved advisories. Sample: {str(advisories_data)[:200]}...")

            # Parse actual advisory data from API response
            advisory_count = 0
            advisory_types = []
            affected_products = []
            severity_levels = []
            
            if isinstance(advisories_data, dict):
                advisories = (advisories_data.get("advisories") or 
                            advisories_data.get("data") or 
                            advisories_data.get("results") or [])
                
                if isinstance(advisories, list):
                    advisory_count = len(advisories)
                    for advisory in advisories:
                        if isinstance(advisory, dict):
                            # Extract actual advisory details
                            adv_type = advisory.get("type") or advisory.get("category") or "Security"
                            product = advisory.get("product") or advisory.get("affectedProduct") or advisory.get("platform")
                            severity = advisory.get("severity") or advisory.get("impact") or "Medium"
                            
                            if adv_type not in advisory_types:
                                advisory_types.append(adv_type)
                            if product and product not in affected_products:
                                affected_products.append(product)
                            if severity not in severity_levels:
                                severity_levels.append(severity)

            # Build summary from actual data
            if advisory_count > 0:
                summary = f"I found {advisory_count} advisor{'y' if advisory_count == 1 else 'ies'} for your environment. "
                if advisory_types:
                    summary += f"Types: {', '.join(advisory_types[:3])}{'...' if len(advisory_types) > 3 else ''}. "
                if affected_products:
                    summary += f"Affected products: {', '.join(affected_products[:3])}{'...' if len(affected_products) > 3 else ''}. "
                if severity_levels:
                    summary += f"Severity levels: {', '.join(severity_levels)}."
            else:
                summary = "Good news! I didn't find any active advisories affecting your environment."

            return {
                "advisories": advisories_data,
                "total": advisory_count,
                "types": advisory_types,
                "affected_products": affected_products,
                "severity_levels": severity_levels,
                "message": summary
            }

        except Exception as e:
            logger.error(f"Error getting advisories: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving advisories information."
            }

    def _get_network_interfaces(self, message: str) -> Optional[dict]:
        """
        Enhanced: Get network interfaces from verified endpoint.
        Parses actual interface data to extract names, states, types, and utilization.
        """
        from agents.verified_endpoints import get_endpoint
        
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }

            interfaces_endpoint = get_endpoint("interfaces")
            if not interfaces_endpoint:
                logger.error("Interfaces endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve network interfaces information."
                }

            logger.info(f"Attempting to get network interfaces: {interfaces_endpoint}")
            response = self._api_call_with_retry(interfaces_endpoint)
            
            if not response:
                return {
                    "error": "Failed to retrieve interfaces",
                    "message": "Unable to retrieve network interfaces information."
                }

            interfaces_data = response
            logger.info(f"Successfully retrieved network interfaces. Sample: {str(interfaces_data)[:200]}...")

            # Parse actual interface data from API response
            interface_count = 0
            interface_names = []
            interface_states = []
            interface_types = []
            up_interfaces = 0
            down_interfaces = 0
            
            if isinstance(interfaces_data, dict):
                interfaces = (interfaces_data.get("interfaces") or 
                            interfaces_data.get("data") or 
                            interfaces_data.get("results") or [])
                
                if isinstance(interfaces, list):
                    interface_count = len(interfaces)
                    for interface in interfaces:
                        if isinstance(interface, dict):
                            # Extract actual interface details
                            name = interface.get("name") or interface.get("interfaceName") or interface.get("id")
                            state = interface.get("state") or interface.get("status") or interface.get("operationalState")
                            iface_type = interface.get("type") or interface.get("interfaceType") or "Ethernet"
                            
                            if name and name not in interface_names:
                                interface_names.append(name)
                            if state:
                                if state.lower() in ['up', 'active', 'online']:
                                    up_interfaces += 1
                                elif state.lower() in ['down', 'inactive', 'offline']:
                                    down_interfaces += 1
                                if state not in interface_states:
                                    interface_states.append(state)
                            if iface_type not in interface_types:
                                interface_types.append(iface_type)

            # Build summary from actual data
            if interface_count > 0:
                summary = f"I found {interface_count} network interface{'s' if interface_count != 1 else ''} in your environment. "
                if up_interfaces > 0 or down_interfaces > 0:
                    summary += f"Status: {up_interfaces} up, {down_interfaces} down. "
                if interface_types:
                    summary += f"Types: {', '.join(interface_types[:3])}{'...' if len(interface_types) > 3 else ''}. "
                if interface_names:
                    summary += f"Sample interfaces: {', '.join(interface_names[:3])}{'...' if len(interface_names) > 3 else ''}."
            else:
                summary = "I didn't find any network interfaces in your environment."

            return {
                "interfaces": interfaces_data,
                "total": interface_count,
                "names": interface_names,
                "states": interface_states,
                "types": interface_types,
                "up_count": up_interfaces,
                "down_count": down_interfaces,
                "message": summary
            }

        except Exception as e:
            logger.error(f"Error getting network interfaces: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving network interfaces information."
            }

    def _get_network_endpoints(self, message: str) -> Optional[dict]:
        """
        Enhanced: Get network endpoints from verified endpoint.
        Parses actual endpoint data to extract counts, types, IP addresses, and connection details.
        """
        from agents.verified_endpoints import get_ndi_endpoint
        
        try:
            # 1. Client validation
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }

            # 2. Get verified endpoint path
            endpoints_endpoint = get_ndi_endpoint("endpoints", "get_all_endpoints")
            if not endpoints_endpoint:
                logger.error("Network endpoints endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve network endpoints information."
                }

            # 3. Make API call with retry logic
            logger.info(f"Attempting to get network endpoints: {endpoints_endpoint}")
            response = self._api_call_with_retry(endpoints_endpoint)
            
            if not response:
                return {
                    "error": "Failed to retrieve endpoints",
                    "message": "Unable to retrieve network endpoints information."
                }

            # 4. Parse response data
            endpoints_data = response
            logger.info(f"Successfully retrieved network endpoints. Sample: {str(endpoints_data)[:200]}...")

            # 5. Extract specific data points
            endpoint_count = 0
            endpoint_types = []
            ip_addresses = []
            mac_addresses = []
            vlans = []
            
            if isinstance(endpoints_data, dict):
                endpoints = (endpoints_data.get("endpoints") or 
                           endpoints_data.get("data") or 
                           endpoints_data.get("items") or [])
                
                if isinstance(endpoints, list):
                    endpoint_count = len(endpoints)
                    for endpoint in endpoints:
                        if isinstance(endpoint, dict):
                            # Extract actual endpoint details
                            ep_type = endpoint.get("type") or endpoint.get("endpointType") or "Unknown"
                            ip = endpoint.get("ip") or endpoint.get("ipAddress")
                            mac = endpoint.get("mac") or endpoint.get("macAddress")
                            vlan = endpoint.get("vlanId") or endpoint.get("vlan")
                            
                            if ep_type and ep_type not in endpoint_types:
                                endpoint_types.append(ep_type)
                            if ip and ip not in ip_addresses:
                                ip_addresses.append(ip)
                            if mac and mac not in mac_addresses:
                                mac_addresses.append(mac)
                            if vlan and vlan not in vlans and vlan != 0:
                                vlans.append(vlan)

            # 6. Build factual summary
            if endpoint_count > 0:
                summary = f"I found {endpoint_count} network endpoint{'s' if endpoint_count != 1 else ''} in your environment. "
                if endpoint_types:
                    summary += f"Types: {', '.join(endpoint_types[:3])}{'...' if len(endpoint_types) > 3 else ''}. "
                if vlans:
                    summary += f"They span {len(vlans)} VLAN{'s' if len(vlans) > 1 else ''}: {', '.join(str(v) for v in vlans[:3])}{'...' if len(vlans) > 3 else ''}. "
                if ip_addresses:
                    summary += f"Sample IPs: {', '.join(ip_addresses[:3])}{'...' if len(ip_addresses) > 3 else ''}."
            else:
                summary = "I didn't find any network endpoints in your environment."

            # 7. Return structured data and summary
            return {
                "endpoints": endpoints_data,
                "total": endpoint_count,
                "types": endpoint_types,
                "ip_addresses": ip_addresses[:10],  # Limit to avoid very large responses
                "mac_addresses": mac_addresses[:10],
                "vlans": vlans,
                "message": summary
            }

        except Exception as e:
            # 8. Exception handling
            logger.error(f"Error getting network endpoints: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving network endpoints information."
            }
            
    def _get_system_information(self, message: str) -> Optional[dict]:
        """
        Enhanced: Get system information from verified endpoint.
        Parses actual system data to extract version, cluster info, and node status.
        """
        from agents.verified_endpoints import get_endpoint
        
        try:
            if not self.nd_client:
                logger.error("No ND client initialized")
                return {
                    "error": "No connection to Nexus Dashboard",
                    "message": "Please check your connection to Nexus Dashboard."
                }

            system_endpoint = get_endpoint("system_info")
            if not system_endpoint:
                logger.error("System info endpoint not found")
                return {
                    "error": "Endpoint not found",
                    "message": "Unable to retrieve system information."
                }

            logger.info(f"Attempting to get system information: {system_endpoint}")
            response = self._api_call_with_retry(system_endpoint)
            
            if not response:
                return {
                    "error": "Failed to retrieve system info",
                    "message": "Unable to retrieve system information."
                }

            system_data = response
            logger.info(f"Successfully retrieved system information. Sample: {str(system_data)[:200]}...")

            # Parse actual system data from API response
            version = "Unknown"
            cluster_name = "Unknown"
            node_count = 0
            node_states = []
            services = []
            
            if isinstance(system_data, dict):
                # Extract actual system details
                version = (system_data.get("version") or 
                          system_data.get("softwareVersion") or 
                          system_data.get("release") or "Unknown")
                
                cluster_name = (system_data.get("clusterName") or 
                              system_data.get("cluster") or 
                              system_data.get("name") or "Unknown")
                
                nodes = (system_data.get("nodes") or 
                        system_data.get("clusterNodes") or [])
                
                if isinstance(nodes, list):
                    node_count = len(nodes)
                    for node in nodes:
                        if isinstance(node, dict):
                            state = node.get("state") or node.get("status") or node.get("health")
                            if state and state not in node_states:
                                node_states.append(state)
                
                service_list = (system_data.get("services") or 
                              system_data.get("applications") or [])
                
                if isinstance(service_list, list):
                    for service in service_list:
                        if isinstance(service, dict):
                            svc_name = service.get("name") or service.get("serviceName")
                            if svc_name and svc_name not in services:
                                services.append(svc_name)

            # Build summary from actual data
            summary = f"Nexus Dashboard system information: Version {version}, Cluster '{cluster_name}'. "
            if node_count > 0:
                summary += f"Nodes: {node_count} total"
                if node_states:
                    summary += f" ({', '.join(node_states)})"
                summary += ". "
            if services:
                summary += f"Services: {', '.join(services[:3])}{'...' if len(services) > 3 else ''}."

            return {
                "system": system_data,
                "version": version,
                "cluster_name": cluster_name,
                "node_count": node_count,
                "node_states": node_states,
                "services": services,
                "message": summary
            }

        except Exception as e:
            logger.error(f"Error getting system information: {str(e)}")
            return {
                "error": str(e),
                "message": "An error occurred while retrieving system information."
            }
    def _get_network_interfaces(self, message: str) -> Optional[dict]:
        """
        Retrieves network interfaces information from Nexus Dashboard Insights
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with interfaces information or error message
        """
        try:
            logger.info(f"[NETWORK INTERFACES] Processing request: {message}")
            
            # Get the verified NDI endpoint for interfaces
            from agents.verified_endpoints import get_ndi_endpoint
            endpoint_url = get_ndi_endpoint("interfaces", "get_all_interfaces")
            
            if not endpoint_url:
                logger.error("[NETWORK INTERFACES] No verified endpoint found for interfaces")
                return {"error": "Interfaces endpoint not available", "message": "I couldn't access the network interfaces information right now."}
            
            logger.info(f"[NETWORK INTERFACES] Using endpoint: {endpoint_url}")
            
            # Make API call with retry logic
            response = self._api_call_with_retry(endpoint_url)
            if not response:
                return {"error": "API call failed", "message": "I couldn't retrieve network interfaces data at the moment."}
            
            # Parse interfaces data with flexible structure handling
            interfaces_data = response.get("interfaces") or response.get("data") or response.get("results") or response or {}
            
            # Handle different possible response structures
            interfaces_list = []
            if isinstance(interfaces_data, list):
                interfaces_list = interfaces_data
            elif isinstance(interfaces_data, dict):
                interfaces_list = (interfaces_data.get("interfaces") or 
                                 interfaces_data.get("data") or 
                                 interfaces_data.get("results") or 
                                 interfaces_data.get("entries") or [])
            
            # Parse interface details
            interface_count = len(interfaces_list)
            interface_names = []
            interface_types = []
            interface_states = []
            error_counts = []
            
            for interface in interfaces_list[:50]:  # Limit to first 50 for summary
                name = interface.get("name") or interface.get("interfaceName") or interface.get("id") or "Unknown"
                iface_type = interface.get("type") or interface.get("interfaceType") or interface.get("kind") or "Unknown"
                state = interface.get("state") or interface.get("status") or interface.get("operState") or "Unknown"
                errors = interface.get("errorCount") or interface.get("errors") or interface.get("faults") or 0
                
                interface_names.append(name)
                interface_types.append(iface_type)
                interface_states.append(state)
                error_counts.append(errors)
            
            # Build conversational summary
            if interface_count == 0:
                summary = "I didn't find any network interfaces in your environment."
            else:
                unique_types = list(set(interface_types))
                unique_states = list(set(interface_states))
                total_errors = sum([e for e in error_counts if isinstance(e, (int, float))])
                
                summary = f"I found {interface_count} network interfaces in your environment. "
                if unique_types:
                    summary += f"Interface types include: {', '.join(unique_types[:5])}. "
                if unique_states:
                    summary += f"Current states: {', '.join(unique_states)}. "
                if total_errors > 0:
                    summary += f"Total error count across interfaces: {total_errors}. "
                
                if interface_count > 10:
                    summary += f"Here are the first 10 interfaces: {', '.join(interface_names[:10])}."
                else:
                    summary += f"Interface names: {', '.join(interface_names)}."
            
            logger.info(f"[NETWORK INTERFACES] Successfully processed {interface_count} interfaces")
            
            return {
                "interfaces": interfaces_data,
                "total": interface_count,
                "names": interface_names,
                "types": interface_types,
                "states": interface_states,
                "error_counts": error_counts,
                "message": summary
            }
            
        except Exception as e:
            logger.error(f"[NETWORK INTERFACES] Error retrieving interfaces: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "message": "I encountered an error while retrieving network interfaces information."}

    def _get_network_routes(self, message: str) -> Optional[dict]:
        """
        Retrieves network routes information from Nexus Dashboard Insights
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with routes information or error message
        """
        try:
            logger.info(f"[NETWORK ROUTES] Processing request: {message}")
            
            # Get the verified NDI endpoint for routes
            from agents.verified_endpoints import get_ndi_endpoint
            endpoint_url = get_ndi_endpoint("routes", "get_all_routes")
            
            if not endpoint_url:
                logger.error("[NETWORK ROUTES] No verified endpoint found for routes")
                return {"error": "Routes endpoint not available", "message": "I couldn't access the network routes information right now."}
            
            logger.info(f"[NETWORK ROUTES] Using endpoint: {endpoint_url}")
            
            # Make API call with retry logic
            response = self._api_call_with_retry(endpoint_url)
            if not response:
                return {"error": "API call failed", "message": "I couldn't retrieve network routes data at the moment."}
            
            # Parse routes data with flexible structure handling
            routes_data = response.get("routes") or response.get("data") or response.get("results") or response or {}
            
            # Handle different possible response structures
            routes_list = []
            if isinstance(routes_data, list):
                routes_list = routes_data
            elif isinstance(routes_data, dict):
                routes_list = (routes_data.get("routes") or 
                              routes_data.get("data") or 
                              routes_data.get("results") or 
                              routes_data.get("entries") or [])
            
            # Parse route details
            route_count = len(routes_list)
            destinations = []
            next_hops = []
            protocols = []
            metrics = []
            
            for route in routes_list[:50]:  # Limit to first 50 for summary
                dest = route.get("destination") or route.get("prefix") or route.get("network") or "Unknown"
                next_hop = route.get("nextHop") or route.get("gateway") or route.get("via") or "Unknown"
                protocol = route.get("protocol") or route.get("type") or route.get("source") or "Unknown"
                metric = route.get("metric") or route.get("cost") or route.get("distance") or 0
                
                destinations.append(dest)
                next_hops.append(next_hop)
                protocols.append(protocol)
                metrics.append(metric)
            
            # Build conversational summary
            if route_count == 0:
                summary = "I didn't find any network routes in your environment."
            else:
                unique_protocols = list(set(protocols))
                unique_next_hops = list(set(next_hops))
                
                summary = f"I found {route_count} network routes in your environment. "
                if unique_protocols:
                    summary += f"Routing protocols include: {', '.join(unique_protocols[:5])}. "
                if unique_next_hops and len(unique_next_hops) <= 10:
                    summary += f"Next hops: {', '.join(unique_next_hops)}. "
                
                if route_count > 10:
                    summary += f"Here are some example destinations: {', '.join(destinations[:5])}."
                else:
                    summary += f"Route destinations: {', '.join(destinations)}."
            
            logger.info(f"[NETWORK ROUTES] Successfully processed {route_count} routes")
            
            return {
                "routes": routes_data,
                "total": route_count,
                "destinations": destinations,
                "next_hops": next_hops,
                "protocols": protocols,
                "metrics": metrics,
                "message": summary
            }
            
        except Exception as e:
            logger.error(f"[NETWORK ROUTES] Error retrieving routes: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "message": "I encountered an error while retrieving network routes information."}

    def _get_network_events(self, message: str) -> Optional[dict]:
        """
        Retrieves network events information from Nexus Dashboard Insights
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with events information or error message
        """
        try:
            logger.info(f"[NETWORK EVENTS] Processing request: {message}")
            
            # Get the verified NDI endpoint for events
            from agents.verified_endpoints import get_ndi_endpoint
            endpoint_url = get_ndi_endpoint("events", "get_events_summary")
            
            if not endpoint_url:
                logger.error("[NETWORK EVENTS] No verified endpoint found for events")
                return {"error": "Events endpoint not available", "message": "I couldn't access the network events information right now."}
            
            logger.info(f"[NETWORK EVENTS] Using endpoint: {endpoint_url}")
            
            # Make API call with retry logic
            response = self._api_call_with_retry(endpoint_url)
            if not response:
                return {"error": "API call failed", "message": "I couldn't retrieve network events data at the moment."}
            
            # Parse events data with flexible structure handling
            events_data = response.get("events") or response.get("data") or response.get("results") or response or {}
            
            # Handle different possible response structures
            events_list = []
            if isinstance(events_data, list):
                events_list = events_data
            elif isinstance(events_data, dict):
                events_list = (events_data.get("events") or 
                              events_data.get("data") or 
                              events_data.get("results") or 
                              events_data.get("entries") or 
                              events_data.get("buckets") or [])
            
            # Parse event details
            event_count = len(events_list)
            event_types = []
            severities = []
            sources = []
            timestamps = []
            
            for event in events_list[:50]:  # Limit to first 50 for summary
                event_type = event.get("type") or event.get("eventType") or event.get("category") or "Unknown"
                severity = event.get("severity") or event.get("level") or event.get("priority") or "Medium"
                source = event.get("source") or event.get("node") or event.get("device") or "Unknown"
                timestamp = event.get("timestamp") or event.get("time") or event.get("date") or "Unknown"
                
                event_types.append(event_type)
                severities.append(severity)
                sources.append(source)
                timestamps.append(timestamp)
            
            # Build conversational summary
            if event_count == 0:
                summary = "I didn't find any network events in your environment."
            else:
                unique_types = list(set(event_types))
                unique_severities = list(set(severities))
                unique_sources = list(set(sources))
                
                summary = f"I found {event_count} network events in your environment. "
                if unique_types:
                    summary += f"Event types include: {', '.join(unique_types[:5])}. "
                if unique_severities:
                    summary += f"Severity levels: {', '.join(unique_severities)}. "
                if unique_sources and len(unique_sources) <= 10:
                    summary += f"Event sources: {', '.join(unique_sources)}. "
                
                # Count recent events if timestamps are available
                recent_count = sum(1 for ts in timestamps if ts != "Unknown")
                if recent_count > 0:
                    summary += f"{recent_count} events have timestamp information."
            
            logger.info(f"[NETWORK EVENTS] Successfully processed {event_count} events")
            
            return {
                "events": events_data,
                "total": event_count,
                "types": event_types,
                "severities": severities,
                "sources": sources,
                "timestamps": timestamps,
                "message": summary
            }
            
        except Exception as e:
            logger.error(f"[NETWORK EVENTS] Error retrieving events: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "message": "I encountered an error while retrieving network events information."}

    # Prevent character-by-character breakdown in LLM responses
    def _clean_llm_response(self, response_text):
        """Clean up LLM response to prevent character breakdown"""
        if isinstance(response_text, str):
            # Remove character-by-character breakdown patterns
            response_text = re.sub(r'([a-zA-Z]),\s+([a-zA-Z])', r'\1\2', response_text)
            # Fix common breakdown patterns
            response_text = re.sub(r'\b([a-zA-Z])\s*,\s*([a-zA-Z])\s*,\s*([a-zA-Z])\b', r'\1\2\3', response_text)
        return response_text
    

    def _get_flow_rules(self, message: str) -> Optional[dict]:
        """
        Retrieves flow rules information from Nexus Dashboard Insights
        
        Args:
            message: Original user query text
            
        Returns:
            Dictionary with flow rules information or error message
        """
        try:
            logger.info(f"[FLOW RULES] Processing request: {message}")
            
            # Get the verified NDI endpoint for flow rules
            from agents.verified_endpoints import get_ndi_endpoint
            endpoint_url = get_ndi_endpoint("flows", "get_flow_rules")
            
            if not endpoint_url:
                logger.error("[FLOW RULES] No verified endpoint found for flow rules")
                return {"error": "Flow rules endpoint not available", "message": "I couldn't access the flow rules information right now."}
            
            logger.info(f"[FLOW RULES] Using endpoint: {endpoint_url}")
            
            # Make API call with retry logic
            response = self._api_call_with_retry(endpoint_url)
            if not response:
                return {"error": "API call failed", "message": "I couldn't retrieve flow rules data at the moment."}
            
            # Parse flow rules data with flexible structure handling
            flows_data = response.get("flows") or response.get("data") or response.get("results") or response or {}
            
            # Handle different possible response structures
            flows_list = []
            if isinstance(flows_data, list):
                flows_list = flows_data
            elif isinstance(flows_data, dict):
                flows_list = (flows_data.get("flows") or 
                             flows_data.get("data") or 
                             flows_data.get("results") or 
                             flows_data.get("entries") or 
                             flows_data.get("rules") or [])
            
            # Parse flow rule details
            flow_count = len(flows_list)
            sources = []
            destinations = []
            protocols = []
            actions = []
            
            for flow in flows_list[:50]:  # Limit to first 50 for summary
                source = flow.get("source") or flow.get("srcIp") or flow.get("sourceAddress") or "Unknown"
                dest = flow.get("destination") or flow.get("dstIp") or flow.get("destinationAddress") or "Unknown"
                protocol = flow.get("protocol") or flow.get("ipProtocol") or flow.get("type") or "Unknown"
                action = flow.get("action") or flow.get("disposition") or flow.get("verdict") or "Unknown"
                
                sources.append(source)
                destinations.append(dest)
                protocols.append(protocol)
                actions.append(action)
            
            # Build conversational summary
            if flow_count == 0:
                summary = "I didn't find any flow rules in your environment."
            else:
                unique_protocols = list(set(protocols))
                unique_actions = list(set(actions))
                unique_sources = list(set(sources))
                
                summary = f"I found {flow_count} flow rules in your environment. "
                if unique_protocols:
                    summary += f"Protocols include: {', '.join(unique_protocols[:5])}. "
                if unique_actions:
                    summary += f"Actions: {', '.join(unique_actions)}. "
                
                if flow_count > 10:
                    summary += f"Flow rules are configured for various source-destination pairs."
                else:
                    summary += f"Sources include: {', '.join(sources[:5])}."
            
            logger.info(f"[FLOW RULES] Successfully processed {flow_count} flow rules")
            
            return {
                "flows": flows_data,
                "total": flow_count,
                "sources": sources,
                "destinations": destinations,
                "protocols": protocols,
                "actions": actions,
                "message": summary
            }
            
        except Exception as e:
            logger.error(f"[FLOW RULES] Error retrieving flow rules: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "message": "I encountered an error while retrieving flow rules information."}
    
    def _get_auth_status(self, message: str) -> Optional[dict]:
        """Get authentication and security status"""
        try:
            endpoint = "/nexus/infra/api/aaa/v4/securitydomains"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved authentication status successfully",
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve auth status"}
        except Exception as e:
            return {"error": f"Auth status error: {str(e)}"}
    
    def _get_cluster_info(self, message: str) -> Optional[dict]:
        """Get cluster information"""
        try:
            endpoint = "/nexus/infra/api/platform/v2/clusters"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved cluster information successfully", 
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve cluster info"}
        except Exception as e:
            return {"error": f"Cluster info error: {str(e)}"}
    
    def _get_external_ips(self, message: str) -> Optional[dict]:
        """Get external IP information"""
        try:
            endpoint = "/nexus/infra/api/platform/v1/externalips"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved external IP information successfully",
                    "data": response, 
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve external IPs"}
        except Exception as e:
            return {"error": f"External IPs error: {str(e)}"}
    
    def _get_sites(self, message: str) -> Optional[dict]:
        """Get site information from Nexus Dashboard"""
        try:
            endpoint = "/nexus/api/sitemanagement/v4/sites"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response and isinstance(response, dict):
                sites = response.get('items', [])
                site_count = len(sites)
                
                summary = f"Found {site_count} sites in your Nexus Dashboard"
                if sites:
                    site_names = [site.get('spec', {}).get('name', 'Unknown') for site in sites[:3]]
                    summary += f". Sites include: {', '.join(site_names)}"
                    if site_count > 3:
                        summary += f" and {site_count - 3} more"
                
                return {
                    "summary": summary,
                    "data": response,
                    "endpoint_used": endpoint,
                    "site_count": site_count
                }
            return {"error": "Failed to retrieve sites information"}
        except Exception as e:
            return {"error": f"Sites error: {str(e)}"}
    
    def _get_site_groups(self, message: str) -> Optional[dict]:
        """Get site groups information from Nexus Dashboard"""
        try:
            endpoint = "/nexus/api/sitemanagement/v4/sitegroups"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response and isinstance(response, dict):
                groups = response.get('items', [])
                group_count = len(groups)
                
                summary = f"Found {group_count} site groups"
                if groups:
                    group_names = [group.get('spec', {}).get('name', 'Unknown') for group in groups[:3]]
                    summary += f": {', '.join(group_names)}"
                    if group_count > 3:
                        summary += f" and {group_count - 3} more"
                
                return {
                    "summary": summary,
                    "data": response,
                    "endpoint_used": endpoint,
                    "group_count": group_count
                }
            return {"error": "Failed to retrieve site groups"}
        except Exception as e:
            return {"error": f"Site groups error: {str(e)}"}
    
    def _get_event_config(self, message: str) -> Optional[dict]:
        """Get event monitoring configuration"""
        try:
            endpoint = "/nexus/infra/api/eventmonitoring/v1/eventconfigs"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved event monitoring configuration successfully",
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve event configuration"}
        except Exception as e:
            return {"error": f"Event config error: {str(e)}"}
    
    def _get_login_domains(self, message: str) -> Optional[dict]:
        """Get login domains information"""
        try:
            endpoint = "/nexus/infra/api/aaa/v4/logindomains"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response and isinstance(response, dict):
                domains = response.get('items', [])
                domain_count = len(domains)
                
                summary = f"Found {domain_count} login domains configured"
                if domains:
                    domain_names = [domain.get('spec', {}).get('domain', 'Unknown') for domain in domains]
                    summary += f": {', '.join(domain_names)}"
                
                return {
                    "summary": summary,
                    "data": response,
                    "endpoint_used": endpoint,
                    "domain_count": domain_count
                }
            return {"error": "Failed to retrieve login domains"}
        except Exception as e:
            return {"error": f"Login domains error: {str(e)}"}
    
    def _get_login_stats(self, message: str) -> Optional[dict]:
        """Get login statistics"""
        try:
            endpoint = "/login/stats"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved login statistics successfully",
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve login statistics"}
        except Exception as e:
            return {"error": f"Login stats error: {str(e)}"}
    
    def _get_fabric_config(self, message: str) -> Optional[dict]:
        """Get fabric configuration"""
        try:
            endpoint = "/appcenter/cisco/ndfc/ui/manage/lan-fabrics/config"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved fabric configuration successfully",
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve fabric configuration"}
        except Exception as e:
            return {"error": f"Fabric config error: {str(e)}"}
    
    def _get_fabric_metrics(self, message: str) -> Optional[dict]:
        """Get fabric metrics and statistics"""
        try:
            endpoint = "/appcenter/cisco/ndfc/ui/manage/lan-fabrics/metrics"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response:
                return {
                    "summary": "Retrieved fabric metrics successfully",
                    "data": response,
                    "endpoint_used": endpoint
                }
            return {"error": "Failed to retrieve fabric metrics"}
        except Exception as e:
            return {"error": f"Fabric metrics error: {str(e)}"}
    
    def _get_network_routes(self, message: str) -> Optional[dict]:
        """Get network routing information"""
        try:
            endpoint = "/nexus/infra/api/platform/v1/routes"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response and isinstance(response, dict):
                routes = response.get('items', [])
                route_count = len(routes)
                
                summary = f"Found {route_count} network routes configured"
                
                return {
                    "summary": summary,
                    "data": response,
                    "endpoint_used": endpoint,
                    "route_count": route_count
                }
            return {"error": "Failed to retrieve network routes"}
        except Exception as e:
            return {"error": f"Network routes error: {str(e)}"}
    
    def _get_system_info(self, message: str) -> Optional[dict]:
        """Get system information from platform clusters"""
        try:
            endpoint = "/nexus/infra/api/platform/v1/clusters"
            response = self._api_call_with_retry(endpoint, max_retries=3, cache_duration=300)
            
            if response and isinstance(response, dict):
                clusters = response.get('items', [])
                cluster_count = len(clusters)
                
                summary = f"System shows {cluster_count} clusters configured"
                if clusters:
                    cluster_names = [cluster.get('spec', {}).get('name', 'Unknown') for cluster in clusters[:3]]
                    summary += f": {', '.join(cluster_names)}"
                
                return {
                    "summary": summary,
                    "data": response,
                    "endpoint_used": endpoint,
                    "cluster_count": cluster_count
                }
            return {"error": "Failed to retrieve system information"}
        except Exception as e:
            return {"error": f"System info error: {str(e)}"}


# Create a singleton instance for use across the application
improved_chat = ImprovedChatArchitecture()

# Export the singleton instance
__all__ = ['improved_chat', 'ImprovedChatArchitecture']
