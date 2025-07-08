#!/usr/bin/env python3
"""
Base Handler for NDI Chatbot
This module provides a base class for all intent handlers.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import requests
from abc import ABC, abstractmethod
from cisco_api.nexus.dashboard_client import NexusDashboardClient
from utils.api_parsers import standardize_api_response

# Import parameter validation utilities
try:
    from agents.verified_endpoints import (
        get_endpoint_params_with_validation,
        validate_endpoint_params,
        get_endpoint_info,
        list_all_endpoints
    )
except ImportError:
    # Fallback if import fails
    logger.warning("Could not import parameter validation utilities from verified_endpoints")
    def get_endpoint_params_with_validation(endpoint_name: str, **provided_params) -> dict:
        return provided_params
    def validate_endpoint_params(endpoint_name: str, provided_params: dict) -> tuple[bool, list[str]]:
        return True, []
    def get_endpoint_info(endpoint_name: str) -> dict:
        return {"required": [], "optional": [], "defaults": {}}
    def list_all_endpoints() -> list[str]:
        return []

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseHandler(ABC):
    """
    Base class for all intent handlers
    
    This abstract class defines the interface that all handlers must implement
    and provides common functionality for API calls, response formatting, and
    error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the handler
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.cache = {}
        self.cache_duration = self.config.get('cache_duration', 300)  # Default 5 minutes
        self.debug_mode = self.config.get('debug_mode', False)
        
        # API configuration
        self.api_base_url = self.config.get('api_base_url', '')
        self.api_headers = self.config.get('api_headers', {})
        self.verify_ssl = self.config.get('verify_ssl', False)  # Default to False for development
        
        # Use the passed nd_client if available, otherwise create new instance
        self.client = self.config.get('nd_client')
        if self.client:
            logger.info(f"[{self.__class__.__name__}] Using passed authenticated ND client")
        elif self.api_base_url:
            logger.info(f"[{self.__class__.__name__}] Creating new ND client instance")
            self.client = NexusDashboardClient.get_instance(
                base_url=self.api_base_url,
                verify_ssl=self.verify_ssl
            )
            # Credentials should be provided in the config
            username = self.config.get('username')
            password = self.config.get('password')
            if username and password:
                logger.info(f"[{self.__class__.__name__}] Authenticating with Nexus Dashboard")
                auth_success = self.client.authenticate(username, password)
                if auth_success:
                    logger.info(f"[{self.__class__.__name__}] Authentication successful")
                else:
                    logger.error(f"[{self.__class__.__name__}] Authentication failed")
            else:
                logger.warning(f"[{self.__class__.__name__}] No credentials provided, client will not be authenticated")
        else:
            logger.warning(f"[{self.__class__.__name__}] No ND client or API base URL provided")
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def handle(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an intent with the given entities and context
        
        Args:
            entities: Dictionary of validated entities
            context: Context information
            
        Returns:
            Response dictionary with data and metadata
        """
        pass
    
    def _make_api_call(self, endpoint: str, method: str = 'GET', 
                       params: Optional[Dict[str, Any]] = None,
                       data: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Make an API call using the NexusDashboardClient singleton instance.
        
        This method handles token validation, CSRF token management, and error handling.
        
        Args:
            endpoint: API endpoint path (can be relative or absolute URL)
            method: HTTP method (GET, POST, PUT, DELETE)
            params: Query parameters for the request
            data: Request body data (for form data)
            headers: Additional headers to include in the request
            
        Returns:
            Tuple of (response_data, success) where:
            - response_data: Parsed JSON response or error details
            - success: Boolean indicating if the request was successful
        """
        # Log the API call for debugging and traceability
        logger.info(f"[{self.__class__.__name__}] API Call: {method} {endpoint}")
        if self.debug_mode:
            logger.debug(f"[{self.__class__.__name__}] Params: {params}")
            logger.debug(f"[{self.__class__.__name__}] Data: {data}")
        
        # Check if we have a client instance
        if not self.client:
            error_msg = f"[{self.__class__.__name__}] No NexusDashboardClient instance available for API call"
            logger.error(error_msg)
            return {"error": error_msg, "details": "Client not initialized"}, False
        
        try:
            # Ensure we have a valid token
            if not self.client.validate_token():
                logger.warning(f"[{self.__class__.__name__}] Token validation failed, attempting to re-authenticate")
                username = self.config.get('username')
                password = self.config.get('password')
                if username and password:
                    auth_success = self.client.authenticate(username, password)
                    if not auth_success:
                        error_msg = f"[{self.__class__.__name__}] Re-authentication failed"
                        logger.error(error_msg)
                        return {"error": error_msg}, False
                else:
                    error_msg = f"[{self.__class__.__name__}] No credentials available for re-authentication"
                    logger.error(error_msg)
                    return {"error": error_msg}, False
            
            # Prepare headers with CSRF token if available
            request_headers = {}
            if headers:
                request_headers.update(headers)
                
            # Add Content-Type if not specified and we have data
            if data and 'Content-Type' not in request_headers:
                request_headers['Content-Type'] = 'application/json'
            
            # Make the API call using the client's request method
            response = self.client._make_api_call(
                endpoint=endpoint,
                method=method,
                params=params,
                data=data,
                headers=request_headers
            )
            
            # Check for successful response
            if response.status_code >= 200 and response.status_code < 300:
                try:
                    # Try to parse JSON response
                    response_data = response.json()
                    logger.debug(f"[{self.__class__.__name__}] API Response: {response_data}")
                    return response_data, True
                except ValueError:
                    # If response is not JSON, return text
                    logger.debug(f"[{self.__class__.__name__}] Non-JSON response: {response.text}")
                    return {"response": response.text}, True
            else:
                # Handle error responses
                error_msg = f"API request failed with status {response.status_code}"
                logger.error(f"[{self.__class__.__name__}] {error_msg}: {response.text}")
                try:
                    error_details = response.json()
                    return {"error": error_msg, "details": error_details}, False
                except ValueError:
                    return {"error": error_msg, "response": response.text}, False
                    
        except Exception as e:
            # Handle any exceptions during the API call
            error_msg = f"Exception during API call to {endpoint}: {str(e)}"
            logger.error(f"[{self.__class__.__name__}] {error_msg}", exc_info=True)
            return {"error": error_msg, "exception": str(e)}, False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {str(e)}")
            return {"error": str(e)}, False

    def _format_response(self, data: Any, success: bool = None, message: Optional[str] = None, 
                        intent_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Format a standardized response with consistent structure across all handlers.
        
        This method ensures all responses follow a predictable format while preserving
        the working interface functionality. It wraps the response in a standardized
        structure that includes metadata, pagination, and error handling.

        Args:
            data: Raw API response data or result from handler
            success: Whether the operation was successful (auto-detected if None)
            message: Optional human-readable message
            intent_key: The intent this response corresponds to
            **kwargs: Additional metadata to include in the response
            
        Returns:
            Standardized response dictionary with the following structure:
            {
                "success": bool,          # Overall operation success
                "data": Any,              # The actual response data
                "message": str,           # Human-readable message
                "timestamp": str,         # ISO 8601 timestamp
                "metadata": {             # Additional metadata
                    "handler": str,       # Name of the handler class
                    "intent": str,        # Intent key if provided
                    **kwargs              # Any additional metadata
                },
                "error": Optional[Dict],  # Error details if success=False
                "pagination": Optional[Dict]  # Pagination info if applicable
            }
        """
        handler_name = self.__class__.__name__
        timestamp = self._get_timestamp()
        
        # Initialize response with common fields
        response = {
            "success": success if success is not None else True,  # Default to success if not specified
            "data": data,
            "message": message or "",
            "timestamp": timestamp,
            "metadata": {
                "handler": handler_name,
                "intent": intent_key,
                **kwargs
            }
        }
        
        # Handle error cases
        if not response["success"]:
            if isinstance(data, dict) and "error" in data:
                response["error"] = data["error"]
                response["message"] = data.get("message", message or "An error occurred")
            elif not response["message"]:
                response["message"] = "An unknown error occurred"
        
        # Add pagination info if available in the data
        if isinstance(data, dict):
            pagination_keys = ["page", "pageSize", "total", "totalPages"]
            pagination = {k: data.pop(k) for k in pagination_keys if k in data}
            if pagination:
                response["pagination"] = pagination
        
        # Preserve backward compatibility with existing code
        if "success" not in response:
            response["success"] = True
            
        return response
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _extract_data_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        Extract data from a nested dictionary using a dot-notation path
        
        Args:
            data: Dictionary to extract from
            path: Dot-notation path (e.g., "results.items.0.name")
            
        Returns:
            Extracted value or None if not found
        """
        parts = path.split('.')
        current = data
        
        for part in parts:
            # Handle array indices
            if part.isdigit() and isinstance(current, list):
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            # Handle dictionary keys
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _filter_data(self, data: List[Dict[str, Any]], 
                    filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter a list of dictionaries based on filter criteria
        
        Args:
            data: List of dictionaries to filter
            filters: Dictionary of field-value pairs to filter on
            
        Returns:
            Filtered list of dictionaries
        """
        if not filters:
            return data
        
        filtered_data = []
        
        for item in data:
            match = True
            
            for field, value in filters.items():
                # Handle nested fields with dot notation
                if '.' in field:
                    item_value = self._extract_data_by_path(item, field)
                else:
                    item_value = item.get(field)
                
                # Check if the value exists
                if item_value is None:
                    match = False
                    break
                
                # Special handling for status filter
                if field == 'status' and value == 'error':
                    # Match any error status (down, err-disabled, etc.)
                    if str(item_value).lower() not in ['down', 'err-disabled', 'error', 'errdisabled']:
                        match = False
                        break
                elif str(item_value).lower() != str(value).lower():  # Case-insensitive match for other fields
                    match = False
                    break
            
            if match:
                filtered_data.append(item)
        
        return filtered_data
    
    def _aggregate_data(self, data: List[Dict[str, Any]], 
                       group_by: str, 
                       aggregate_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate data by a field
        
        Args:
            data: List of dictionaries to aggregate
            group_by: Field to group by
            aggregate_field: Optional field to aggregate
            
        Returns:
            Dictionary of aggregated data
        """
        result = {}
        
        for item in data:
            # Get the group key
            if '.' in group_by:
                key = self._extract_data_by_path(item, group_by)
            else:
                key = item.get(group_by)
            
            if key is None:
                key = "unknown"
            
            # Initialize group if it doesn't exist
            if key not in result:
                result[key] = []
            
            # Add the item or just the aggregate field
            if aggregate_field:
                if '.' in aggregate_field:
                    value = self._extract_data_by_path(item, aggregate_field)
                else:
                    value = item.get(aggregate_field)
                
                if value is not None:
                    result[key].append(value)
            else:
                result[key].append(item)
        
        return result
    
    def _get_summary_stats(self, data: List[Dict[str, Any]], 
                          fields: List[str]) -> Dict[str, Any]:
        """
        Get summary statistics for a list of dictionaries
        
        Args:
            data: List of dictionaries
            fields: List of fields to get statistics for
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            "total_count": len(data)
        }
        
        for field in fields:
            field_values = {}
            
            for item in data:
                # Get the field value
                if '.' in field:
                    value = self._extract_data_by_path(item, field)
                else:
                    value = item.get(field)
                
                if value is not None:
                    # Convert value to string for counting
                    str_value = str(value)
                    
                    if str_value in field_values:
                        field_values[str_value] += 1
                    else:
                        field_values[str_value] = 1
            
            # Add to stats
            stats[f"{field}_distribution"] = field_values
        
        return stats
    
    def _log_handler_execution(self, intent: str, entities: Dict[str, Any], 
                              success: bool, response_summary: str) -> None:
        """
        Log handler execution for traceability
        
        Args:
            intent: Intent name
            entities: Entities used
            success: Whether execution was successful
            response_summary: Summary of the response
        """
        logger.info(f"Handler execution: {intent}")
        logger.info(f"Entities: {entities}")
        logger.info(f"Success: {success}")
        logger.info(f"Response: {response_summary}")
    
    def validate_required_entities(self, entities: Dict[str, Any], 
                                  required_entities: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required entities are present
        
        Args:
            entities: Dictionary of entities
            required_entities: List of required entity names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_entities = []
        
        for entity_name in required_entities:
            if entity_name not in entities or entities[entity_name] is None:
                missing_entities.append(entity_name)
        
        if missing_entities:
            error_message = f"Missing required entities: {', '.join(missing_entities)}"
            return False, error_message
        
        return True, None
    
    def validate_endpoint_parameters(self, endpoint_name: str, provided_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all required parameters are provided for an API endpoint
        
        Args:
            endpoint_name: Name of the endpoint to validate parameters for
            provided_params: Dictionary of parameters provided by the caller
            
        Returns:
            Tuple of (is_valid: bool, missing_params: list[str])
        """
        try:
            return validate_endpoint_params(endpoint_name, provided_params)
        except Exception as e:
            logger.warning(f"Parameter validation failed for {endpoint_name}: {e}")
            return True, []  # Fallback to allow execution if validation fails
    
    def get_validated_parameters(self, endpoint_name: str, **provided_params) -> Dict[str, Any]:
        """
        Get complete parameter set for an endpoint with validation and defaults
        
        Args:
            endpoint_name: Name of the endpoint
            **provided_params: Parameters provided by the caller
            
        Returns:
            Dictionary of complete parameters for the endpoint
            
        Raises:
            ValueError: If required parameters are missing
        """
        try:
            return get_endpoint_params_with_validation(endpoint_name, **provided_params)
        except Exception as e:
            logger.warning(f"Parameter validation failed for {endpoint_name}: {e}")
            return provided_params  # Fallback to provided params if validation fails
    
    def get_endpoint_parameter_info(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get information about endpoint parameters (required, optional, defaults)
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Dictionary containing parameter information
        """
        try:
            return get_endpoint_info(endpoint_name)
        except Exception as e:
            logger.warning(f"Could not get parameter info for {endpoint_name}: {e}")
            return {"required": [], "optional": [], "defaults": {}}
