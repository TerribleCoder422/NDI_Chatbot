#!/usr/bin/env python3
"""
Network Endpoint Handler for NDI Chatbot
This module provides a handler for network endpoint queries.
"""

import logging
import traceback
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from ..handlers.base_handler import BaseHandler
from agents.verified_endpoints import get_ndi_endpoint, get_endpoints_for_intent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkEndpointHandler(BaseHandler):
    """
    Handler for network endpoint queries
    
    This handler processes queries related to network endpoints and
    retrieves data from the appropriate API endpoints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the network endpoint handler
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # Default base URL
        self.base_url = "https://10.6.11.10"  # Default Nexus Dashboard URL
        
        # Override base URL with config if provided
        if config and 'api_base_url' in config:
            self.base_url = config.get('api_base_url')
        
        # Get verified endpoints from verified_endpoints.py
        # For endpoints, we use the verified endpoints from NDI_ENDPOINTS
        self.endpoint_endpoints = get_endpoints_for_intent("network_endpoints")
        
        # If no verified endpoints found, use fallback endpoints
        if not self.endpoint_endpoints:
            logger.warning("No verified endpoints found for network_endpoints intent. Using fallback endpoints.")
            self.endpoint_list_endpoint = "/api/v1/endpoints"
            self.endpoint_count_endpoint = "/api/v1/endpoints/count"
            self.device_endpoint_endpoint = "/api/v1/devices/{device}/endpoints"
        else:
            # Use the first endpoint as the endpoint list endpoint
            self.endpoint_list_endpoint = self.endpoint_endpoints[0]
            # Use the second endpoint as the endpoint count endpoint if available
            if len(self.endpoint_endpoints) > 1:
                self.endpoint_count_endpoint = self.endpoint_endpoints[1]
            else:
                self.endpoint_count_endpoint = "/api/v1/endpoints/count"
            # Device endpoint endpoint remains the same as we don't have a specific one in verified_endpoints
            self.device_endpoint_endpoint = "/api/v1/devices/{device}/endpoints"
            
            logger.info(f"Using verified endpoints: {self.endpoint_list_endpoint}, {self.endpoint_count_endpoint}")
        
        # Override with config if provided
        if config:
            if 'endpoint_list_endpoint' in config:
                self.endpoint_list_endpoint = config.get('endpoint_list_endpoint')
            if 'endpoint_count_endpoint' in config:
                self.endpoint_count_endpoint = config.get('endpoint_count_endpoint')
            if 'device_endpoint_endpoint' in config:
                self.device_endpoint_endpoint = config.get('device_endpoint_endpoint')
        
        # Cache flag
        self._using_cached_data = False
        self._cache_timestamp = None
        
        logger.info("NetworkEndpointHandler initialized")
    
    def handle(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle network endpoint intent
        
        Args:
            entities: Validated entities
            context: Context information
            
        Returns:
            Response dictionary
        """
        try:
            # Log the handling of this intent
            logger.info(f"[NetworkEndpointHandler] Handling network_endpoints intent with entities: {entities}")
            
            # Use the new parameter validation system
            endpoint_name = 'get_endpoints_summary'
            
            # Prepare base parameters from entities
            provided_params = {}
            
            # Add device filter if specified
            if 'device' in entities and entities['device']:
                provided_params['device'] = entities['device']
                logger.info(f"[NetworkEndpointHandler] Filtering endpoints for device: {entities['device']}")
            
            # Add endpoint filter if specified
            if 'endpoint' in entities and entities['endpoint']:
                provided_params['endpoint'] = entities['endpoint']
                logger.info(f"[NetworkEndpointHandler] Filtering for endpoint: {entities['endpoint']}")
            
            # Use the same parameter validation pattern as NetworkInterfaceHandler
            try:
                from agents.verified_endpoints import get_endpoint_params_with_validation
                from datetime import datetime, timedelta
                
                # Get validated parameters for endpoints summary
                api_params = get_endpoint_params_with_validation(
                    endpoint_name="get_endpoints_summary",
                    provided_params=provided_params
                )
                
                # Convert relative date strings to timezone-aware timestamps (matching ND UI format)
                from datetime import timezone, timedelta
                pst = timezone(timedelta(hours=-7))  # PST timezone
                
                if 'startDate' in api_params and api_params['startDate'] == "now-24h":
                    api_params['startDate'] = (datetime.now(pst) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S-07:00')
                if 'endDate' in api_params and api_params['endDate'] == "now":
                    api_params['endDate'] = datetime.now(pst).strftime('%Y-%m-%dT%H:%M:%S-07:00')
                
                # Add required site parameters and additional parameters from ND UI
                if 'siteName' not in api_params:
                    api_params['siteName'] = 'EDNJ-NXOS-Standalone'
                if 'siteGroupName' not in api_params:
                    api_params['siteGroupName'] = 'default'
                if 'count' not in api_params:
                    api_params['count'] = '10'
                if 'offset' not in api_params:
                    api_params['offset'] = '0'
                if 'sort' not in api_params:
                    api_params['sort'] = '+ip'
                
                logger.info(f"[NetworkEndpointHandler] Using validated parameters: {api_params}")
            except Exception as e:
                logger.error(f"[NetworkEndpointHandler] Parameter validation failed: {e}")
                # Fallback to basic parameters with timezone-aware timestamps matching ND UI
                from datetime import datetime, timedelta, timezone
                pst = timezone(timedelta(hours=-7))  # PST timezone
                api_params = {
                    'startDate': (datetime.now(pst) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S-07:00'),
                    'endDate': datetime.now(pst).strftime('%Y-%m-%dT%H:%M:%S-07:00'),
                    'siteName': 'EDNJ-NXOS-Standalone',
                    'siteGroupName': 'default',
                    'count': '10',
                    'offset': '0',
                    'sort': '+ip'
                }
                logger.info(f"[NetworkEndpointHandler] Using fallback parameters: {api_params}")
            
            # Determine which endpoint to use
            endpoint = self.endpoint_list_endpoint
            logger.info("[NetworkEndpointHandler] Using endpoint list endpoint")
            
            # Get endpoint data from NDI API
            logger.info("[NetworkEndpointHandler] Retrieving endpoint data from NDI API")
            full_endpoint_url = f"{self.base_url}{endpoint}" if not endpoint.startswith("http") else endpoint
            response_data, success = self._make_api_call(full_endpoint_url, params=api_params)
            
            # Get endpoint count data with proper parameters
            full_count_endpoint_url = f"{self.base_url}{self.endpoint_count_endpoint}" if not self.endpoint_count_endpoint.startswith("http") else self.endpoint_count_endpoint
            # Count endpoint may not need date parameters, but include them for consistency
            count_params = {k: v for k, v in api_params.items() if k not in ['count', 'offset']}
            count_data, count_success = self._make_api_call(full_count_endpoint_url, params=count_params)
            
            # Log data source information
            data_source = "NDI API - Real Data"
            self._using_cached_data = hasattr(self, '_cache_timestamp')
            if self._using_cached_data:
                data_source = f"Cached NDI API Data (timestamp: {getattr(self, '_cache_timestamp', 'unknown')})"
            logger.info(f"[NetworkEndpointHandler] Data source: {data_source}")
            
            # Standardized output: always return endpoint data under the 'endpoints' key
            endpoint_data = []
            if success and isinstance(response_data, dict):
                # Try to get endpoint data from the response
                if 'endpointStats' in response_data:
                    # If we have endpointStats, extract the endpoints from it
                    endpoint_data = response_data['endpointStats'].get('endpoints', [])
                    logger.info(f"[NetworkEndpointHandler] Found {len(endpoint_data)} endpoints in endpointStats")
                else:
                    # Fallback to other possible keys
                    endpoint_data = response_data.get('items', response_data.get('data', response_data.get('endpoints', [])))
                    logger.info(f"[NetworkEndpointHandler] Found {len(endpoint_data)} endpoints in fallback keys")
                
                if not endpoint_data:
                    logger.warning(f"[NetworkEndpointHandler] No endpoint data found in API response. Response keys: {list(response_data.keys())}")
            elif success and isinstance(response_data, list):
                endpoint_data = response_data
                logger.info(f"[NetworkEndpointHandler] Received list of {len(endpoint_data)} endpoints directly")
            
            return {
                "endpoints": endpoint_data,
                "totalCount": response_data.get("totalCount", len(endpoint_data)) if isinstance(response_data, dict) else len(endpoint_data),
                "success": success,
                "source": "NDI API - Real Data",
                "raw_response": response_data,
            }
        except Exception as e:
            # Always use the hybrid parser for response formatting (exception case)
            logger.error(f"[NetworkEndpointHandler] Error handling network_endpoints intent: {str(e)}\n{traceback.format_exc()}")
            return {
                "endpoints": [],
                "totalCount": 0,
                "success": False,
                "source": "NDI API - Real Data",
                "raw_response": {"error": str(e)},
            }
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        
        Returns:
            Current timestamp string
        """
        return datetime.now().isoformat()
    
    def _process_endpoint_data(self, data: Any, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process endpoint data based on entities
        
        Args:
            data: Raw endpoint data from API
            entities: Dictionary of entities for filtering
            
        Returns:
            Processed and filtered endpoint data
        """
        # Ensure data is a list
        if not isinstance(data, list):
            if isinstance(data, dict) and 'endpoints' in data:
                endpoints = data['endpoints']
            else:
                logger.warning("Unexpected data format, expected list or dict with 'endpoints' key")
                return []
        else:
            endpoints = data
        
        # Create filters based on entities
        filters = {}
        
        if 'device' in entities and entities['device']:
            filters['device'] = entities['device']
        
        if 'status' in entities and entities['status']:
            filters['status'] = entities['status']
        
        if 'type' in entities and entities['type']:
            filters['type'] = entities['type']
        
        # Apply filters
        filtered_data = self._filter_data(endpoints, filters)
        
        # Extract relevant fields for each endpoint
        processed_endpoints = []
        
        for endpoint in filtered_data:
            processed_endpoint = {
                'id': endpoint.get('id', 'Unknown'),
                'name': endpoint.get('name', 'Unknown'),
                'ip': endpoint.get('ip', 'Unknown'),
                'mac': endpoint.get('mac', 'Unknown'),
                'device': endpoint.get('device', 'Unknown'),
                'interface': endpoint.get('interface', 'Unknown'),
                'vlan': endpoint.get('vlan', 'Unknown'),
                'status': endpoint.get('status', 'Unknown'),
                'type': endpoint.get('type', 'Unknown'),
                'last_seen': endpoint.get('lastSeen', 'Unknown')
            }
            processed_endpoints.append(processed_endpoint)
        
        return processed_endpoints
    
    def _filter_data(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter data based on filters
        
        Args:
            data: List of data items
            filters: Dictionary of filters
            
        Returns:
            Filtered data
        """
        if not filters:
            return data
        
        filtered_data = data
        
        for key, value in filters.items():
            filtered_data = [item for item in filtered_data if self._matches_filter(item, key, value)]
        
        return filtered_data
    
    def _matches_filter(self, item: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Check if an item matches a filter
        
        Args:
            item: Data item
            key: Filter key
            value: Filter value
            
        Returns:
            True if the item matches the filter, False otherwise
        """
        # Handle special cases
        if key == 'device':
            # Check device name or ID
            return (
                (item.get('device', '').lower() == value.lower()) or
                (item.get('deviceId', '').lower() == value.lower()) or
                (item.get('deviceName', '').lower() == value.lower())
            )
        
        # Handle camelCase variations
        camel_case_key = ''.join([key[0].lower(), key[1:]])
        
        # Check if the key exists in the item
        if key in item:
            return str(item[key]).lower() == str(value).lower()
        elif camel_case_key in item:
            return str(item[camel_case_key]).lower() == str(value).lower()
        
        return False
    
    def _get_endpoint_stats(self, endpoints: List[Dict[str, Any]], count_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate statistics for endpoints
        
        Args:
            endpoints: List of processed endpoints
            count_data: Optional count data from API
            
        Returns:
            Dictionary of endpoint statistics
        """
        stats = {
            "total": len(endpoints),
            "by_status": {},
            "by_type": {},
            "by_device": {}
        }
        
        # Count by status
        for endpoint in endpoints:
            status = endpoint.get('status', 'Unknown')
            if status not in stats["by_status"]:
                stats["by_status"][status] = 0
            stats["by_status"][status] += 1
            
            # Count by type
            endpoint_type = endpoint.get('type', 'Unknown')
            if endpoint_type not in stats["by_type"]:
                stats["by_type"][endpoint_type] = 0
            stats["by_type"][endpoint_type] += 1
            
            # Count by device
            device = endpoint.get('device', 'Unknown')
            if device not in stats["by_device"]:
                stats["by_device"][device] = 0
            stats["by_device"][device] += 1
        
        # Add count data if available
        if count_data and isinstance(count_data, dict):
            stats["api_count"] = count_data
        
        return stats
    
    def _make_api_call(self, endpoint: str, method: str = 'GET', 
                       params: Optional[Dict[str, Any]] = None,
                       data: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, Any]] = None) -> Tuple[Any, bool]:
        """
        Make an API call to the specified endpoint
        
        Args:
            endpoint: API endpoint
            method: HTTP method (default: GET)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Tuple of (response data, success flag)
        """
        try:
            # Use the base class implementation for real API calls
            return super()._make_api_call(endpoint, method, params, data, headers)
        except Exception as e:
            logger.error(f"Error making API call to {endpoint}: {str(e)}")
            return {"error": str(e)}, False
