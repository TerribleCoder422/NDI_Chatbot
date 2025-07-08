#!/usr/bin/env python3
"""
Network Interface Handler for NDI Chatbot
This module provides a handler for network interface queries.
"""

import json
import logging
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from cisco_api.common.api_client import BaseAPIClient
from cisco_api.nexus.dashboard_client import NexusDashboardClient
from agents.verified_endpoints import get_endpoints_for_intent
from ..handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class NetworkInterfaceHandler(BaseHandler):
    """Handler for network interface queries
    
    This handler processes queries related to network interfaces and
    retrieves data from the appropriate API endpoints using the NexusDashboardClient singleton.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the network interface handler
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # Default base URL
        self.base_url = "https://10.6.11.10"  # Default Nexus Dashboard URL
        
        # Override base URL with config if provided
        if config and 'api_base_url' in config:
            self.base_url = config.get('api_base_url')
            
        # Get the singleton NexusDashboardClient instance with the base_url
        self.nd_client = NexusDashboardClient.get_instance(base_url=self.base_url)
        
        # Update the ND client's base URL if it's different
        if self.nd_client.base_url != self.base_url:
            self.nd_client.base_url = self.base_url
        
        # Get verified endpoints from verified_endpoints.py
        self.interface_endpoints = get_endpoints_for_intent("network_interfaces")
        
        # If no verified endpoints found, use fallback endpoints
        if not self.interface_endpoints:
            logger.warning("No verified endpoints found for network_interfaces intent. Using fallback endpoints.")
            self.interface_endpoint = "/sedgeapi/v1/cisco-nir/api/api/v1/interfaces/summary"
            self.device_interface_endpoint = "/sedgeapi/v1/cisco-nir/api/api/v1/protocols/details"
        else:
            # Use the first endpoint as the default interface endpoint
            self.interface_endpoint = self.interface_endpoints[0]
            # Use the second endpoint as the device interface endpoint if available
            self.device_interface_endpoint = (
                self.interface_endpoints[1] 
                if len(self.interface_endpoints) > 1 
                else self.interface_endpoint
            )
            logger.info(f"Using verified endpoints: {self.interface_endpoints}")
        
        # Override with config if provided
        if config:
            if 'interface_endpoint' in config:
                self.interface_endpoint = config.get('interface_endpoint')
            if 'device_interface_endpoint' in config:
                self.device_interface_endpoint = config.get('device_interface_endpoint')
        
        logger.info(f"NetworkInterfaceHandler initialized with endpoints: {self.interface_endpoint}, {self.device_interface_endpoint}")
    
    def _make_api_call(self, endpoint: str, method: str = "GET", params: Optional[Dict] = None, 
                      data: Optional[Dict] = None, json_data: Optional[Dict] = None, 
                      headers: Optional[Dict] = None) -> Tuple[Any, bool]:
        """
        Make an API call with proper token handling and retries
        
        Args:
            endpoint: API endpoint (can be relative or absolute)
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            json_data: JSON data for the request body
            headers: Additional headers
            
        Returns:
            Tuple of (response_data, success)
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Ensure we have a valid token
                if not self.nd_client.validate_token():
                    logger.warning("Token validation failed, re-authenticating...")
                    if not hasattr(self.nd_client, 'authenticate') or not callable(self.nd_client.authenticate):
                        logger.error("NexusDashboardClient does not have an authenticate method")
                        return None, False
                    
                    # Use the stored credentials if available
                    username = getattr(self.nd_client, 'username', None) or 'admin'
                    password = getattr(self.nd_client, 'password', None) or '32!somuL'
                    
                    if not self.nd_client.authenticate(username, password):
                        logger.error(f"Failed to re-authenticate with username: {username}")
                        return None, False
                    else:
                        logger.info(f"Successfully re-authenticated with username: {username}")
                
                # Prepare the full URL - ensure it's absolute
                if endpoint.startswith('http'):
                    full_url = endpoint
                else:
                    # Make sure base_url doesn't have trailing slash and endpoint doesn't have leading slash
                    base = self.base_url.rstrip('/')
                    path = endpoint.lstrip('/')
                    full_url = f"{base}/{path}"
                
                logger.info(f"Full API URL: {full_url}")
                
                # Prepare headers
                request_headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                if headers:
                    request_headers.update(headers)
                
                # Add auth token if available
                # Get the latest token from the singleton instance to ensure we're using the most up-to-date token
                latest_client = NexusDashboardClient.get_instance(base_url=self.base_url)
                if hasattr(latest_client, 'token') and latest_client.token:
                    request_headers['Authorization'] = f'Bearer {latest_client.token}'
                    logger.debug(f"Using token: {latest_client.token[:10]}... (truncated)")
                    
                    # Also add CSRF token if available
                    if hasattr(latest_client, 'csrf_token') and latest_client.csrf_token:
                        request_headers['X-CSRF-Token'] = latest_client.csrf_token
                        request_headers['X-Requested-With'] = 'XMLHttpRequest'
                        logger.debug(f"Using CSRF token: {latest_client.csrf_token[:10]}... (truncated)")
                else:
                    logger.warning("No token available for API call")
                
                # Log the request details
                logger.info(f"Making {method} request to {full_url}")
                if params:
                    logger.info(f"Query params: {params}")
                if data:
                    logger.debug(f"Request data: {data}")
                if json_data:
                    logger.debug(f"Request JSON: {json_data}")
                
                # Make the request
                response = requests.request(
                    method=method,
                    url=full_url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    verify=self.nd_client.verify_ssl if hasattr(self.nd_client, 'verify_ssl') else False,  # Default to False for development
                    timeout=(10, 30)  # 10s connect, 30s read
                )
                
                # Log response status and headers for debugging
                logger.info(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                
                # Handle authentication errors
                if response.status_code == 401:
                    logger.warning("Authentication failed (401), refreshing token...")
                    self.nd_client.token = None  # Force re-authentication
                    if attempt < MAX_RETRIES - 1:  # Don't sleep on the last attempt
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                
                # Check for other error status codes
                response.raise_for_status()
                
                # Return the JSON response if available, otherwise return the raw content
                try:
                    json_response = response.json()
                    logger.debug(f"Response JSON keys: {list(json_response.keys()) if isinstance(json_response, dict) else 'Not a dict'}")
                    return json_response, True
                except ValueError:
                    logger.warning("Response not JSON, returning raw content")
                    return response.content, True
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401 and attempt < MAX_RETRIES - 1:
                    logger.warning(f"HTTP 401 Unauthorized (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                logger.error(f"HTTP error: {str(e)}")
                # Log response content for debugging
                try:
                    logger.error(f"Error response content: {e.response.content.decode('utf-8')[:500]}")
                except:
                    pass
                return None, False
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < MAX_RETRIES - 1:  # Don't sleep on the last attempt
                    time.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
                return None, False
        
        return None, False
    
    def handle(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle network interface intent with proper token synchronization and error handling
        
        Args:
            entities: Validated entities
            context: Context information
            
        Returns:
            Response dictionary
        """
        try:
            logger.info(f"[NetworkInterfaceHandler] Handling network_interfaces intent with entities: {entities}")
            
            # Ensure we have a valid token
            if not self.nd_client.validate_token():
                logger.warning("Token validation failed, re-authenticating...")
                
                # Use the stored credentials if available
                username = getattr(self.nd_client, 'username', None) or 'admin'
                password = getattr(self.nd_client, 'password', None) or '32!somuL'
                
                if not self.nd_client.authenticate(username, password):
                    error_msg = f"Failed to authenticate with Nexus Dashboard using username: {username}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "message": error_msg,
                        "data": {},
                        "data_source": "None - Authentication failed",
                        "timestamp": self._get_current_timestamp()
                    }
                else:
                    logger.info(f"Successfully authenticated with username: {username}")
            
            # Normalize entity names and handle aliases
            entities = self._normalize_entities(entities)
            
            # Use the interfaces/summary endpoint which returns real interface data
            # The protocols/details endpoint returns empty data {} for this environment
            endpoint = self.interface_endpoint
            logger.info(f"[NetworkInterfaceHandler] Using endpoint: {endpoint} for interface summary data")
                
            # Use the new parameter validation system
            endpoint_name = 'get_interfaces_summary' if 'interfaces/summary' in endpoint else 'get_protocols_details_interface'
            
            # Prepare base parameters from entities
            provided_params = {}
            
            # Add device filter if specified - map device to siteName for API compatibility
            if 'device' in entities and entities['device']:
                # The interfaces summary API uses 'siteName' parameter, not 'device'
                if entities['device'] != 'all':
                    provided_params['siteName'] = entities['device']
                    logger.info(f"[NetworkInterfaceHandler] Filtering interfaces for siteName: {entities['device']}")
                else:
                    # Even for 'all' devices, use default siteName to ensure API returns data
                    provided_params['siteName'] = 'EDNJ-NXOS-Standalone'  # Default from verified_endpoints.py
                    logger.info(f"[NetworkInterfaceHandler] Using default siteName for 'all' devices: {provided_params['siteName']}")
            else:
                # Always include default siteName if no device specified
                provided_params['siteName'] = 'EDNJ-NXOS-Standalone'
                logger.info(f"[NetworkInterfaceHandler] Using default siteName: {provided_params['siteName']}")
            
            # Add interface filter if specified
            if 'interface' in entities and entities['interface']:
                provided_params['interface'] = entities['interface']
                logger.info(f"[NetworkInterfaceHandler] Filtering for interface: {entities['interface']}")
            
            # Use parameter validation to get complete parameter set
            try:
                params = self.get_validated_parameters(endpoint_name, **provided_params)
                logger.info(f"[NetworkInterfaceHandler] Using validated parameters: {params}")
            except ValueError as e:
                logger.error(f"[NetworkInterfaceHandler] Parameter validation failed: {e}")
                # Fallback to basic parameters if validation fails
                params = provided_params
            
            # Override with entity-specific parameters, but only for appropriate endpoints
            # Note: protocols/details endpoint works best with default parameters only
            # Adding custom parameters like 'device' can cause the API to return empty results
            if 'hostname' in entities:
                params['hostname'] = entities['hostname']
            if 'interface_name' in entities:
                params['interface'] = entities['interface_name']
            if 'status' in entities:
                params['status'] = entities['status']
            
            # CRITICAL: Do not add 'device' parameter as it causes protocols/details to return empty data
            # The endpoint works correctly with the default parameters from verified_endpoints.py
            
            # Use the default parameters from verified_endpoints.py
            # Do not override siteGroupName and siteName as they need to match the environment
            # The default parameters are already set correctly in verified_endpoints.py
            
            # Make the API call with retry logic
            logger.info(f"[NetworkInterfaceHandler] Making API call to {endpoint} with params: {params}")
            response_data, success = self._make_api_call(endpoint, method="GET", params=params)
            
            # Extract interface data from the API response
            interface_data = []
            if isinstance(response_data, dict):
                # Check for interfaceStats and interfaceStatus in the response
                if 'interfaceStats' in response_data and 'interfaceStatus' in response_data:
                    logger.info("[NetworkInterfaceHandler] Found interfaceStats and interfaceStatus in response")
                    # Extract and combine the interface data
                    stats = response_data.get('interfaceStats', {})
                    statuses = response_data.get('interfaceStatus', {})
                    
                    # Create interface entries by combining stats and status
                    for intf_name, intf_stats in stats.items():
                        intf_entry = {
                            'name': intf_name,
                            'stats': intf_stats,
                            'status': statuses.get(intf_name, {})
                        }
                        interface_data.append(intf_entry)
                    
                    logger.info(f"[NetworkInterfaceHandler] Processed {len(interface_data)} interfaces from API response")
                else:
                    # Fallback to other possible keys if interfaceStats/Status not found
                    interface_data = response_data.get('items', 
                                                    response_data.get('data', 
                                                                    response_data.get('interfaces', [])))
                    if not interface_data:
                        logger.warning(f"[NetworkInterfaceHandler] No interface data found in API response. Response keys: {list(response_data.keys())}")
            elif isinstance(response_data, list):
                interface_data = response_data
            
            # Structure the response to match what the query workflow expects:
            # - Success and status message at top level
            # - Actual interface data nested under the 'data' key
            # - Interface data specifically under 'interface_statistics' key inside 'data'
            return {
                'success': success,
                'response': '',
                'data': {
                    'interface_statistics': interface_data
                },
                'source': 'NDI API - Real Data',
                'totalCount': response_data.get('totalCount', len(interface_data)) if isinstance(response_data, dict) else len(interface_data),
                'raw_response': response_data,
            }
                
        except Exception as e:
            # Always use the hybrid parser for response formatting (exception case)
            logger.error(f"[NetworkInterfaceHandler] Error handling network_interfaces intent: {str(e)}\n{traceback.format_exc()}")
            return self._format_response({}, False, message=f"Error handling network_interfaces intent: {str(e)}", intent_key="network_interfaces")
    
    def _normalize_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize entity names and handle aliases
        
        This method standardizes entity names and adds aliases to ensure
        that all relevant forms of entities are recognized.
        
        Args:
            entities: Dictionary of entities extracted from user query
            
        Returns:
            Normalized entities dictionary
        """
        if not entities:
            return {}
        
        normalized = entities.copy()
        
        # Handle interface name aliases
        if 'interface' in normalized and not 'interface_name' in normalized:
            normalized['interface_name'] = normalized['interface']
        if 'port' in normalized and not 'interface_name' in normalized:
            normalized['interface_name'] = normalized['port']
        if 'name' in normalized and not 'interface_name' in normalized:
            normalized['interface_name'] = normalized['name']
            
        # Handle status aliases
        if 'state' in normalized and not 'status' in normalized:
            normalized['status'] = normalized['state']
            
        # Handle device aliases
        if 'switch' in normalized and not 'device' in normalized:
            normalized['device'] = normalized['switch']
        if 'router' in normalized and not 'device' in normalized:
            normalized['device'] = normalized['router']
            
        # Handle interface type aliases
        if 'if_type' in normalized and not 'type' in normalized:
            normalized['type'] = normalized['if_type']
            
        logger.debug(f"[NetworkInterfaceHandler] Entity normalization: {entities} → {normalized}")
        return normalized
        
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        
        Returns:
            Current timestamp string
        """
        return datetime.now().isoformat()
        
    def _process_interface_data(self, data: Any, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process interface data based on entities
        
        Args:
            data: Raw interface data from API
            entities: Dictionary of entities for filtering
            
        Returns:
            Processed and filtered interface data
        """
        # Log the raw data structure for debugging
        logger.info(f"[NetworkInterfaceHandler] Raw data type: {type(data)}")
        if isinstance(data, dict):
            logger.info(f"[NetworkInterfaceHandler] Raw data keys: {list(data.keys())}")
            logger.debug(f"[NetworkInterfaceHandler] Raw data sample: {json.dumps(data, indent=2)[:500]}...")
        
        # Handle different possible response formats
        interfaces = []
        if isinstance(data, dict):
            # Check for empty response
            if not data:
                logger.warning("[NetworkInterfaceHandler] Empty response dictionary received")
                return []
                
            # Try to extract interfaces from different possible structures
            if 'interfaces' in data:
                interfaces = data['interfaces']
            elif 'interfaceStats' in data and 'interfaceStatus' in data:
                # Handle interfaces/summary endpoint response format
                logger.info("[NetworkInterfaceHandler] Processing interfaceStats and interfaceStatus data")
                parsed_stats = self._parse_interface_summary_response(data)
                # Instead of returning a list with a dictionary, just return the parsed stats directly
                # This will be used in the final handler response
                return parsed_stats
            elif 'protocols' in data and isinstance(data['protocols'], list):
                # Handle protocols/details endpoint response format from ND API
                for protocol in data['protocols']:
                    if isinstance(protocol, dict):
                        # Check for 'entries' array which contains the actual interface data
                        if 'entries' in protocol and isinstance(protocol['entries'], list):
                            # Each entry contains interface details
                            for entry in protocol['entries']:
                                if isinstance(entry, dict):
                                    # Add fabric/node context to each interface
                                    entry['fabricName'] = protocol.get('fabricName', 'Unknown')
                                    entry['nodeName'] = protocol.get('nodeName', 'Unknown')
                                    entry['siteName'] = protocol.get('siteName', 'Unknown')
                                    interfaces.append(entry)
                        # Also check for interface details at the top level
                        elif all(k in protocol for k in ['sourceNameLabel', 'adminStatus', 'operStatus']):
                            interfaces.append(protocol)
            elif 'interface' in data:  # Handle single interface response
                interfaces = [data['interface']]
            elif all(isinstance(v, dict) or v is None for v in data.values()):
                # If all values are dicts or None, treat as interface collection
                interfaces = [v for v in data.values() if v is not None]
            elif any(k in data for k in ['name', 'status', 'speed', 'adminStatus', 'operStatus']):
                # If it's a flat dictionary with interface data, use it directly
                interfaces = [data]
            else:
                logger.warning(f"[NetworkInterfaceHandler] Unrecognized dictionary format: {list(data.keys())}")
        elif isinstance(data, list):
            interfaces = data
        else:
            logger.warning(f"[NetworkInterfaceHandler] Unexpected data type: {type(data)}")
        
        # Filter out any None values
        interfaces = [i for i in interfaces if i is not None]
        
        logger.info(f"[NetworkInterfaceHandler] Found {len(interfaces)} interfaces to process")
        if interfaces and isinstance(interfaces[0], dict):
            sample_keys = list(interfaces[0].keys())
            logger.info(f"[NetworkInterfaceHandler] Sample interface keys: {sample_keys}")
            
            # Log sample interface data for debugging
            sample_interface = {k: v for k, v in interfaces[0].items() if not isinstance(v, (dict, list))}
            logger.debug(f"[NetworkInterfaceHandler] Sample interface data: {json.dumps(sample_interface, indent=2)}")
        else:
            logger.warning("[NetworkInterfaceHandler] No valid interfaces found in response")
            
        # Create filters based on entities
        filters = {}
        
        # Handle different possible field names for interface name
        name_fields = ['interface_name', 'name', 'interface', 'port']
        for field in name_fields:
            if field in entities and entities[field]:
                filters['name'] = entities[field]
                break
        
        # Handle different possible field names for status
        status_fields = ['status', 'state', 'operStatus', 'adminStatus']
        for field in status_fields:
            if field in entities and entities[field]:
                filters['status'] = entities[field]
                break
        
        # Handle error status filtering
        error_indicators = ['down', 'err', 'fail', 'disabled', 'shut']
        if any(indicator in str(entities.get('status', '')).lower() for indicator in error_indicators):
            filters['error_state'] = True
        
        # Handle interface type filtering
        if 'type' in entities and entities['type']:
            filters['type'] = entities['type']
        
        # Apply filters
        filtered_data = self._filter_data(interfaces, filters)
        
        # Extract relevant fields for each interface
        processed_interfaces = []
        
        for interface in filtered_data:
            # Map fields from different possible API response formats
            status = interface.get('status', interface.get('operStatus', 'Unknown'))
            admin_status = interface.get('adminStatus', interface.get('adminState', 'Unknown'))
            
            # Get error counters from various possible field names
            error_counters = {
                'in_errors': interface.get('inErrors', interface.get('inputErrors', 0)),
                'out_errors': interface.get('outErrors', interface.get('outputErrors', 0)),
                'crc_errors': interface.get('crcErrors', 0),
                'frame_errors': interface.get('frameErrors', 0),
                'overrun_errors': interface.get('overrunErrors', 0),
                'ignored_errors': interface.get('ignored', 0),
                'collisions': interface.get('collisions', 0)
            }
            
            # Convert all error counts to integers
            error_counters = {k: int(v) if v is not None else 0 for k, v in error_counters.items()}
            
            # Check if interface is in error state using our comprehensive check
            is_error_state = self._is_interface_in_error(interface)
            
            # If we think there's an error but no specific errors were reported, set a flag
            if is_error_state and not any(error_counters.values()):
                error_counters['error_state_detected'] = True
            
            # Get interface name, handling ND API specific field names
            interface_name = next((interface.get(field) for field in 
                                 ['sourceNameLabel', 'name', 'interfaceName', 'ifName', 'interface'] 
                                 if field in interface), 'Unknown')
            
            # Build the processed interface data with ND API field mappings
            processed_interface = {
                'name': interface_name,
                'status': status,
                'admin_status': admin_status,
                'description': interface.get('interfaceDescription', interface.get('description', interface.get('ifDescr', ''))),
                'speed': interface.get('portSpeed', interface.get('speed', interface.get('speedMbps', 'Unknown'))),
                'mac_address': interface.get('macAddress', interface.get('physAddress', 'Unknown')),
                'ip_address': interface.get('ipAddress', interface.get('ipv4Address', 'Unknown')),
                'mtu': interface.get('mtu', interface.get('mtuSize', 'Unknown')),
                'vlan': interface.get('vlan', interface.get('vlanId', 'Unknown')),
                'interface_type': interface.get('interfaceType', 'Unknown'),
                'fabric_name': interface.get('fabricName', 'Unknown'),
                'node_name': interface.get('nodeName', 'Unknown'),
                'site_name': interface.get('siteName', 'Unknown'),
                'anomaly_score': interface.get('anomalyScore', 0),
                'has_errors': is_error_state,
                'last_change': interface.get('lastChange', interface.get('lastStatusChange', 'Unknown')),
                'errors': error_counters
            }
            
            processed_interfaces.append(processed_interface)
        
        return processed_interfaces
    
    def _is_interface_in_error(self, interface: Dict[str, Any]) -> bool:
        """
        Check if an interface is in an error state based on various indicators.
        
        Args:
            interface: Dictionary containing interface data
            
        Returns:
            bool: True if the interface is in an error state, False otherwise
        """
        if not isinstance(interface, dict):
            return False
            
        # Common error indicators in interface status
        error_indicators = [
            'down', 'err', 'fail', 'disabled', 'shut', 'inactive', 'notconnect',
            'error', 'err-disabled', 'suspended', 'disabled', 'notPresent', 'monitor',
            'inconsistent', 'wrong', 'unusable', 'incomplete', 'inactive', 'standby'
        ]
        
        # Check various status fields that might indicate an error
        status_fields = [
            'status', 'operStatus', 'adminStatus', 'oper_state', 'admin_state',
            'operationalStatus', 'adminstrativeStatus', 'state', 'operState'
        ]
        
        # Check if any status field contains an error indicator
        for field in status_fields:
            status = str(interface.get(field, '')).lower()
            if any(indicator in status for indicator in error_indicators):
                return True
        
        # Check error counters
        error_counters = [
            'inErrors', 'outErrors', 'crc', 'frame', 'overrun', 
            'ignored', 'abort', 'reset', 'lateCollision', 'deferred',
            'collisions', 'inputErrors', 'outputErrors', 'rxErrors', 'txErrors'
        ]
        
        # If any error counter is greater than 0
        for counter in error_counters:
            if counter in interface and int(interface.get(counter, 0)) > 0:
                return True
                
        # Check for interface down states
        admin_up = str(interface.get('adminStatus', '')).lower() in ['up', 'enabled', 'true', '1']
        oper_up = str(interface.get('operStatus', '')).lower() in ['up', 'connected', 'true', '1']
        
        # If admin up but operationally down, it's considered an error
        if admin_up and not oper_up:
            return True
            
        # Check for last change timestamps that might indicate flapping
        if 'lastChange' in interface and interface['lastChange']:
            try:
                # If last change was recent (within last 5 minutes), might indicate flapping
                last_change = datetime.fromisoformat(interface['lastChange'].replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - last_change).total_seconds() < 300:
                    return True
            except (ValueError, TypeError):
                # If we can't parse the timestamp, continue
                pass
                
        return False
        
    def _filter_data(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                # Special handling for 'all' value - matches anything
                elif value == 'all':
                    continue
                # Regular equality check
                elif str(item_value).lower() != str(value).lower():
                    match = False
                    break
            
            if match:
                filtered_data.append(item)
        
        return filtered_data
        
    def _extract_data_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        Extract data from a nested dictionary using a dot-notation path
        
        Args:
            data: Dictionary to extract from
            path: Dot-notation path (e.g., "results.items.0.name")
            
        Returns:
            Extracted value or None if not found
        """
        if not path or not data:
            return None
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            # Handle array indices
            if part.isdigit():
                part = int(part)
            
            try:
                current = current[part]
            except (KeyError, TypeError, IndexError):
                return None
        
        return current
    
    def _get_interface_stats(self, interfaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics for interfaces
        
        Args:
            interfaces: List of processed interfaces
            
        Returns:
            Dictionary of interface statistics
        """
        stats = {
            'total_count': len(interfaces),
            'status_distribution': {},
            'speed_distribution': {},
            'error_stats': {
                'interfaces_with_errors': 0,
                'total_errors': 0,
                'error_types': {}
            }
        }
        
        # Count interfaces by status and collect error information
        for interface in interfaces:
            status = interface.get('status', 'Unknown')
            speed = interface.get('speed', 'Unknown')
            
            # Update error statistics
            if interface.get('has_errors', False):
                stats['error_stats']['interfaces_with_errors'] += 1
                
                # Count specific error types
                errors = interface.get('errors', {})
                for error_type, count in errors.items():
                    if isinstance(count, (int, float)) and count > 0:
                        stats['error_stats']['total_errors'] += count
                        if error_type in stats['error_stats']['error_types']:
                            stats['error_stats']['error_types'][error_type] += count
                        else:
                            stats['error_stats']['error_types'][error_type] = count
            
            # Update status distribution
            if status in stats['status_distribution']:
                stats['status_distribution'][status] += 1
            else:
                stats['status_distribution'][status] = 1
            
            # Update speed distribution
            if speed in stats['speed_distribution']:
                stats['speed_distribution'][speed] += 1
            else:
                stats['speed_distribution'][speed] = 1
        
        return stats
    
    def _parse_interface_summary_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from the interfaces/summary endpoint
        
        Args:
            data: Raw API response containing interfaceStats and interfaceStatus
            
        Returns:
            Clean statistics dictionary for LLM consumption
        """
        # Extract interface statistics and status data
        interface_stats = data.get('interfaceStats', {})
        interface_status = data.get('interfaceStatus', {})
        
        logger.info(f"[NetworkInterfaceHandler] interfaceStats keys: {list(interface_stats.keys()) if interface_stats else 'None'}")
        logger.info(f"[NetworkInterfaceHandler] interfaceStatus keys: {list(interface_status.keys()) if interface_status else 'None'}")
        
        # Extract the actual numeric statistics from interfaceStats
        total_interfaces = interface_stats.get('totalInterfaces', 0)
        admin_up = interface_stats.get('adminUp', 0)
        admin_down = interface_stats.get('adminDown', 0)
        oper_up = interface_stats.get('operUp', 0)
        oper_down = interface_stats.get('operDown', 0)
        
        # Also extract interface status data
        status_up = interface_status.get('up', 0)
        status_down = interface_status.get('down', 0)
        status_not_connected = interface_status.get('notConnected', 0)
        
        logger.info(f"[NetworkInterfaceHandler] Real API data - Total: {total_interfaces}, AdminUp: {admin_up}, AdminDown: {admin_down}, OperUp: {oper_up}, OperDown: {oper_down}")
        logger.info(f"[NetworkInterfaceHandler] Status data - Up: {status_up}, Down: {status_down}, NotConnected: {status_not_connected}")
        
        # Return clean, simple statistics that the LLM can easily understand
        clean_stats = {
            'total_interfaces': total_interfaces,
            'operational_status': {
                'up': oper_up,
                'down': oper_down
            },
            'administrative_status': {
                'up': admin_up,
                'down': admin_down
            },
            'connection_status': {
                'up': status_up,
                'down': status_down,
                'not_connected': status_not_connected
            },
            'summary': f"{total_interfaces} total interfaces: {oper_up} operationally up, {oper_down} operationally down"
        }
        
        logger.info(f"[NetworkInterfaceHandler] Returning clean statistics: {clean_stats}")
        return clean_stats
