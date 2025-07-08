#!/usr/bin/env python3
"""
Network Anomaly Handler for NDI Chatbot
This module provides a handler for network anomaly queries.
"""

import logging
import traceback
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from ..handlers.base_handler import BaseHandler
from agents.verified_endpoints import get_ndi_endpoint, get_endpoints_for_intent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkAnomalyHandler(BaseHandler):
    """
    Handler for network anomaly queries
    
    This handler processes queries related to network anomalies and
    retrieves data from the appropriate API endpoints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the network anomaly handler
        
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
        self.anomaly_endpoints = {
            'list': get_ndi_endpoint('overview', 'get_anomalies_summary'),
            'count': get_ndi_endpoint('overview', 'get_anomalies_summary'),  # Same endpoint supports both list and count
            # No direct device-specific endpoint in verified_endpoints, use overview with filters
            'types': get_ndi_endpoint('overview', 'get_anomalies_summary')  # Use same endpoint with type filter
        }
        
        # Log the endpoints being used
        logger.info(f"Using anomaly endpoints: {self.anomaly_endpoints}")
        
        # Override with config if provided
        if config:
            if 'anomaly_list_endpoint' in config:
                self.anomaly_list_endpoint = config.get('anomaly_list_endpoint')
            if 'anomaly_count_endpoint' in config:
                self.anomaly_count_endpoint = config.get('anomaly_count_endpoint')
            if 'device_anomaly_endpoint' in config:
                self.device_anomaly_endpoint = config.get('device_anomaly_endpoint')
            if 'anomaly_types_endpoint' in config:
                self.anomaly_types_endpoint = config.get('anomaly_types_endpoint')
        
        # Cache flag
        self._using_cached_data = False
        self._cache_timestamp = None
        
        logger.info("NetworkAnomalyHandler initialized")
    
    def handle(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle network anomaly intent
        
        Args:
            entities: Validated entities
            context: Context information
            
        Returns:
            Response dictionary
        """
        try:
            # Log the handling of this intent
            logger.info(f"[NetworkAnomalyHandler] Handling network_anomalies intent with entities: {entities}")
            
            # Always use the verified anomalies summary endpoint
            endpoint = self.anomaly_endpoints['list']
            
            # Add device filter if specified in entities
            params = {}
            if 'device' in entities and entities['device']:
                params['device'] = entities['device']
                logger.info(f"[NetworkAnomalyHandler] Filtering anomalies for device: {entities['device']}")
            
            logger.info("[NetworkAnomalyHandler] Using verified anomalies endpoint with params: %s", params)
            
            # Get anomaly data from NDI API using verified endpoint
            logger.info("[NetworkAnomalyHandler] Retrieving anomaly data from NDI API")
            
            # Use the new parameter validation system
            endpoint_name = 'get_anomalies_summary'
            
            # Get the current time for time-based parameters
            current_time = datetime.now()
            end_time = current_time.isoformat()
            # Use a 2-hour window for anomalies (configurable)
            start_time = (current_time - timedelta(hours=2)).isoformat()
            
            # Prepare base parameters
            provided_params = {
                'startDate': start_time,
                'endDate': end_time
            }
            
            # Add optional parameters based on entities
            if 'device' in entities and entities['device']:
                provided_params['siteName'] = entities['device']
                logger.info(f"[NetworkAnomalyHandler] Filtering anomalies for device/site: {entities['device']}")
            
            # Use parameter validation to get complete parameter set
            try:
                api_params = self.get_validated_parameters(endpoint_name, **provided_params)
                logger.info(f"[NetworkAnomalyHandler] Using validated parameters: {api_params}")
            except ValueError as e:
                logger.error(f"[NetworkAnomalyHandler] Parameter validation failed: {e}")
                # Fallback to basic parameters if validation fails
                api_params = provided_params
                api_params.update({
                    'siteGroupName': 'default',
                    'filter': 'cleared:false AND acknowledged:false',
                    'aggr': 'severity',
                    'siteStatus': 'online'
                })
            
            # Make the API call
            full_endpoint_url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
            
            # Log the exact API call details
            logger.info(f"[NetworkAnomalyHandler] Making API call to: {full_endpoint_url}")
            logger.info(f"[NetworkAnomalyHandler] API parameters: {api_params}")
            
            response_data, success = self._make_api_call(full_endpoint_url, method='GET', params=api_params)
            
            # Log the raw API response
            logger.info(f"[NetworkAnomalyHandler] API call success: {success}")
            logger.info(f"[NetworkAnomalyHandler] Raw API response type: {type(response_data)}")
            if isinstance(response_data, dict):
                logger.info(f"[NetworkAnomalyHandler] Raw API response keys: {list(response_data.keys())}")
                if 'totalAnomalyCount' in response_data:
                    logger.info(f"[NetworkAnomalyHandler] Total anomaly count from API: {response_data['totalAnomalyCount']}")
                if 'entries' in response_data:
                    logger.info(f"[NetworkAnomalyHandler] Number of entries in API response: {len(response_data['entries'])}")
                    if response_data['entries']:
                        logger.info(f"[NetworkAnomalyHandler] Sample entry keys: {list(response_data['entries'][0].keys())}")
                        for entry in response_data['entries']:
                            if 'severity' in entry and 'anomalyCount' in entry:
                                logger.info(f"[NetworkAnomalyHandler] Severity {entry['severity']}: {entry['anomalyCount']} anomalies")
            
            # For count, use the same endpoint with count parameter
            count_params = api_params.copy()
            count_params['count'] = 'true'
            logger.info(f"[NetworkAnomalyHandler] Making count API call with params: {count_params}")
            count_data, count_success = self._make_api_call(full_endpoint_url, method='GET', params=count_params)
            
            # Log count response
            if count_success and isinstance(count_data, dict):
                logger.info(f"[NetworkAnomalyHandler] Count API response keys: {list(count_data.keys())}")
            
            # Log data source information
            data_source = "NDI API - Real Data"
            self._using_cached_data = hasattr(self, '_cache_timestamp')
            if self._using_cached_data:
                data_source = f"Cached NDI API Data (timestamp: {getattr(self, '_cache_timestamp', 'unknown')})"
            logger.info(f"[NetworkAnomalyHandler] Data source: {data_source}")
            
            # Standardized output: always return anomalies under the 'anomalies' key
            anomalies = response_data.get('entries', [])
            if not anomalies:
                logger.warning(f"[NetworkAnomalyHandler] No 'entries' key found in API response. Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else type(response_data)}")
                # Optionally try other fallback keys or log the full response
            return {
                "anomalies": anomalies,
                "totalAnomalyCount": response_data.get("totalAnomalyCount", 0),
                "success": success,
                "source": data_source,
                "raw_response": response_data,
            }
        except Exception as e:
            # Log the error
            logger.error(f"[NetworkAnomalyHandler] Error handling network_anomalies intent: {str(e)}\n{traceback.format_exc()}")
            
            # Return error response
            return {
                "success": False,
                "message": f"Error handling network_anomalies intent: {str(e)}",
                "data": {},
                "data_source": "None - Exception occurred",
                "timestamp": self._get_current_timestamp(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        
        Returns:
            Current timestamp string
        """
        return datetime.now().isoformat()
    
    def _process_anomaly_data(self, data: Any, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process anomaly data based on entities
        
        Args:
            data: Raw anomaly data from API
            entities: Dictionary of entities for filtering
            
        Returns:
            Processed and filtered anomaly data
        """
        # Ensure data is a list or extract anomalies from dict
        anomalies = []
        api_stats = {}
        
        if isinstance(data, list):
            # Direct list of anomaly records
            anomalies = data
            logger.info(f"[NetworkAnomalyHandler] Found {len(anomalies)} individual anomaly records")
            
        elif isinstance(data, dict):
            # Extract individual anomaly records if available
            if 'anomalies' in data:
                anomalies = data['anomalies']
                logger.info(f"[NetworkAnomalyHandler] Found {len(anomalies)} anomaly records from 'anomalies' key")
                
            # For the summary API format with totalAnomalyCount and entries by severity
            if 'totalAnomalyCount' in data and 'entries' in data and isinstance(data['entries'], list):
                # Process entries and generate synthetic anomaly summaries by severity
                synthetic_anomalies = []
                
                # Store all stats for later use
                api_stats = {
                    'totalAnomalyCount': data.get('totalAnomalyCount', 0),
                    'totalAnomalyScore': data.get('totalAnomalyScore', 0),
                    'anomalyScore': data.get('anomalyScore', 0),
                    'entries': data.get('entries', [])
                }
                
                # For actionable statistics, create one synthetic entry per severity level
                for entry in data.get('entries', []):
                    if 'severity' in entry and 'anomalyCount' in entry:
                        # Create a synthetic anomaly summary for this severity level
                        severity = entry['severity']
                        count = entry['anomalyCount']
                        score = entry.get('anomalyScore', 0)
                        
                        if count > 0:  # Only add non-zero entries
                            synthetic_anomaly = {
                                'id': f"summary-{severity}",
                                'title': f"{count} {severity} anomalies detected",
                                'description': f"There are {count} {severity} anomalies with a score of {score}",
                                'severity': severity,
                                'type': 'summary',
                                'status': 'active',
                                'count': count,
                                'score': score,
                                'timestamp': self._get_current_timestamp(),
                                'last_updated': self._get_current_timestamp(),
                                'is_synthetic_summary': True  # Flag to indicate this is a summary, not an actual anomaly
                            }
                            synthetic_anomalies.append(synthetic_anomaly)
                
                # If we have no individual anomalies but have synthetic summaries, use them
                if not anomalies and synthetic_anomalies:
                    logger.info(f"[NetworkAnomalyHandler] Using {len(synthetic_anomalies)} synthetic summary entries")
                    anomalies = synthetic_anomalies
            
            # Extract any other statistics available
            for key in ['totalCount', 'count']:
                if key in data:
                    api_stats[key] = data[key]
                    
        else:
            logger.warning(f"[NetworkAnomalyHandler] Unexpected data format: {type(data)}, expected list or dict")
            
        # Store API stats for later use
        self._api_stats = api_stats
        
        # Create filters based on entities
        filters = {}
        
        if 'device' in entities and entities['device']:
            filters['device'] = entities['device']
        
        if 'severity' in entities and entities['severity']:
            filters['severity'] = entities['severity']
        
        if 'type' in entities and entities['type']:
            filters['type'] = entities['type']
        
        if 'status' in entities and entities['status']:
            filters['status'] = entities['status']
        
        # Apply filters
        filtered_data = self._filter_data(anomalies, filters)
        
        # Extract relevant fields for each anomaly
        processed_anomalies = []
        
        for anomaly in filtered_data:
            processed_anomaly = {
                'id': anomaly.get('id', 'Unknown'),
                'title': anomaly.get('title', anomaly.get('name', 'Unknown')),
                'description': anomaly.get('description', 'No description'),
                'severity': anomaly.get('severity', 'Unknown'),
                'type': anomaly.get('type', anomaly.get('category', 'Unknown')),
                'status': anomaly.get('status', 'Unknown'),
                'device': anomaly.get('device', anomaly.get('deviceName', 'Unknown')),
                'timestamp': anomaly.get('timestamp', anomaly.get('createdAt', 'Unknown')),
                'last_updated': anomaly.get('lastUpdated', anomaly.get('updatedAt', 'Unknown')),
                'affected_component': anomaly.get('affectedComponent', anomaly.get('component', 'Unknown')),
                'recommendation': anomaly.get('recommendation', 'No recommendation available')
            }
            processed_anomalies.append(processed_anomaly)
        
        return processed_anomalies
    
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
        
        if key == 'severity':
            # Check severity levels (case insensitive)
            item_severity = item.get('severity', '').lower()
            return item_severity == value.lower()
        
        if key == 'type':
            # Check type or category
            return (
                (item.get('type', '').lower() == value.lower()) or
                (item.get('category', '').lower() == value.lower())
            )
        
        # Handle camelCase variations
        camel_case_key = ''.join([key[0].lower(), key[1:]])
        
        # Check if the key exists in the item
        if key in item:
            return str(item[key]).lower() == str(value).lower()
        elif camel_case_key in item:
            return str(item[camel_case_key]).lower() == str(value).lower()
        
        return False
    
    def _get_anomaly_stats(self, anomalies: List[Dict[str, Any]], count_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate statistics for anomalies
        
        Args:
            anomalies: List of processed anomalies
            count_data: Optional count data from API
            
        Returns:
            Dictionary of anomaly statistics
        """
        stats = {
            "total": len(anomalies),
            "by_severity": {},
            "by_type": {},
            "by_status": {},
            "by_device": {}
        }
        
        # Count by severity
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'Unknown')
            if severity not in stats["by_severity"]:
                stats["by_severity"][severity] = 0
            stats["by_severity"][severity] += 1
            
            # Count by type
            anomaly_type = anomaly.get('type', 'Unknown')
            if anomaly_type not in stats["by_type"]:
                stats["by_type"][anomaly_type] = 0
            stats["by_type"][anomaly_type] += 1
            
            # Count by status
            status = anomaly.get('status', 'Unknown')
            if status not in stats["by_status"]:
                stats["by_status"][status] = 0
            stats["by_status"][status] += 1
            
            # Count by device
            device = anomaly.get('device', 'Unknown')
            if device not in stats["by_device"]:
                stats["by_device"][device] = 0
            stats["by_device"][device] += 1
        
        # Add count data if available
        if count_data and isinstance(count_data, dict):
            stats["api_count"] = count_data
        
        # Incorporate API statistics if available (even when no anomaly details are present)
        if hasattr(self, '_api_stats') and self._api_stats:
            # If we have API stats but no anomalies, use the API stats for total count
            if not anomalies and 'totalAnomalyCount' in self._api_stats:
                stats['total_from_api'] = self._api_stats['totalAnomalyCount']
            
            # Add API stats to the response
            stats['api_stats'] = self._api_stats
            
            # If we have severity breakdown from API, add it
            if 'entries' in self._api_stats:
                severity_counts = {}
                for entry in self._api_stats['entries']:
                    if 'severity' in entry and 'anomalyCount' in entry:
                        severity_counts[entry['severity']] = entry['anomalyCount']
                if severity_counts:
                    stats['severity_counts'] = severity_counts
        
        return stats
    
    # Removed custom _make_api_call method - using base handler's method for real API calls
