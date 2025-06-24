"""
INTERFACE SILO HANDLERS
----------------------
This module contains all handler functions/classes for interface-related intents.
- All logic for querying, filtering, and formatting interface data from the Nexus Dashboard API lives here.
- Each handler should correspond to a specific intent or sub-intent defined in intents/interfaces.yaml.
- Filtering for 'down', 'critical', 'port-channel', etc. interfaces should be implemented here.
- This file should not handle non-interface API endpoints.

Intent mapping is managed via intents/interfaces.yaml.
"""
# Handler for Interface Statistics Intents (auto-generated)
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# === ENHANCED INTERFACE DATA EXTRACTOR & INTENT HANDLERS ===
def extract_interfaces_from_json(api_data):
    """
    Flattens ND API nested JSON to a list of interface dicts with fabric, node, site, and key interface fields.
    """
    interfaces = []
    for fabric in api_data.get("entries", []):
        fabric_name = fabric.get("fabricName")
        node_name = fabric.get("nodeName")
        site_name = fabric.get("siteName")
        for iface in fabric.get("entries", []):
            interfaces.append({
                "fabric": fabric_name,
                "node": node_name,
                "site": site_name,
                "name": iface.get("sourceNameLabel"),
                "adminStatus": iface.get("adminStatus"),
                "operStatus": iface.get("operStatus"),
                "anomalyScore": iface.get("anomalyScore"),
                "description": iface.get("interfaceDescription"),
                "type": iface.get("interfaceType"),
                "portSpeed": iface.get("portSpeed"),
                "domainId": iface.get("domainId"),
                "ip": iface.get("ipAddress"),
            })
    return interfaces


def get_down_interfaces(api_data, fabric=None):
    """
    Returns a human-readable list of operationally down interfaces, optionally filtered by fabric.
    """
    interfaces = extract_interfaces_from_json(api_data)
    if fabric:
        interfaces = [i for i in interfaces if i['fabric'] == fabric]
    down_ifaces = [i for i in interfaces if str(i.get("operStatus")).lower() == "down"]
    if not down_ifaces:
        return "All interfaces appear to be up."
    response_lines = [
        f"{i['name']} on {i['node']} ({i['fabric']}) is operationally down. Description: {i['description'] or 'N/A'}"
        for i in down_ifaces
    ]
    return "\n".join(response_lines)


def get_interface_details(api_data, iface_name):
    """
    Returns details for a specific interface by name.
    """
    interfaces = extract_interfaces_from_json(api_data)
    match = next((i for i in interfaces if i['name'] and i['name'].lower() == iface_name.lower()), None)
    if not match:
        return f"I couldn't find any details for interface {iface_name}."
    return (
        f"Interface {match['name']} on {match['node']} ({match['fabric']}):\n"
        f"- Admin Status: {match['adminStatus']}\n"
        f"- Oper Status: {match['operStatus']}\n"
        f"- Anomaly Score: {match['anomalyScore']}\n"
        f"- Port Speed: {match['portSpeed']} Mbps\n"
        f"- Description: {match['description'] or 'None'}\n"
        f"- IP: {match['ip'] or 'N/A'}"
    )


def get_interface_summary(api_data):
    """
    Returns a summary of interface health (total, down, high anomaly).
    """
    interfaces = extract_interfaces_from_json(api_data)
    total = len(interfaces)
    down = len([i for i in interfaces if str(i.get("operStatus")).lower() == "down"])
    critical = len([i for i in interfaces if i.get("anomalyScore", 0) >= 80])
    return (
        f"Total interfaces: {total}\n"
        f"Operationally down: {down}\n"
        f"High anomaly (>80): {critical}"
    )


def get_interface_summary_stats(message=None, context=None, *args, **kwargs):
    """
    Handler for interface_summary_stats intent. Extracts nd_client from context, calls ND /interfaces/summary, and returns conversational summary.
    """
    try:
        nd_client = None
        # Try context object (HandlerContext or dict)
        if context is not None:
            if hasattr(context, 'get_param'):
                nd_client = context.get_param('nd_client')
            elif isinstance(context, dict):
                nd_client = context.get('nd_client')
        # Try kwargs fallback
        if nd_client is None and 'nd_client' in kwargs:
            nd_client = kwargs['nd_client']
        if nd_client is None:
            return {
                "success": False,
                "response": "I couldn't connect to the Nexus Dashboard API to get interface statistics.",
                "data": {},
            }
        # Query the ND API for interface summary
        summary_json = nd_client.api_get("/interfaces/summary")
        if not summary_json or not isinstance(summary_json, dict):
            return {
                "success": False,
                "response": "I couldn't retrieve interface summary statistics from the Nexus Dashboard.",
                "data": {},
            }
        # Format the summary
        lines = []
        status_counts = summary_json.get('statusCounts', {})
        if status_counts:
            lines.append("Interface Health by Status:")
            for status, count in status_counts.items():
                lines.append(f"- {status.title()}: {count}")
        type_counts = summary_json.get('typeCounts', {})
        if type_counts:
            lines.append("\nBy Interface Type:")
            for t, count in type_counts.items():
                lines.append(f"- {t}: {count}")
        anomaly_buckets = summary_json.get('anomalyScoreBuckets', {})
        if anomaly_buckets:
            lines.append("\nBy Anomaly Score:")
            for bucket, count in anomaly_buckets.items():
                lines.append(f"- {bucket}: {count}")
        top_errors = summary_json.get('topInterfacesByErrors', [])
        if top_errors:
            lines.append("\nTop Interfaces by Errors:")
            for iface in top_errors:
                name = iface.get('name', 'Unknown')
                errors = iface.get('errorsTotal', 0)
                node = iface.get('nodeName', '')
                lines.append(f"- {name} on {node}: {errors} errors")
        top_anomaly = summary_json.get('topInterfacesByAnomaly', [])
        if top_anomaly:
            lines.append("\nTop Interfaces by Anomaly Score:")
            for iface in top_anomaly:
                name = iface.get('name', 'Unknown')
                score = iface.get('anomalyScore', 0)
                node = iface.get('nodeName', '')
                lines.append(f"- {name} on {node}: anomaly score {score}")
        if not lines:
            conversational_summary = "No summary statistics found in the response."
        else:
            conversational_summary = "\n".join(lines)
        return {
            "success": True,
            "response": conversational_summary,
            "data": summary_json,
        }
    except Exception as e:
        import traceback
        logger.error(f"[get_interface_summary_stats] Error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "response": f"I ran into a problem getting interface summary statistics: {e}",
            "data": {},
        }


def interface_summary_stats_handler(message, context=None, *args, **kwargs):
    """
    Handler for interface_summary_stats intent. Extracts nd_client from context, calls ND /interfaces/summary, and returns conversational summary.
    """
    try:
        nd_client = None
        # Try context object (HandlerContext or dict)
        if context is not None:
            if hasattr(context, 'get_param'):
                nd_client = context.get_param('nd_client')
            elif isinstance(context, dict):
                nd_client = context.get('nd_client')
        # Try kwargs fallback
        if nd_client is None and 'nd_client' in kwargs:
            nd_client = kwargs['nd_client']
        if nd_client is None:
            return {
                "success": False,
                "response": "I couldn't connect to the Nexus Dashboard API to get interface statistics.",
                "data": {},
            }
        # Query the ND API for interface summary
        summary_json = nd_client.api_get("/interfaces/summary")
        if not summary_json or not isinstance(summary_json, dict):
            return {
                "success": False,
                "response": "I couldn't retrieve interface summary statistics from the Nexus Dashboard.",
                "data": {},
            }
        # Use the original summary stats formatter, but avoid double-passing 'context'
        # Only pass message and nd_client as kwarg to avoid multiple values for 'context'
        conversational_summary_result = get_interface_summary_stats(message, nd_client=nd_client)
        # If the result is a dict (error or formatted), return it directly
        if isinstance(conversational_summary_result, dict):
            return conversational_summary_result
        # Otherwise, wrap as expected
        return {
            "success": True,
            "response": conversational_summary_result,
            "data": summary_json,
        }
    except Exception as e:
        import traceback
        logger.error(f"[interface_summary_stats_handler] Error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "response": f"I ran into a problem getting interface summary statistics: {e}",
            "data": {},
        }

# Modular filter dictionary for extensibility
FILTERS = {
    "down": lambda iface: str(iface.get('operStatus', '')).lower() == 'down',
    "critical": lambda iface: str(iface.get('anomalyLevel', '')).lower() == 'critical',
    "port-channel": lambda iface: str(iface.get('type', '')).lower() == 'vpc' or str(iface.get('name', '')).lower().startswith('port-channel'),
    "high-errors": lambda iface: iface.get('errorsTotal', 0) > 100,
}

def apply_filters(message, interfaces):
    message_lower = message.lower() if message else ""
    for keyword, fn in FILTERS.items():
        if keyword in message_lower:
            filtered = [iface for iface in interfaces if fn(iface)]
            return filtered, keyword
    return interfaces, None

def summarize_interface_health(interfaces):
    return {
        "total": len(interfaces),
        "down": len([i for i in interfaces if str(i.get("operStatus", "")).lower() == "down"]),
        "critical": len([i for i in interfaces if str(i.get("anomalyLevel", "")).lower() == "critical"]),
        "with_errors": len([i for i in interfaces if i.get("errorsTotal", 0) > 0]),
    }

def get_interface_status(message, nd_client=None, interface_name=None, *args, **kwargs):
    # Accept interface_name as explicit parameter, fallback to kwargs
    if interface_name is None:
        interface_name = kwargs.get('interface_name')
    if nd_client is None and 'context' in kwargs:
        context = kwargs['context']
        if hasattr(context, 'get_param'):
            nd_client = context.get_param('nd_client')
        elif isinstance(context, dict):
            nd_client = context.get('nd_client')
    # nd_client should now be set if available
    logger.info(f"[INTERFACE HANDLER] get_interface_status called with client: {nd_client}")
    """
    Get interface status information using direct NDI client access.
    
    Args:
        interface_name: Optional specific interface name to query
        message: User message that triggered this handler
        nd_client: Authenticated NDI client instance
        *args, **kwargs: Additional arguments
    
    Returns:
        List of interface information dictionaries
    """
    try:
        # Log that we're executing this handler
        logger.info(f"[INTERFACE HANDLER] get_interface_status called with client: {nd_client}")
        
        # Check client
        if not nd_client:
            logger.error("[INTERFACE HANDLER] No NDI client provided")
            return [{"error": "No NDI client available"}]
            
        # Try to extract interface name from message if not provided
        if message and isinstance(message, str) and not interface_name:
            import re
            match = re.search(r'(eth\d+/\d+|port-channel\d+|vfc\d+|mgmt\d+)', message.lower())
            if match:
                interface_name = match.group(1)
                logger.info(f"[INTERFACE HANDLER] Extracted interface name: {interface_name}")
        
        # Set up params for the API call
        params = {"statName": "interface", "count": 1500, "offset": 0, "sort": "-anomalyScore"}
        if interface_name and isinstance(interface_name, str):
            params["sourceNameLabel"] = interface_name
            
        # Get base endpoint path
        endpoint = "/sedgeapi/v1/cisco-nir/api/api/v1/protocols/details"
        
        # Use the NDI client's direct get method (already authenticated)
        logger.info(f"[INTERFACE HANDLER] Calling NDI API: {endpoint} with params: {params}")
        try:
            response = nd_client.get(endpoint, params=params)
        except Exception as e:
            logger.error(f"[INTERFACE HANDLER] Exception during API call: {str(e)}")
            return [{"error": f"API call failed: {str(e)}"}]

        # Log status code and raw response text for debugging
        status_code = getattr(response, 'status_code', None)
        raw_text = getattr(response, 'text', None)
        if status_code is not None:
            logger.info(f"[INTERFACE HANDLER] Raw response status: {status_code}")
        if raw_text is not None:
            logger.debug(f"[INTERFACE HANDLER] Raw response text: {raw_text[:500]}")

        # Check for non-200 status code
        if status_code is not None and status_code != 200:
            logger.error(f"[INTERFACE HANDLER] Bad status code: {status_code}")
            return [{"error": f"API returned status {status_code}"}]

        # Robust type checking and error handling for API response
        try:
            if hasattr(response, 'json') and callable(response.json):
                data = response.json()
            elif isinstance(response, dict):
                data = response
            elif isinstance(response, str):
                logger.error(f"[INTERFACE HANDLER] API returned string response: {response[:300]}")
                return [{"error": f"Unexpected string from API: {response[:100]}"}]
            else:
                logger.error(f"[INTERFACE HANDLER] Unknown response type: {type(response)}")
                return [{"error": f"Unknown response type: {type(response)}"}]
        except Exception as e:
            logger.error(f"[INTERFACE HANDLER] Error parsing API response: {str(e)}")
            return [{"error": f"API call failed: {str(e)}"}]

        if not isinstance(data, dict) or ("items" not in data and "entries" not in data):
            logger.error(f"[INTERFACE HANDLER] Unexpected API response structure: {data}")
            return [{"error": f"Unexpected API response structure", "raw": str(data)[:300]}]

        interfaces = data.get("items") if "items" in data else data.get("entries")
        if not interfaces:
            logger.info("[INTERFACE HANDLER] No interface data found in response")
            return [{"info": "No interface data found."}]

        # Apply modular filters
        filtered, filter_used = apply_filters(message, interfaces)
        logger.info(f"[INTERFACE HANDLER] Filter used: {filter_used if filter_used else 'none'}; {len(filtered)} interfaces after filtering")

        # Build result list with more fields
        result = []
        for iface in filtered:
            entry = {
                "name": iface.get("name"),
                "adminStatus": iface.get("adminStatus"),
                "operStatus": iface.get("operStatus"),
                "type": iface.get("type"),
                "mtu": iface.get("mtu"),
                "fabric": iface.get("fabricName", "unknown"),
                "crcErrors": next((c.get("value") for c in iface.get("entries", []) if c.get("counterName") == "crcErrors"), None),
                "inputErrors": next((c.get("value") for c in iface.get("entries", []) if c.get("counterName") == "inputErrors"), None),
                "rxRate": next((c.get("value") for c in iface.get("entries", []) if c.get("counterName") == "rxRate"), None),
                "txRate": next((c.get("value") for c in iface.get("entries", []) if c.get("counterName") == "txRate"), None),
                "errorsTotal": iface.get("errorsTotal", 0),
            }
            result.append(entry)

        # Add summary at the end for reporting
        summary = summarize_interface_health(result)
        logger.info(f"[INTERFACE HANDLER] Interface summary: {summary}")
        return {"interfaces": result, "summary": summary, "filter": filter_used}

        if not response:
            logger.error("[INTERFACE HANDLER] Empty response from NDI API")
            return [{"error": "No data returned from interface API"}]

        # Extract JSON data from response object
        try:
            # Check if response has a json method (standard requests Response object)
            if hasattr(response, 'json') and callable(response.json):
                response_data = response.json()
            # Check if response has text attribute to parse
            elif hasattr(response, 'text'):
                import json
                response_data = json.loads(response.text)
            # Response might already be parsed JSON
            elif isinstance(response, dict):
                response_data = response
            else:
                logger.error(f"[INTERFACE HANDLER] Unknown response type: {type(response)}")
                return [{"error": f"Cannot parse response of type {type(response)}"  }]
        except Exception as e:
            logger.error(f"[INTERFACE HANDLER] Error parsing response: {str(e)}")
            return [{"error": f"Error parsing API response: {str(e)}"  }]

        logger.info(f"[INTERFACE HANDLER] Parsed response data: {str(response_data)[:500]}...")

        # If the response is not what we expect, return a clear error
        if not isinstance(response_data, dict) or not response_data.get("entries"):
            logger.error(f"[INTERFACE HANDLER] Unexpected or empty response structure: {response_data}")
            return [{"error": f"Unexpected or empty response structure from API", "raw": response_data}]

        # Process response
        entries = []
        for fabric in response_data.get("entries", []):
            for iface in fabric.get("entries", []):
                entries.append({
                    "name": iface.get("sourceNameLabel"),
                    "adminStatus": iface.get("adminStatus"),
                    "operStatus": iface.get("operStatus"),
                    "anomalyScore": iface.get("anomalyScore"),
                    "description": iface.get("interfaceDescription"),
                    "type": iface.get("interfaceType"),
                    "portSpeed": iface.get("portSpeed"),
                    "errorsTotal": next((c.get("value") for c in iface.get("entries", []) if c.get("counterName") == "errorsTotal"), None),
                    "neighborInfo": iface.get("neighborInfo", {})  
                })
                
        logger.info(f"[INTERFACE HANDLER] Retrieved {len(entries)} interfaces")

        # --- FILTERING LOGIC BASED ON USER QUERY ---
        def filter_down(interfaces):
            return [iface for iface in interfaces if str(iface.get('operStatus', '')).lower() == 'down']
        def filter_critical(interfaces):
            return [iface for iface in interfaces if str(iface.get('anomalyLevel', '')).lower() == 'critical']
        def filter_portchannel(interfaces):
            return [iface for iface in interfaces if str(iface.get('type', '')).lower() == 'vpc' or str(iface.get('name', '')).lower().startswith('port-channel')]

        message_lower = str(message or '').lower()
        filtered_entries = entries
        filter_applied = None
        if 'down' in message_lower:
            filtered_entries = filter_down(entries)
            filter_applied = 'down'
        elif 'critical' in message_lower:
            filtered_entries = filter_critical(entries)
            filter_applied = 'critical'
        elif 'port-channel' in message_lower or 'portchannel' in message_lower:
            filtered_entries = filter_portchannel(entries)
            filter_applied = 'port-channel'

        logger.info(f"[INTERFACE HANDLER] Filter applied: {filter_applied or 'none'}, {len(filtered_entries)} interfaces returned")
        return filtered_entries
        
    except Exception as e:
        logger.error(f"[INTERFACE HANDLER] Error in get_interface_status: {str(e)}")
        logger.error(traceback.format_exc())
        return [{"error": f"Interface handler error: {str(e)}"}]


def get_interfaces_with_errors(message, context=None, *args, **kwargs):
    nd_client = None
    if context is not None:
        if hasattr(context, 'get_param'):
            nd_client = context.get_param('nd_client')
        elif isinstance(context, dict):
            nd_client = context.get('nd_client')
    if nd_client is None and 'nd_client' in kwargs:
        nd_client = kwargs['nd_client']
    logger.info(f"[INTERFACE HANDLER] get_interfaces_with_errors called with client: {nd_client}")

    """
    Get interfaces with error counts greater than zero.
    
    Args:
        message: User message that triggered this handler
        nd_client: Authenticated NDI client instance
        *args, **kwargs: Additional arguments
    
    Returns:
        List of interface information dictionaries with errors
    """
    try:
        logger.info("[INTERFACE HANDLER] get_interfaces_with_errors called")
        all_ifaces = get_interface_status(message, context=context, *args, **kwargs)
        # Filter for interfaces with errors
        error_ifaces = [iface for iface in all_ifaces if iface.get("errorsTotal", 0) > 0 and not iface.get("error")]
        logger.info(f"[INTERFACE HANDLER] Found {len(error_ifaces)} interfaces with errors")
        return error_ifaces or [{"message": "No interfaces with errors found"}]
    except Exception as e:
        logger.error(f"[INTERFACE HANDLER] Error in get_interfaces_with_errors: {str(e)}")
        logger.error(traceback.format_exc())
        return [{"error": f"Interface error handler error: {str(e)}"}]
