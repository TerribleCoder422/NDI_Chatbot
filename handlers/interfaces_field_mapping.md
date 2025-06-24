# Field Mapping for Interface Statistics Endpoint

## Endpoint
- **Path:** /sedgeapi/v1/cisco-nir/api/api/v1/protocols/details
- **Category:** protocols
- **Name:** get_protocols_details
- **Parameters:** startDate, endDate, siteName, siteGroupName, statName=interface, history, count, offset, sort

## Top-level Response Fields
- `totalResultsCount`, `totalItemsCount`, `offset`, `description`, `statName`, `entries`

## Per-Entry Fields (by fabric/node)
- `fabricName`: Fabric name
- `nodeName`: Node name (switch)
- `siteName`: Site name
- `entries`: List of interface objects

## Per-Interface Fields
- `sourceName`: Unique interface identifier
- `protocolName`: Should be "interface"
- `sourceNameLabel`: Human-friendly interface name (e.g. eth1/33)
- `neighborInfo`: L2/L3 neighbor details
- `adminStatus`: "up" or "down"
- `operStatus`: "up" or "down"
- `interfaceType`: "physical", "vpc", etc.
- `domainId`: Domain ID if present
- `interfaceUniqueId`: Unique ID
- `portSpeed`: Integer, speed in Mbps
- `ipAddress`, `ipv6Address`: IP addresses if present
- `interfaceDescription`: User description
- `anomalyScore`: Integer, anomaly score for interface
- `entries`: List of counters (see below)

## Per-Counter Fields (inside interface `entries`)
- `counterName`: e.g. "errorsTotal"
- `counterNameLabel`: Human label
- `counterType`, `units`, `value`, `trending`

## Example Questions Mapped to Fields
- "What is the status of eth1/33?" → `adminStatus`, `operStatus`, `sourceNameLabel`
- "Show all interfaces with errors" → `entries` (counterName == "errorsTotal" and value > 0)
- "Which interfaces are vpc?" → `interfaceType`
- "Show anomaly scores for all interfaces" → `anomalyScore`
- "Show description for eth1/33" → `interfaceDescription`
- "Show neighbors for port-channel4" → `neighborInfo`
- "What is the port speed of eth1/33?" → `portSpeed`
