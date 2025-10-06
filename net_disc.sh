#!/bin/bash

# File to store the gateway IP
GATEWAY_FILE="$HOME/.wsl_gateway"

# Check current default route
CURRENT_ROUTE=$(ip route | grep '^default')

if [ -z "$CURRENT_ROUTE" ]; then
    # No default route — restore it
    if [ -f "$GATEWAY_FILE" ]; then
        GATEWAY=$(cat "$GATEWAY_FILE")
        echo "Restoring default route via $GATEWAY..."
        sudo ip route add default via "$GATEWAY"
    else
        echo "No saved gateway found. Cannot restore default route."
    fi
else
    # Default route exists — remove it
    GATEWAY=$(echo "$CURRENT_ROUTE" | awk '{print $3}')
    echo "Removing default route via $GATEWAY..."
    echo "$GATEWAY" > "$GATEWAY_FILE"
    sudo ip route del default
fi