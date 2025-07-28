#!/bin/bash

# Default interval (in seconds)
interval=2

# Function to show usage
usage() {
    echo "Usage: $0 [-n interval] command"
    exit 1
}

# Parse options
while getopts ":n:" opt; do
    case $opt in
        n)
            interval="$OPTARG"
            if ! [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                echo "Invalid interval: must be a number"
                exit 1
            fi
            ;;
        \?)
            usage
            ;;
    esac
done

# Shift parsed options, remaining arguments are the command
shift $((OPTIND -1))

# Ensure there's a command to run
if [ $# -eq 0 ]; then
    usage
fi

# The command to execute
cmd="$@"

# Main loop
while true; do
    clear
    echo "Every ${interval}s: $cmd  $(date)"
    echo "----------------------------------------"
    eval "$cmd"
    sleep "$interval"
done
