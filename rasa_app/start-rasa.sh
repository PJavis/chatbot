#!/bin/sh
exec rasa run --enable-api --port 5005 --cors "*" --debug
