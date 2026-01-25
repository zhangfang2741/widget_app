#source .venv/bin/activate
#uv run <脚本名>
uv sync
uv export --format requirements-txt > requirements.txt
