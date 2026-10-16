# FabFlow — Agent-Powered Business Assistant

## Features
- PDF and TXT business knowledge loaded into the system prompt.
- Separately declared `PERSONA` and `INSTRUCTIONS`.
- LLM router (`GENERAL`, `LEAD`, `FEEDBACK`) runs before answering.
- Native OpenAI function/tool calling, with two MCP-style documented schemas.
- Multi-turn lead capture, feedback/unanswered-question JSONL logging.
- Gradio ChatInterface, plus optional before/after reflection prints.

## Project structure
```
business_bot/
  me/about_business.pdf
  me/business_summary.txt
  business_agent.ipynb      # Colab/Jupyter walkthrough and demo
  agent.py                  # shared agent engine and tool definitions
  app.py                    # Gradio launch
  requirements.txt
  .env.example             
  .gitignore
  tests/test_agent.py       # offline unit tests using mock responses
```

## Setup
Requires Python 3.10+ and an OpenAI API key. From the `business_bot` directory:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows: 
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env 
 # Windows: 
copy .env.example .env
# Edit .env to set OPENAI_API_KEY
python app.py
```
Open the local Gradio URL in your terminal. Alternatively, start Jupyter (`jupyter notebook`) and execute `business_agent.ipynb` top-to-bottom. The notebook imports the same `agent.py` implementation, so the UI and notebook stay in sync.

**Important:** This project needs a paid/usable API key and network access for actual LLM chat. Installing dependencies, creating `.env`, and testing live calls must be performed in your environment. The `.example` address is fictional.

## Tool specifications (MCP-style)
| Tool | Required parameters | Return | Side effect |
| --- | --- | --- | --- |
| `record_customer_interest` | `email: string`, `name: string`, `message: string` | `{status, message}` | Appends `logs/leads.jsonl` |
| `record_feedback` | `question: string` | `{status, message}` | Appends `logs/feedback.jsonl` |

Tool execution is performed by **Python after a model tool-call request**; the LLM itself cannot directly write logs. JSONL files contain customer-provided content, so treat as private and do not commit them.

## Quick checks
```bash
python -m compileall -q agent.py app.py
python -m unittest discover -s tests -v
```
The unit tests use mocks and do not require API tokens or network access.

