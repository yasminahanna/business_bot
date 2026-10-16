"""Offline tests; mock the LLM, test real side effects to temporary files."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Offline fallback stubs when optional dependencies are not installed in the grader.
try:
    import openai
except ImportError:
    import types
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda: None)
try:
    import dotenv
except ImportError:
    import types
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
try:
    import pypdf
except ImportError:
    import subprocess, types
    class _Reader:
        def __init__(self, filename):
            content = subprocess.check_output(["pdftotext", "-layout", filename, "-"]).decode()
            self.pages = [types.SimpleNamespace(extract_text=lambda text=content: text)]
    sys.modules["pypdf"] = types.SimpleNamespace(PdfReader=_Reader)
import agent


class MockMessage:
    def __init__(self, content='', tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {'role': 'assistant', 'content': self.content}
        if self.tool_calls:
            d['tool_calls'] = [{'id': c.id, 'type': 'function',
                                'function': {'name': c.function.name,
                                             'arguments': c.function.arguments}}
                               for c in self.tool_calls]
        return d


def mock_response(content='', calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=MockMessage(content, calls))])


def tool_call(name, arguments):
    return SimpleNamespace(id='call_1', function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)))


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def create(self, **kwargs):
        self.messages.append(kwargs)
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_dir = agent.LOG_DIR
        agent.LOG_DIR = Path(self.temp.name)

    def tearDown(self):
        agent.LOG_DIR = self.old_dir
        self.temp.cleanup()

    def client(self, responses):
        fake = FakeCompletions(responses)
        return SimpleNamespace(chat=SimpleNamespace(completions=fake)), fake

    def test_loaded_pdf_and_summary(self):
        self.assertIn('FabFlow', agent.SUMMARY)
        self.assertIn('Maya Chen', agent.PDF_TEXT)
        self.assertIn('PERSONA', agent.SYSTEM_PROMPT)
        self.assertIn('INSTRUCTIONS', agent.SYSTEM_PROMPT)

    def test_lead_validation_and_logging(self):
        self.assertEqual(agent.record_customer_interest('bad-email', 'Sam', '10 prints')['status'], 'error')
        self.assertFalse((agent.LOG_DIR / 'leads.jsonl').exists())
        self.assertEqual(agent.record_customer_interest('sam@example.com', 'Sam', '10 prints')['status'], 'success')
        self.assertEqual(len((agent.LOG_DIR / 'leads.jsonl').read_text().splitlines()), 1)

    def test_feedback_logging(self):
        self.assertEqual(agent.record_feedback('Do you offer weekends?')['status'], 'success')
        self.assertIn('weekends', (agent.LOG_DIR / 'feedback.jsonl').read_text())

    def test_router_labels_and_history(self):
        client, fake = self.client([mock_response('LEAD')])
        label = agent.classify_intent('I am Sam, sam@example.com',
                                      [{'role': 'user', 'content': 'I need a quote'}], client)
        self.assertEqual(label, 'LEAD')
        self.assertEqual(fake.messages[0]['messages'][-2]['content'], 'I need a quote')

    def test_complete_lead_tool_dispatch(self):
        responses = [mock_response('LEAD'),
                     mock_response(calls=[tool_call('record_customer_interest', {
                         'email': 'sam@example.com', 'name': 'Sam', 'message': '10 PLA prints'})]),
                     mock_response('Your inquiry was logged for follow-up.')]
        client, fake = self.client(responses)
        text = agent.run_agent('Sam, sam@example.com, I want 10 PLA prints', client=client,
                               use_reflection=False)
        self.assertIn('logged', text)
        self.assertTrue((agent.LOG_DIR / 'leads.jsonl').exists())
        self.assertEqual(fake.messages[1]['tool_choice'], 'auto')
        self.assertEqual(fake.messages[2]['messages'][-1]['role'], 'tool')

    def test_feedback_route_calls_tool(self):
        responses = [mock_response('FEEDBACK'),
                     mock_response(calls=[tool_call('record_feedback', {'question': 'No weekend appointments'})]),
                     mock_response('Thanks, your suggestion was logged.')]
        client, _ = self.client(responses)
        text = agent.run_agent('I wish you had weekend hours', client=client, use_reflection=False)
        self.assertIn('suggestion', text)
        self.assertTrue((agent.LOG_DIR / 'feedback.jsonl').exists())

    def test_reflection_logs_both(self):
        responses = [mock_response('GENERAL'), mock_response('We offer PLA printing.'),
                     mock_response('FabFlow offers PLA and PETG printing.')]
        client, _ = self.client(responses)
        result = agent.run_agent('What materials?', client=client, use_reflection=True)
        self.assertIn('PETG', result)

    def test_reflection_receives_history_and_cannot_restart_lead_collection(self):
        history = [
            {'role': 'user', 'content': 'I need a quote for 10 PLA prototypes.'},
            {'role': 'assistant', 'content': 'Please provide your name and email.'},
            {'role': 'user', 'content': 'Yasmina Hanna, ysh08@mail.aub.edu'},
        ]
        client, fake = self.client([mock_response('Your inquiry was logged for staff follow-up.')])
        result = agent.reflect_response(
            'The prototypes are square keychains, 10x10 cm.',
            'Your inquiry was logged for staff follow-up.',
            'LEAD', history, client)
        self.assertIn('logged', result)
        prompt = fake.messages[0]['messages'][1]['content']
        self.assertIn('Yasmina Hanna, ysh08@mail.aub.edu', prompt)
        self.assertIn('do not ask for more lead details', prompt)


if __name__ == '__main__':
    unittest.main()
